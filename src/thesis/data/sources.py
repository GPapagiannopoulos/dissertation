"""Adapter for the MIMIC4Dataset class from PyHealth."""

from pathlib import Path

import polars as pl
from polars.exceptions import InvalidOperationError

from thesis.constants import DTYPE_TO_POLARS_DTYPE_MAP


def cast_frame(lf: pl.LazyFrame, dtype_map: dict[str, str]) -> pl.LazyFrame:
    """Apply dtype casts declared in the manifest to a LazyFrame.

    Columns absent from the frame are silently skipped. String→String
    casts are no-ops and are omitted. Date columns use str.to_date()
    because Polars 1.x does not support cast(pl.Date) from String.

    Args:
        lf: The LazyFrame to cast.
        dtype_map: Mapping of column name to dtype string (e.g. "UInt", "Date").

    Returns:
        LazyFrame with the cast expressions applied.

    Raises:
        ValueError: If the LazyFrame has no fields
        ValueError: If the dtype map is empty
        KeyError: If the dtype str name is not found
        InvalidOperationError: If the values are unparseable as the new dtype
    """
    available = set(lf.collect_schema().names())
    if len(available) == 0:
        raise ValueError("LazyFrame contains no fields. Please review.")
    if len(dtype_map.items()) == 0:
        raise ValueError("Map contains no key:value pairs. Please review.")
    exprs: list[pl.Expr] = []
    for field, dtype_str in dtype_map.items():
        if field not in available:
            raise ValueError(
                f"{field} is not an available field. Please review the schema."
            )
        polars_dtype = DTYPE_TO_POLARS_DTYPE_MAP.get(dtype_str, None)
        if not polars_dtype:
            valid_polars_dtypes = (", ").join(list(DTYPE_TO_POLARS_DTYPE_MAP.keys()))
            raise KeyError(
                f"{dtype_str} is not a valid key. Please select one of: "
                f"{valid_polars_dtypes}"
            )
        if polars_dtype is pl.String:
            continue
        elif polars_dtype is pl.Date:
            exprs.append(pl.col(field).str.to_date(format="%Y-%m-%d", strict=True))
        elif polars_dtype is pl.Datetime:
            exprs.append(
                pl.col(field).str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=True)
            )
        elif polars_dtype is pl.Boolean:
            exprs.append(
                pl.col(field).cast(pl.Int8, strict=True).cast(polars_dtype, strict=True)
            )
        else:
            exprs.append(pl.col(field).cast(polars_dtype, strict=True))
    if not exprs:
        return lf
    return lf.with_columns(exprs)


def replace_mimic4_icd_codes(
    data_source: pl.LazyFrame, path_to_map: Path, event_type: str
) -> pl.LazyFrame:
    """Replaces the ICD codes in the EHR mimic_data with human-readable descriptions.

    Joins the MIMIC-IV EHR dataset to a frame containing the mapping of ICD codes
    to human-readable descriptions. Subsequently, drops ICD codes and versions.

    Args:
        data_source: The LazyFrame containing MIMIC-IV mimic_data
        path_to_map: A Path object to the mapping held in .csv form
        event_type: The event type for which we are replacing ICD codes

    Returns:
        pl.LazyFrame: LazyFrame where ICD-codes are replaced with
        human-readable descriptions.

    Raises:
        KeyError: If the mapping csv does not contain columns named icd_version
        and icd_code respectively.
    """
    mapping_df = pl.scan_csv(
        path_to_map, schema_overrides={"icd_version": pl.String, "icd_code": pl.String}
    )
    if "icd_code" not in mapping_df.collect_schema():
        raise KeyError('Mapping frame is missing "icd_code" field. Please review.')
    if "icd_version" not in mapping_df.collect_schema():
        raise KeyError('Mapping frame is missing "icd_version" field. Please review.')

    combined_source = data_source.join(
        mapping_df,
        how="left",
        left_on=[f"{event_type}/icd_version", f"{event_type}/icd_code"],
        right_on=["icd_version", "icd_code"],
        coalesce=True,
    )

    combined_source = combined_source.drop(
        [f"{event_type}/icd_version", f"{event_type}/icd_code"]
    )
    return combined_source.rename({"long_title": f"{event_type}/description"})


def replace_mimic4_non_icd_codes(
    data_source: pl.LazyFrame, path_to_map: Path, event_type: str
) -> pl.LazyFrame:
    """Maps non-ICD ID columns to human-readable descriptions."""
    mapping_df = pl.scan_csv(path_to_map, schema_overrides={"itemid": pl.String})
    if "itemid" not in mapping_df.collect_schema():
        raise KeyError("Mapping frame is missing itemid column. Please review.")

    combined_source = data_source.join(
        mapping_df,
        how="left",
        left_on=[f"{event_type}/itemid"],
        right_on="itemid",
        coalesce=True,
    )

    return combined_source.drop(f"{event_type}/itemid").rename(
        {"label": f"{event_type}/description"}
    )


def cleanse_float_values(
    data_source: pl.LazyFrame, target_cols: list[str]
) -> pl.LazyFrame:
    """Helper function for removing ranges and commas from floats.

    Ensures that columns loaded as strings via PyHealth are safe to cast
    into Floats. Because it replaces ranges with means, it is necessary to represent
    the final string as a float (e.g. '1' -> '1.0').

    Notes:
        An intermediate step casts to Float64. If working
        with bigger values than that the value might overflow.

    Args:
        data_source (pl.DataFrame): dataframe object with the
        data to cleanse
        target_cols (list[str]): name of the columns to cleanse

    Returns:
        pl.DataFrame: cleansed dataframe

    Raises:
        InvalidOperationError: if any of the target fields dtypes
        are not strings
    """
    expressions: list[pl.Expr] = []
    for col in target_cols:
        # Polars normally raises InvalidOperationError at collection
        # Guard is against downstream failure propagation
        if str(data_source.collect_schema()[col]) != "String":
            raise InvalidOperationError(
                f"InvalidOperationError: expected String type, got: "
                f"{data_source.collect_schema()[col]}"
            )
        expr = (
            pl.col(col)
            .str.replace_all(",", "", literal=True)
            .str.split("-")
            .cast(pl.List(pl.Float64))
            .list.mean()
            .cast(pl.String)
        )
        expressions.append(expr)

    return data_source.with_columns(expressions)


class PolarsEDASource:
    """EDA adapter over PyHealth's global event dataframe.

    PyHealth loads MIMIC-IV mimic_data from the directory to create a
    MIMIC4Dataset object powered by an underlying Polars
    dataframe. This adapter accepts a Polars dataframe and exposes
    methods that allow for exploratory mimic_data analysis. This way
    any mimic_data source can be plugged into the dashboard as long as it
    is a polars dataframe (e.g. non-EHR MIMIC-IV mimic_data loaded via PyHealth).
    """

    _PATIENT = "patient_id"
    _TYPE = "event_type"

    def __init__(self, events: pl.DataFrame):
        """Constructor for the PolarsEDASource class.

        Args:
            events (pl.DataFrame): dataframe around which
            to initialize the wrapper

        Raises:
            ValueError: If missing patient_id or event_type column
            ValueError: If patient_id or event_type column contains None values

        """
        for col in [self._PATIENT, self._TYPE]:
            if col not in events.columns:
                raise ValueError(
                    f"Missing '{col}' column. The DataFrame is invalid. Please review."
                )
            if None in events.select(col).to_series():
                raise ValueError(f"'None' value in '{col}' detected.")
        self._events = events

    def event_types(self) -> list[str]:
        """Return an alphabetically sorted list of event types.

        Returns:
            list[str]: an alphabetically sorted list of the unique values
            in the event_type column.

        """
        return self._events.select(self._TYPE).unique().to_series().sort().to_list()

    def n_events(self, event_type: str) -> int:
        """Return the number of rows where the event_type field matches event_type.

        Args:
            event_type(str): the name of the event type to filter for

        Returns:
            int: the number of rows after filtering for that event_type

        """
        return self._events.filter(pl.col(self._TYPE) == event_type).height

    def n_patients(self, event_type: str) -> int:
        """Return the number of distinct patients with a specific event type.

        Args:
            event_type(str): the name of the event type to filter for

        Returns:
            int: the number of distinct patients after filtering for that event_type

        """
        return (
            self._events.filter(pl.col(self._TYPE) == event_type)
            .select(self._PATIENT)
            .unique()
            .height
        )

    def fields(self, event_type: str) -> list[str]:
        """Return an alphabetically sorted list of dataframe field names.

        In PyHealth the fields belonging to an event_type are prefixed
        {event_type}/{field_name}.

        Args:
            event_type(str): the event_type whose attributes we filter for

        Returns:
            list[str]: an alphabetically sorted list of dataframe field names

        """
        prefix = f"{event_type}/"
        return sorted([c for c in self._events.columns if c.startswith(prefix)])

    def field_dtypes(self, event_type: str) -> pl.DataFrame:
        """Return a dataframe to display the field names and dtypes.

        Returns:
            pl.DataFrame: a Nx2 dataframe containing the attributes of
            an event type and their corresponding dtype.
        """
        valid_cols = self.fields(event_type)
        col_dtypes = [str(self._events.schema[c]) for c in valid_cols]

        return pl.DataFrame({"field": valid_cols, "dtype": col_dtypes})

    def describe_field(self, field_name: str) -> pl.DataFrame:
        """Return a dataframe with summary measures for a given field.

        Calculates descriptors for a field at the event type level. It first
        filters by event type, before selecting the target field. The summary
        statistics are calculated at event type level because records of different
        type are null, leading to skewing of event proportion.

        Args:
            field_name(str): the name of the field to filter for

        Returns:
            pd.DataFrame: a dataframe offering summary measures. If the field is
            numeric, it returns summary statistics as per the pl.Series.describe()
            method.Otherwise, it returns the proportion of each value in the field.

        """
        target_field = (
            self._events.filter(pl.col(self._TYPE) == field_name.split("/")[0])
            .select(field_name)
            .to_series()
        )
        if self._events.schema[field_name].is_numeric():
            return target_field.describe()
        else:
            return target_field.value_counts(normalize=True).sort(
                "proportion", descending=True
            )

    def preview_table(self, event_type: str, n_rows: int = 10) -> pl.DataFrame:
        """Returns the head of the dataframe."""
        table_fields = self.fields(event_type)
        return (
            self._events.select(table_fields)
            .filter(pl.sum_horizontal(pl.all().is_not_null()) > 0.6 * len(table_fields))
            .head(n_rows)
        )
