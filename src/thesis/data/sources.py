"""Adapter for the MIMIC4Dataset class from PyHealth."""

from pathlib import Path

import polars as pl
from polars.exceptions import ColumnNotFoundError, InvalidOperationError

from thesis.constants import DTYPE_TO_POLARS_DTYPE_MAP
from thesis.data.eda_source import EmptyHistError, MixedUnitsError, NumericSummary


def cast_frame(lf: pl.LazyFrame, dtype_map: dict[str, str]) -> pl.LazyFrame:
    """Apply dtype casts declared in the manifest to a LazyFrame.

    String→String casts are no-ops and are omitted.
    Date columns use str.to_date() because Polars 1.x does not support
    cast(pl.Date) from String.

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


def mimic4_add_descriptions_to_icd_codes(
    data_source: pl.LazyFrame, path_to_map: Path, event_type: str
) -> pl.LazyFrame:
    """Matches the ICD codes in the EHR mimic_data with their human-readable names.

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
    for field in ["icd_code", "icd_version"]:
        if field not in mapping_df.collect_schema():
            raise KeyError(f'Mapping frame is missing "{field}" field. Please review.')
        if f"{event_type}/{field}" not in data_source.collect_schema():
            raise KeyError(
                f'Target LazyFrame is missing "{field}" field. Please review.'
            )

    combined_source = data_source.join(
        mapping_df,
        how="left",
        left_on=[f"{event_type}/icd_version", f"{event_type}/icd_code"],
        right_on=["icd_version", "icd_code"],
        coalesce=True,
    )

    return combined_source.rename({"long_title": f"{event_type}/description"})


def replace_mimic4_non_icd_codes(
    data_source: pl.LazyFrame, path_to_map: Path, event_type: str
) -> pl.LazyFrame:
    """Maps non-ICD ID columns to human-readable descriptions."""
    mapping_df = pl.scan_csv(path_to_map, schema_overrides={"itemid": pl.String})
    for field in ["itemid", "label"]:
        if field not in mapping_df.collect_schema():
            raise KeyError(f"Mapping frame is missing '{field}' column. Please review.")

    if f"{event_type}/itemid" not in data_source.collect_schema():
        raise KeyError(
            f"Mapping frame is missing '{event_type}/itemid' column. Please review."
        )
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
    dtypes = data_source.collect_schema()
    for col in target_cols:
        # Polars normally raises InvalidOperationError at collection
        # Guard is against downstream failure propagation
        if str(dtypes[col]) != "String":
            raise InvalidOperationError(
                f"InvalidOperationError: expected String type, got: "
                f"{data_source.collect_schema()[col]}"
            )
        expr = (
            pl.col(col)
            .str.replace_all(",", "", literal=True)
            .str.replace_all(r"(\d)\s*-", "${1}|", literal=False)
            .str.extract_all(r"-?\d*\.?\d+")
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
    LazyFrame. This adapter accepts a Polars LazyFrame and exposes
    methods that allow for exploratory mimic_data analysis. This way
    any mimic_data source can be plugged into the dashboard as long as it
    is a polars LazyFrame (e.g. non-EHR MIMIC-IV mimic_data loaded via PyHealth).
    """

    _PATIENT = "patient_id"
    _TYPE = "event_type"

    def __init__(self, events: pl.LazyFrame):
        """Constructor for the PolarsEDASource class.

        Args:
            events (pl.LazyFrame): LazyFrame around which
            to initialize the wrapper

        Raises:
            ValueError: If missing patient_id or event_type column
            ValueError: If patient_id or event_type column contains None values

        """
        for col in [self._PATIENT, self._TYPE]:
            if col not in events.collect_schema():
                raise ValueError(
                    f"Missing '{col}' column. The DataFrame is invalid. Please review."
                )
        null_counts = events.select(
            pl.col(c).null_count() for c in [self._PATIENT, self._TYPE]
        ).collect(engine="streaming")
        for col in [self._PATIENT, self._TYPE]:
            if null_counts.get_column(col).item():
                raise ValueError(f"'None' value in '{col}' detected.")

        self._events = events
        self._schema = events.collect_schema()

    @classmethod
    def from_parquet(cls, path_to_parquet: Path) -> "PolarsEDASource":
        """Constructs an instance from a parquet path."""
        return cls(pl.scan_parquet(path_to_parquet, low_memory=True))

    def event_types(self) -> list[str]:
        """Return an alphabetically sorted list of event types.

        Returns:
            list[str]: an alphabetically sorted list of the unique values
            in the event_type column.

        """
        return (
            self._events.select(self._TYPE)
            .unique()
            .sort(self._TYPE)
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )

    def n_events(self, event_type: str) -> int:
        """Return the number of rows where the event_type field matches event_type.

        Args:
            event_type(str): the name of the event type to filter for

        Returns:
            int: the number of rows after filtering for that event_type

        """
        return (
            self._events.filter(pl.col(self._TYPE) == event_type)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )

    def n_patients(self, event_type: str) -> int:
        """Return the number of distinct patients with a specific event type.

        Args:
            event_type(str): the name of the event type to filter for

        Returns:
            int: the number of distinct patients after filtering for that event_type

        """
        return (
            self._events.filter(pl.col(self._TYPE) == event_type)
            .select(pl.col(self._PATIENT).n_unique())
            .collect(engine="streaming")
            .item()
        )

    def get_unique_field_values(
        self, target_field: str, filters: dict[str, str] | None = None
    ) -> pl.Series:
        """Returns unique values of a target field sorted alphabetically.

        Args:
            target_field(str): the name of the field to return
            filters(dict[str, str] | None): an optional dict of predicates
            to filter on in the format {'column' = 'value'}

        Returns:
            pl.Series: the unique values of the target field
        """
        return (
            self._events.filter(
                pl.col(self._TYPE) == target_field.split("/")[0],
                *[pl.col(col) == val for col, val in filters.items()]
                if filters
                else [pl.lit(True)],
            )
            .select(target_field)
            .unique()
            .drop_nulls()
            .sort(target_field)
            .collect(engine="streaming")
            .to_series()
        )

    def is_numeric(self, target_field: str) -> bool:
        """Checks whether the target field is numeric.

        Raises:
            ColumnNotFoundError: if the target field is not in the schema
        """
        if target_field not in self._schema.names():
            raise ColumnNotFoundError(f"Unable to find column '{target_field}'")
        return self._schema[target_field].is_numeric()

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
        return sorted([c for c in self._schema.names() if c.startswith(prefix)])

    def field_dtypes(self, event_type: str) -> pl.DataFrame:
        """Return a dataframe to display the field names and dtypes.

        Returns:
            pl.DataFrame: a Nx2 dataframe containing the attributes of
            an event type and their corresponding dtype.
        """
        valid_cols = self.fields(event_type)
        col_dtypes = [str(self._schema[c]) for c in valid_cols]

        return pl.DataFrame({"field": valid_cols, "dtype": col_dtypes})

    def _numeric_subset(
        self, target_field: str, filters: dict[str, str], uom_field: str | None
    ) -> pl.LazyFrame:
        """Internal helper for retrieving a filter aggregation of the lazyframe.

        In describing the numeric fields it is necessary to aggregate and filter
        data to make it readable to digestible for the dashboard. This is a separate
        concern from the actual summary - the separation of concerns allows for cleaner
        changes and testing of functionality.

        Args:
            target_field(str): the name of the field of interest
            filters (dict[str, str]): subsequent filters to be applied
                in the format {field_name: value_to_filter_for}
            uom_field (str): field containing the unit of measurement
        Returns:
            pl.LazyFrame: a df filtered according to specification
        """
        filtered_by_event_type = self._events.filter(
            pl.col(self._TYPE) == target_field.split("/")[0]
        )
        expr: list[pl.Expr] = (
            [pl.col(field) == value for field, value in filters.items()]
            if filters
            else [pl.lit(True)]
        )

        additional_filters = filtered_by_event_type.filter(expr)
        projection = [target_field, *filters.keys()]
        if uom_field is not None:
            projection.append(uom_field)
        return additional_filters.select(projection)

    def describe_categorical_field(self, field_name: str) -> pl.DataFrame:
        """Return a dataframe with summary measures for a given field.

        Returns the normalized value counts for each value in a given categorical field.

        Args:
            field_name(str): the name of the field to filter for

        Returns:
            pd.DataFrame: a dataframe containing the proportion of each value
            as a pl.Float64 value, sorted in descending order by proportion
            and ascending order by target_field name

        Raises:
            ColumnNotFoundError: if the target field does not exist in the schema
            ValueError: if the target field is a numeric column
        """
        if field_name not in self._schema.names():
            raise ColumnNotFoundError(f"Unable to find column '{field_name}'")
        if self.is_numeric(field_name):
            raise ValueError(f"'{field_name}' is not a categorical field.")

        return (
            self._events.filter(pl.col(self._TYPE) == field_name.split("/")[0])
            .group_by(field_name)
            .len("counts")
            .with_columns(
                (pl.col("counts") / pl.col("counts").sum()).alias("proportion")
            )
            .drop("counts")
            .sort(["proportion", field_name], descending=[True, False])
            .collect(engine="streaming")
        )

    def describe_numeric_field(
        self,
        target_field: str,
        filters: dict[str, str],
        uom_field: str | None = None,
    ) -> NumericSummary:
        """Return summary statistics for a numeric field scoped to one cohort.

        PyHealth loads data as an Entity-Attribute-Value (EAV) model, so a
        numeric column such as ``labevents/valuenum`` mixes many measurements.
        The ``filters`` pin the cohort (e.g. a single lab label) so that the
        summary is meaningful, and the unit is validated to be homogeneous.

        Args:
            target_field (str): the numeric field to summarize.
            filters (dict[str, str]): field:value equality filters pinning the
                cohort (e.g. {"labevents/label": "Red Blood Cells"}).
            uom_field (str | None): the field holding the unit of measurement,
                or None for fields with no unit.

        Returns:
            NumericSummary: the describe() statistics of the target field and
            the cohort's unit (None when uom_field is None or the cohort is
            empty).

        Raises:
            MixedUnitsError: if the filtered slice spans multiple units.
        """
        df_slice = self._numeric_subset(target_field, filters, uom_field).collect(
            engine="streaming"
        )
        unit: str | None = None
        if uom_field is not None:
            units = df_slice.get_column(uom_field).unique().drop_nulls().to_list()
            if len(units) > 1:
                raise MixedUnitsError(units)
            unit = units[0] if units else None
        stats = df_slice.select(target_field).describe()
        return NumericSummary(stats, unit)

    def numeric_histogram(
        self, target_field: str, filters: dict[str, str], bin_count: int = 30
    ) -> pl.DataFrame:
        """Return a count per bins for a filtered numeric field.

        Raises:
            EmptyHistError: if after filtering the field contains no data.
        """
        df_slice = self._numeric_subset(target_field, filters, None)
        hist = (
            df_slice.select(target_field)
            .collect(engine="streaming")
            .to_series()
            .hist(bin_count=bin_count)
        )
        if hist["count"].sum() == 0:
            raise EmptyHistError(target_field, filters)

        return hist

    def get_admission_timeline(self, hadm_id: str) -> pl.DataFrame:
        """Long-format events for one admission.

        Returns all events associated with a particular admission, one record
        per event.

        Raises:
            InvalidOperationError: if no hadm_id columns are present.
        """
        hadm = pl.coalesce(pl.selectors.ends_with("/hadm_id"))
        if "timestamp" not in self._schema:
            raise ValueError(
                "Missing unified 'timestamp' field. LazyFrame is malformed."
            )
        return (
            self._events.filter(hadm == hadm_id)
            .select(
                pl.col("patient_id"),
                hadm.alias("hadm_id"),
                pl.col("event_type"),
                pl.col("timestamp"),
            )
            .sort("timestamp")
            .collect(engine="streaming")
        )

    def preview_table(self, event_type: str, n_rows: int = 10) -> pl.DataFrame:
        """Returns the head of the lazyframe."""
        table_fields = self.fields(event_type)
        return (
            self._events.select(table_fields)
            .filter(pl.sum_horizontal(pl.all().is_not_null()) > 0.6 * len(table_fields))
            .head(n_rows)
            .collect(engine="streaming")
        )
