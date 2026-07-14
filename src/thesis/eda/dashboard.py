"""Dashboard for EDA."""

import json
from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st
from pyhealth.datasets import MIMIC4Dataset

from thesis.config import settings
from thesis.data.eda_source import EmptyHistError, MixedUnitsError
from thesis.data.sources import (
    PolarsEDASource,
    cast_frame,
    cleanse_float_values,
    mimic4_add_descriptions_to_icd_codes,
    replace_mimic4_non_icd_codes,
)
from thesis.eda.filters import valid_fields

from .cache import _fingerprint


def build_event_pipeline(ds: MIMIC4Dataset) -> pl.LazyFrame:
    """Lazily generates a transformation pipeline schema.

    Args:
        ds (MIMIC4Dataset): PyHealth's native MIMIC4Dataset loader object.

    Returns:
        LazyFrame: a lf containing the transformed data
    """
    float_fields = [
        col
        for col, dtype in settings.mimic4_ehr_dtype_mapping.items()
        if dtype == "Float"
    ]
    cleansed_df = cleanse_float_values(ds.global_event_df, float_fields)
    lf = cast_frame(cleansed_df, settings.mimic4_ehr_dtype_mapping)
    event_type_icd_maps: list[tuple[str, Path]] = [
        ("procedures_icd", settings.mimic4_ehr_d_icd_procedures),
        ("diagnoses_icd", settings.mimic4_ehr_d_icd_diagnoses),
    ]

    for event_type, mapping in event_type_icd_maps:
        lf = mimic4_add_descriptions_to_icd_codes(lf, mapping, event_type)

    lf = replace_mimic4_non_icd_codes(lf, settings.mimic4_ehr_d_labitems, "labevents")
    return lf


def sink_global_event_frame(lf: pl.LazyFrame, out: Path) -> None:
    """Sink the transformed MIMIC4Dataset into a parquet file.

    Uses the streaming engine to sink a parquet file. The file is marked
    .tmp until the sinking completes. If the process crashes then the file
    is clearly marked as .tmp and won't be used downstream.

    Args:
        lf (pl.LazyFrame): the lf to be sunk into a parquet file
        out (Path): path at which to sink the file.
    """
    tmp = out.with_name(out.name + ".tmp")
    lf.sink_parquet(tmp)
    tmp.rename(out)


def ensure_event_cache() -> Path:
    """Confirms whether a cached ds already exists."""
    ds = MIMIC4Dataset(
        ehr_root=settings.mimic4_ehr_data_path,
        dev=settings.mimic4_ehr_dev_mode,
        ehr_tables=settings.mimic4_ehr_tables,
    )
    cache_root = ds.cache_dir / "thesis_eda"
    parquet = cache_root / "events.parquet"
    sidecar = cache_root / "events.fingerprint.json"

    if parquet.exists() and sidecar.exists():
        if json.loads(sidecar.read_text()) == _fingerprint(ds.cache_dir):
            return parquet

    cache_root.mkdir(parents=True, exist_ok=True)
    sidecar.unlink(missing_ok=True)
    sink_global_event_frame(build_event_pipeline(ds), parquet)
    sidecar.write_text(json.dumps(_fingerprint(ds.cache_dir)))
    return parquet


@st.cache_resource
def get_source() -> PolarsEDASource:
    """Pass the cached dataset to the Adapter class."""
    return PolarsEDASource.from_parquet(ensure_event_cache())


def _render_overview(src: PolarsEDASource, etype: str) -> None:
    """Counts and the field/dtype table for the selected event type."""
    c1, c2 = st.columns(2)
    c1.metric("Events", f"{src.n_events(etype):,}")
    c2.metric("Patients", f"{src.n_patients(etype):,}")
    st.subheader("Field Attributes")
    st.caption("Datetime and id fields are excluded from the field summary.")
    st.dataframe(src.field_dtypes(etype), use_container_width=True)


def _render_numeric_summary(
    src: PolarsEDASource,
    field: str,
    filters: dict[str, str],
    uom: str | None,
    n_bin: int,
) -> None:
    """Stats and histogram for a numeric field's cohort."""
    try:
        summary = src.describe_numeric_field(field, filters, uom)
    except MixedUnitsError as e:
        st.error(e)
        return
    if summary.unit:
        st.caption(f"Unit: {summary.unit}")
    st.dataframe(summary.stats, use_container_width=True)
    try:
        hist = src.numeric_histogram(field, filters, n_bin)
    except EmptyHistError as e:
        st.info(str(e))
    else:
        st.plotly_chart(
            px.bar(hist, x="breakpoint", y="count"), use_container_width=True
        )


def _render_categorical_summary(src: PolarsEDASource, field: str) -> None:
    """Value-count proportions + a top-20 bar chart for a categorical field."""
    counts = src.describe_categorical_field(field)
    st.dataframe(counts, use_container_width=True)
    st.plotly_chart(
        px.bar(counts.head(20), x=field, y="proportion"),
        use_container_width=True,
    )


def run_dashboard():
    """Load dataset and run the dashboard."""
    st.set_page_config(page_title="MIMIC-IV explorer", layout="wide")
    src = get_source()
    st.title("MIMIC-IV — structure explorer")

    with st.sidebar:
        st.header("Controls")
        etype = st.selectbox("Event type", src.event_types())
        field = st.selectbox("Field", valid_fields(src, etype))
        numeric = src.is_numeric(field)
        filters: dict[str, str] = {}
        uom: str | None = None
        n_bins = 20
        if numeric:
            field_info = settings.mimic4_ehr_eav_fields.get(field)
            if field_info is not None:
                for filter_col in field_info.filters:
                    filters[filter_col] = st.selectbox(
                        filter_col,
                        src.get_unique_field_values(filter_col, filters),
                    )
                uom = field_info.uom
            n_bins = st.slider("Number of bins", 5, 50, 20)

    overview_tab, summary_tab, preview_tab = st.tabs(
        ["Overview", "Field summary", "Preview"]
    )
    with overview_tab:
        _render_overview(src, etype)
    with summary_tab:
        if numeric:
            _render_numeric_summary(src, field, filters, uom, n_bins)
        else:
            _render_categorical_summary(src, field)
    with preview_tab:
        st.dataframe(src.preview_table(etype), use_container_width=True)


# Guard necessary for Windows machines
if __name__ == "__main__":
    run_dashboard()
