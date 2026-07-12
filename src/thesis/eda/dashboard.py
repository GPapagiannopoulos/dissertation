"""Dashboard for EDA."""

from pathlib import Path

import plotly.express as px
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


@st.cache_resource
def load_global_event_frame():
    """Load, transform, and cache the MIMIC-IV dataset.

    This function is responsible for loading the MIMIC-IV dataset
    using the PyHealth MIMIC4Dataset class. The underlying dataframe
    is used for transformations and then cached for EDA via a Streamlit
    dashboard.
    """
    ds = MIMIC4Dataset(
        ehr_root=str(settings.mimic4_ehr_data_path),
        dev=settings.mimic4_ehr_dev_mode,
        ehr_tables=settings.mimic4_ehr_tables,
    )
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
    return lf.collect()


def get_source() -> PolarsEDASource:
    """Pass the cached dataset to the Adapter class."""
    return PolarsEDASource(load_global_event_frame())


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
