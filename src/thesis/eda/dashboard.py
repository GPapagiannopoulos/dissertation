"""Dashboard for EDA."""

from dotenv import load_dotenv

load_dotenv()

import plotly.express as px
import streamlit as st

from thesis.config import settings
from thesis.data.eda_source import EmptyHistError, MixedUnitsError
from thesis.data.sources import PolarsEDASource
from thesis.eda.filters import valid_fields
from thesis.feature_engineering.diagnoses_cache import load_enriched


@st.cache_resource
def get_source() -> PolarsEDASource:
    """Pass the cached dataset to the Adapter class."""
    return PolarsEDASource(load_enriched())


def _render_overview(src: PolarsEDASource, etype: str) -> None:
    """Counts and the field/dtype table for the selected event type."""
    c1, c2 = st.columns(2)
    c1.metric("Events", f"{src.n_events(etype):,}")
    c2.metric("Patients", f"{src.n_patients(etype):,}")
    st.subheader("Field Attributes")
    st.caption("Datetime and id fields are excluded from the field summary.")
    st.dataframe(src.field_dtypes(etype), width="stretch")


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
    st.dataframe(summary.stats, width="stretch")
    try:
        hist = src.numeric_histogram(field, filters, n_bin)
    except EmptyHistError as e:
        st.info(str(e))
    else:
        st.plotly_chart(px.bar(hist, x="breakpoint", y="count"), width="stretch")


def _render_categorical_summary(src: PolarsEDASource, field: str) -> None:
    """Value-count proportions + a top-20 bar chart for a categorical field."""
    counts = src.describe_categorical_field(field)
    st.dataframe(counts, width="stretch")
    st.plotly_chart(
        px.bar(counts.head(20), x=field, y="proportion"),
        width="stretch",
    )


def _render_timeline(src: PolarsEDASource, hadm_id: str | None) -> None:
    """Determines data to return for timeline."""
    if not hadm_id:
        st.info("Please select an admission to view.")
        return
    events = src.get_admission_timeline(hadm_id)
    if events.is_empty():
        st.info(f"No events for admission: {hadm_id}")
        return
    st.plotly_chart(
        px.scatter(events, "timestamp", "event_type", color="event_type"),
        width="stretch",
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
                        src.get_unique_field_values([filter_col], filters),
                    )
                uom = field_info.uom
            n_bins = st.slider("Number of bins", 5, 500, 20)

    overview_tab, summary_tab, preview_tab, timeline_tab = st.tabs(
        ["Overview", "Field summary", "Preview", "Timeline"]
    )
    with overview_tab:
        _render_overview(src, etype)
    with summary_tab:
        if numeric:
            _render_numeric_summary(src, field, filters, uom, n_bins)
        else:
            _render_categorical_summary(src, field)
    with preview_tab:
        st.dataframe(src.preview_table(etype), width="stretch")
    with timeline_tab:
        dx = src.get_unique_field_values(
            ["diagnoses_icd/icd_code", "diagnoses_icd/description"]
        )
        descriptions = dict(
            zip(
                dx["diagnoses_icd/icd_code"],
                dx["diagnoses_icd/description"],
                strict=True,
            )
        )

        code = st.selectbox(
            "Diagnosis (ICD)",
            descriptions,
            format_func=lambda x: f"{x} - {descriptions[x]}",
        )

        hadm_id = None
        if code:
            hadm_id = st.selectbox(
                "Admission ID",
                src.get_unique_field_values(
                    ["diagnoses_icd/hadm_id"], {"diagnoses_icd/icd_code": code}
                ),
            )
        _render_timeline(src, hadm_id)


# Guard necessary for Windows machines
if __name__ == "__main__":
    run_dashboard()
