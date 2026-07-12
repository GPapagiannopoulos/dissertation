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
        dev=True,
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


def run_dashboard():
    """Load dataset and run the dashboard."""
    src = get_source()
    st.title("MIMIC-IV — structure explorer")

    etype = st.selectbox("Event type", src.event_types())
    c1, c2 = st.columns(2)
    c1.metric("Events", f"{src.n_events(etype):,}")
    c2.metric("Patients", f"{src.n_patients(etype):,}")

    st.subheader("Field Attributes")
    st.dataframe(src.field_dtypes(etype))

    st.subheader("Field Summary")
    st.text("The summary below excludes datetime and id fields.")
    ftype = st.selectbox("Field", valid_fields(src, etype))

    if src.is_numeric(ftype):
        field_info = settings.mimic4_ehr_eav_fields.get(ftype)
        filter_values: dict[str, str] = {}
        uom = None
        if field_info is not None:
            for filter_col in field_info.filters:
                filter_values[filter_col] = st.selectbox(
                    filter_col,
                    src.get_unique_field_values(filter_col, filter_values),
                )
            uom = field_info.uom
        try:
            summary = src.describe_numeric_field(ftype, filter_values, uom)
        except MixedUnitsError as e:
            st.error(e)
            st.stop()
        if summary.unit:
            st.caption(f"Unit: {summary.unit}")
        st.dataframe(summary.stats)
        try:
            n_bin = st.slider("Number of bins", min_value=5, max_value=50)
            hist = src.numeric_histogram(ftype, filter_values, n_bin)
        except EmptyHistError as e:
            st.info(str(e))
        else:
            st.plotly_chart(
                px.bar(hist, x="breakpoint", y="count"), use_container_width=True
            )
    else:
        counts = src.describe_categorical_field(ftype)
        st.dataframe(counts)
        st.plotly_chart(
            px.bar(counts.head(20), x=ftype, y="proportion"),
            use_container_width=True,
        )

    st.subheader(f"{etype} Preview")
    st.dataframe(src.preview_table(etype))


# Guard necessary for Windows machines
if __name__ == "__main__":
    run_dashboard()
