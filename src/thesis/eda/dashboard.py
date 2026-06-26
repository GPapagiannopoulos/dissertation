"""Dashboard for EDA."""

import streamlit as st
from pyhealth.datasets import MIMIC4Dataset

from thesis.config import settings
from thesis.data.sources import PolarsEDASource, cast_frame


@st.cache_resource
def load_global_event_frame():
    """Cache the MIMIC-IV dataset."""
    ds = MIMIC4Dataset(
        ehr_root=str(settings.mimic4_ehr_data_path),
        dev=True,
        ehr_tables=settings.mimic4_ehr_tables,
    )
    lf = cast_frame(ds.global_event_df, settings.mimic4_ehr_dtype_mapping)
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

    ftype = st.selectbox("Field", src.fields(etype))
    st.subheader("Field Summary")
    st.dataframe(src.describe_field(ftype))

    st.subheader(f"{etype} Preview")
    st.dataframe(src.preview_table(etype))


# Guard necessary for Windows machines
if __name__ == "__main__":
    run_dashboard()
