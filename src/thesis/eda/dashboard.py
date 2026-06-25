"""Dashboard for EDA."""

import streamlit as st
from pyhealth.datasets import MIMIC4Dataset

from thesis.config import settings
from thesis.data.sources import from_pyhealth


@st.cache_resource
def load_source():
    """Load the MIMIC-IV dataset."""
    ds = MIMIC4Dataset(
        ehr_root=settings.mimic4_ehr_data_path,
        dev=True,
        ehr_tables=settings.mimic4_ehr_tables,
    )
    ds.load_data()
    return from_pyhealth(ds)


def run_dashboard():
    """Load dataset and run the dashboard."""
    src = load_source()
    st.title("MIMIC-IV — structure explorer")

    etype = st.selectbox("Event type", src.event_types())
    c1, c2 = st.columns(2)
    c1.metric("Events", f"{src.n_events(etype):,}")
    c2.metric("Patients", f"{src.n_patients(etype):,}")

    st.subheader("Field Attributes")
    st.dataframe(src.fields(etype))

    ftype = st.selectbox("Field", src.fields(etype))
    st.subheader("Field Summary")
    st.dataframe(src.describe_field(etype, ftype))


# Guard necessary for Windows machines
if __name__ == "__main__":
    run_dashboard()
