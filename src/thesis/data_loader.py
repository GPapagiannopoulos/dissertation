"""Simple scaffolding for accessing the data and confirming configuration is valid."""

from pyhealth.datasets import MIMIC4Dataset

from thesis.config import settings

if __name__ == "__main__":
    EHR_ROOT = settings.mimic4_ehr_data_path

    mimic4_dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT, dev=True, ehr_tables=settings.mimic4_ehr_tables
    )

    mimic4_dataset.load_data()

    patient_id = mimic4_dataset.unique_patient_ids[0]
    patient = mimic4_dataset.get_patient(patient_id)
    patient = patient.get_events(event_type="admissions", return_df=True)
