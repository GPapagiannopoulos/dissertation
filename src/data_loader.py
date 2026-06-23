import os
from pathlib import Path

from dotenv import load_dotenv
from pyhealth.datasets import MIMIC4Dataset

if __name__ == "__main__":
    load_dotenv()
    EHR_ROOT = Path(os.getenv("MIMIC_IV_EHR_DATA"))
    mimic4_dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        dev=True,
        ehr_tables=["patients", "admissions", "diagnoses_icd"],
    )

    mimic4_dataset.load_data()

    patient_id = mimic4_dataset.unique_patient_ids[0]
    patient = mimic4_dataset.get_patient(patient_id)
    patient = patient.get_events(event_type="admissions", return_df=True)
    print(patient[[s.name for s in patient if not (s.null_count() == patient.height)]])
