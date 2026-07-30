"""Module for extracting patients with a positive AKI diagnosis as per criteria."""

from thesis.feature_engineering.driver import driver


def identify_surviving_aki_admissions():
    """Identifies the admissions with valid tokens post normalization.

    The process of normalizing MIMIC-IV data resulted into dropping of
    events that could not be mapped to any valid MOTOR token. For positive
    diagnosis events to be useful they need surviving events.
    """
    diagnoses = driver()
    print(diagnoses.collect_schema().names())


if __name__ == "__main__":
    identify_surviving_aki_admissions()
