"""Handwritten mappings for MIMIC codes that OMOP has no concept for.

This module is a bridge for MIMIC-native codes, that cannot reach OMOP.
and being hand-written it is the least authoritative layer, so it runs last.

Only three families are mapped, because they are the only ones whose targets
MOTOR kept. Insurance, marital status, language, hospital service, transfers and
both location families have OMOP concepts but none survive in MOTOR's
vocabulary, so mapping them would produce targets that tokenize to nothing.

Values deliberately left out are recorded in the UNMAPPED_* tuples below, so a
future reader can tell a considered omission from an oversight.

Caveat: our Athena download excludes the Race, Ethnicity and Gender vocabularies,
so the concept codes below follow the standard OMOP/OMB coding rather than a
lookup against local files. Re-pull those three (small) vocabularies to verify.
"""

# MIMIC records one race value per admission and conflates two OMOP axes: Race
# (the five OMB categories) and Ethnicity (Hispanic / not Hispanic). Values naming
# a Hispanic origin carry no race information at all, so they map across to the
# Ethnicity vocabulary rather than being forced into a race bucket.
RACE_MAP: dict[str, str] = {
    # Race/5, White
    "WHITE": "Race/5",
    "WHITE - OTHER EUROPEAN": "Race/5",
    "WHITE - RUSSIAN": "Race/5",
    "WHITE - EASTERN EUROPEAN": "Race/5",
    "WHITE - BRAZILIAN": "Race/5",
    # Race/3, Black or African American
    "BLACK/AFRICAN AMERICAN": "Race/3",
    "BLACK/CAPE VERDEAN": "Race/3",
    "BLACK/CARIBBEAN ISLAND": "Race/3",
    "BLACK/AFRICAN": "Race/3",
    # Race/2, Asian
    "ASIAN": "Race/2",
    "ASIAN - CHINESE": "Race/2",
    "ASIAN - SOUTH EAST ASIAN": "Race/2",
    "ASIAN - ASIAN INDIAN": "Race/2",
    "ASIAN - KOREAN": "Race/2",
    # Race/1 and Race/4
    "AMERICAN INDIAN/ALASKA NATIVE": "Race/1",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Race/4",
    # Ethnicity, not race: these values say nothing about which race box applies
    "HISPANIC OR LATINO": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - PUERTO RICAN": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - DOMINICAN": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - GUATEMALAN": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - SALVADORAN": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - COLUMBIAN": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - MEXICAN": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - HONDURAN": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - CUBAN": "Ethnicity/Hispanic",
    "HISPANIC/LATINO - CENTRAL AMERICAN": "Ethnicity/Hispanic",
}

# Recording an absence of information as a race would be fabrication, and OMOP's
# five categories have no bucket for a mixed or purely geographic answer. The last
# three are self-identifications no OMB category cleanly receives, so they are left
# alone rather than inferred from a label (decided 2026-07-28).
UNMAPPED_RACE: tuple[str, ...] = (
    "OTHER",  # 19,788 events
    "UNKNOWN",  # 13,870
    "UNABLE TO OBTAIN",  # 3,478
    "PATIENT DECLINED TO ANSWER",  # 2,162
    "PORTUGUESE",  # 2,082, OMB reads it as White, MIMIC records it apart
    "SOUTH AMERICAN",  # 774, geographic; Brazil and Guyana are not Hispanic
    "MULTIPLE RACE/ETHNICITY",  # 596, no OMB category covers it
)

GENDER_MAP: dict[str, str] = {
    "M": "Gender/M",
    "F": "Gender/F",
}

# MIMIC's admission type mixes the route in (emergency, direct, elective) with the
# billing status (observation vs admitted). OMOP's visit concepts encode the
# setting, so each value is placed by where the patient actually was.
ADMISSION_MAP: dict[str, str] = {
    "EW EMER.": "Visit/ERIP",
    "DIRECT EMER.": "Visit/IP",
    "URGENT": "Visit/IP",
    "ELECTIVE": "Visit/IP",
    "SURGICAL SAME DAY ADMISSION": "Visit/IP",
    "OBSERVATION ADMIT": "Visit/IP",
    "EU OBSERVATION": "Visit/ER",
    "DIRECT OBSERVATION": "Visit/OP",
    "AMBULATORY OBSERVATION": "Visit/OP",
}

# Families whose OMOP concepts exist but which MOTOR dropped from its vocabulary.
# Mapping them would emit targets that tokenize to nothing.
UNMAPPED_FAMILIES: tuple[str, ...] = (
    "MIMIC_IV_Admission_Location",
    "MIMIC_IV_Discharge_Location",
    "MIMIC_IV_Insurance",
    "MIMIC_IV_Language",
    "MIMIC_IV_Marital_Status",
    "MIMIC_IV_Service",
    "MIMIC_IV_Transfer",
)

MANUAL_CONCEPT_MAP: dict[str, str] = {
    f"{family}/{value}": target
    for family, mapping in (
        ("MIMIC_IV_Gender", GENDER_MAP),
        ("MIMIC_IV_Race", RACE_MAP),
        ("MIMIC_IV_Admission", ADMISSION_MAP),
    )
    for value, target in mapping.items()
}
