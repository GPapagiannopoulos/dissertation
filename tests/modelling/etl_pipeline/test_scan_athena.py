"""Testing suite for the Athena vocabulary export reader."""

from pathlib import Path

import polars as pl

from thesis.modelling.etl_pipeline.coverage import scan_athena


def _write(tmp_path: Path, rows: list[list[str]]) -> Path:
    """Writes a tab separated export, header first, exactly as Athena ships it."""
    path = tmp_path / "CONCEPT.csv"
    path.write_text("\n".join("\t".join(row) for row in rows) + "\n")
    return path


HEADER = ["concept_id", "concept_name", "vocabulary_id", "concept_code"]


def test_reads_tab_separated_columns(tmp_path: Path) -> None:
    """The exports are tab separated despite the csv extension."""
    path = _write(
        tmp_path,
        [HEADER, ["45591000", "Heart failure", "ICD10CM", "I50.84"]],
    )

    assert scan_athena(path).collect().to_dicts() == [
        {
            "concept_id": "45591000",
            "concept_name": "Heart failure",
            "vocabulary_id": "ICD10CM",
            "concept_code": "I50.84",
        }
    ]


def test_reads_every_column_as_string(tmp_path: Path) -> None:
    """Ids are only ever joined on, and the resolvers cast what they need."""
    path = _write(
        tmp_path, [HEADER, ["45591000", "Heart failure", "ICD10CM", "I50.84"]]
    )

    schema = scan_athena(path).collect_schema()

    assert set(schema.values()) == {pl.String}


def test_keeps_rows_a_stray_quote_would_swallow(tmp_path: Path) -> None:
    """The trap: with quoting on, one unbalanced quote eats the rest of the file."""
    path = _write(
        tmp_path,
        [
            HEADER,
            ["1", 'Salter-Harris Type II "physeal" fracture', "SNOMED", "a"],
            ["2", 'Barrett"s oesophagus', "SNOMED", "b"],
            ["3", "Fracture of femur", "SNOMED", "c"],
        ],
    )

    frame = scan_athena(path).collect()

    assert frame.height == 3
    assert frame["concept_name"][1] == 'Barrett"s oesophagus'


def test_keeps_commas_inside_a_field(tmp_path: Path) -> None:
    """Concept names carry commas, which the tab separator must not split on."""
    path = _write(
        tmp_path, [HEADER, ["1", "Poisoning by heroin, accidental", "SNOMED", "a"]]
    )

    frame = scan_athena(path).collect()

    assert frame.height == 1
    assert frame["concept_name"][0] == "Poisoning by heroin, accidental"


def test_returns_a_lazyframe(tmp_path: Path) -> None:
    """A 766 MB export is scanned, never read, so the resolvers can push work down."""
    path = _write(tmp_path, [HEADER, ["1", "Heart failure", "ICD10CM", "I50.84"]])

    assert isinstance(scan_athena(path), pl.LazyFrame)
