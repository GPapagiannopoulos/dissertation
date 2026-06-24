"""Application configuration: typed settings and EHR manifest loading."""

import functools
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Lazily loaded application configuration settings.

    Attributes:
        mimic4_ehr_data_path: Filesystem path to the MIMIC-IV EHR root
            read from 'MIMIC4_EHR_DATA_PATH'. Validated at startup.

    """

    model_config = {"env_file": Path(__file__).resolve().parent.parent.parent / ".env"}

    mimic4_ehr_data_path: Path = Field(..., frozen=True)

    @functools.cached_property
    def mimic4_ehr_manifest(self) -> Any:
        """Extract MIMIC-IV EHR manifest from YAML file.

        Returns:
            An indexable object containing MIMIC-IV EHR manifest data. Empty if
            file is empty

        Raises:
            ModuleNotFoundError: If the package isn't importable under the exact name
            FileNotFoundError: If the file doesn't exist under the constructed path
            yaml.YAMLError: If the manifest is not valid YAML

        """
        text = resources.files("thesis").joinpath("mimic4_ehr.yaml").read_text()

        return yaml.safe_load(text)

    @functools.cached_property
    def mimic4_ehr_tables(self) -> list[str]:
        """Table names declared in the EHR manifest.

        Raises:
            KeyError: MIMIC-IV EHR tables not declared in the manifest.

        """
        return list(self.mimic4_ehr_manifest["tables"].keys())


settings = Settings()
