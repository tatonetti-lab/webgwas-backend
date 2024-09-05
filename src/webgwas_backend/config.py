from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dynaconf import Dynaconf
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class IndirectGWASSettings(BaseModel):
    pass


class Settings(BaseSettings):
    dry_run: bool
    s3_bucket: str
    s3_result_path: str
    sqlite_db: str
    indirect_gwas: IndirectGWASSettings
    n_workers: int
    fit_quality_file: Path
    root_data_directory: Path

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> Settings:
        return cls.model_validate(json_data)

    @classmethod
    def from_json_file(cls, json_file: str) -> Settings:
        with open(json_file) as f:
            return cls.from_json(json.load(f))

    @classmethod
    def from_dynaconf(cls) -> Settings:
        dynaconf_settings = Dynaconf(settings_files=["settings.toml", ".secrets.toml"])
        return cls.model_validate(dynaconf_settings, from_attributes=True)


settings = Settings.from_dynaconf()
