from __future__ import annotations

import json
from typing import Any

from dynaconf import Dynaconf
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class IndirectGWASSettings(BaseModel):
    batch_size: int = Field(10000, description="Batch size (in variants)")


class Settings(BaseSettings):
    dry_run: bool
    s3_bucket: str
    sqlite_db: str
    indirect_gwas: IndirectGWASSettings

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
