from __future__ import annotations

import json
from typing import Any

import psutil
from dynaconf import Dynaconf
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class IndirectGWASSettings(BaseModel):
    num_threads: int = Field(psutil.cpu_count(), description="Number of threads")
    chunk_size: int = Field(1000000, description="Chunk size (in variants)")
    capacity: int = Field(25, description="Capacity (in phenotypes)")
    compress: bool = Field(True, description="Compress output (zstd)")
    quiet: bool = Field(False, description="Quiet mode")


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
