from __future__ import annotations

import json
from typing import Any

import psutil
from pydantic import BaseModel, DirectoryPath, Field
from pydantic_settings import BaseSettings


class IndirectGWASSettings(BaseModel):
    num_threads: int = Field(psutil.cpu_count(), description="Number of threads")
    chunk_size: int = Field(1000000, description="Chunk size (in variants)")
    capacity: int = Field(25, description="Capacity (in phenotypes)")
    compress: bool = Field(True, description="Compress output (zstd)")
    quiet: bool = Field(False, description="Quiet mode")


class Settings(BaseSettings):
    lru_cache_size: int = 100
    s3_bucket: str = "webgwas"
    indirect_gwas: IndirectGWASSettings
    cohort_paths: list[DirectoryPath]

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> Settings:
        return cls.model_validate(json_data)

    @classmethod
    def from_json_file(cls, json_file: str) -> Settings:
        with open(json_file) as f:
            return cls.from_json(json.load(f))
