from typing import Type

import psutil
from pydantic import BaseModel, DirectoryPath, Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class IndirectGWASSettings(BaseModel):
    num_covar: int = Field(12, description="Number of covariates")
    num_threads: int = Field(psutil.cpu_count(), description="Number of threads")
    chunk_size: int = Field(1000000, description="Chunk size (in variants)")
    capacity: int = Field(25, description="Capacity (in phenotypes)")
    compress: bool = Field(True, description="Compress output (zstd)")
    quiet: bool = Field(False, description="Quiet mode")


class DataDirectorySettings(BaseModel):
    data_path: DirectoryPath
    cohorts: list[str]
    cohort_to_features: dict[str, list[str]]
    cohort_to_data: dict[str, str]
    cohort_to_covariance: dict[str, str]
    cohort_to_gwas: dict[str, list[str]]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        json_file="settings.json", json_file_encoding="utf-8"
    )

    lru_cache_size: int = 100
    s3_bucket: str = "webgwas-results"
    indirect_gwas: IndirectGWASSettings
    data_client: DataDirectorySettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            JsonConfigSettingsSource(settings_cls),
        )
