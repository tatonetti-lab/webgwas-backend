from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from dynaconf import Dynaconf
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from rich.logging import RichHandler

logger = logging.getLogger(__name__)


class IndirectGWASSettings(BaseModel):
    pass


class Settings(BaseSettings):
    log_level: (
        Literal["DEBUG"] | Literal["INFO"] | Literal["WARNING"] | Literal["ERROR"]
    )
    dry_run: bool
    s3_bucket: str
    s3_result_path: str
    sqlite_db: str
    indirect_gwas: IndirectGWASSettings
    n_workers: int
    fit_quality_file: Path
    root_data_directory: Path
    cache_capacity: int

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


def init_logging() -> None:
    match settings.log_level:
        case "DEBUG":
            level = logging.DEBUG
        case "INFO":
            level = logging.INFO
        case "WARNING":
            level = logging.WARNING
        case "ERROR":
            level = logging.ERROR
    logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler()])
    logger.debug(f"Settings: {settings}")
