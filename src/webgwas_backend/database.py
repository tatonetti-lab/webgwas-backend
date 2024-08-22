import pathlib

from sqlalchemy import Engine
from sqlmodel import create_engine

from webgwas_backend.config import settings
from webgwas_backend.models import SQLModel

engine = create_engine(settings.sqlite_db)


def db_exists(path: str = settings.sqlite_db) -> bool:
    return pathlib.Path(path.lstrip("sqlite:").lstrip("/")).exists()


def init_db(engine: Engine = engine) -> None:
    SQLModel.metadata.create_all(engine)
