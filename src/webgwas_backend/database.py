import pathlib

from sqlmodel import create_engine

from webgwas_backend.config import settings
from webgwas_backend.models import SQLModel

engine = create_engine(settings.sqlite_db)


def db_exists() -> bool:
    return pathlib.Path(settings.sqlite_db.lstrip("sqlite:").lstrip("/")).exists()


def init_db():
    SQLModel.metadata.create_all(engine)
