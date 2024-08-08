from sqlmodel import create_engine

from webgwas_backend.config import settings
from webgwas_backend.models import SQLModel

engine = create_engine(settings.sqlite_db)


def init_db():
    SQLModel.metadata.create_all(engine)
