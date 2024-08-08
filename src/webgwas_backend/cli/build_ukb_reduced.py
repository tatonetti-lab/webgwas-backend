import json
import pathlib

import polars as pl
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from webgwas_backend.config import settings
from webgwas_backend.database import engine, init_db
from webgwas_backend.models import Cohort, Feature

CONCEPT_PATH = "/Users/zietzm/Documents/projects/webgwas-server/etl/data/CONCEPT.csv"
METADATA_PATH = (
    "/Users/zietzm/Documents/projects/webgwas-server/webgwas-deploy/data/"
    "ukb_wb_100k_reduced_anon/metadata.json"
)


def load_icd_codes_athena() -> pl.DataFrame:
    return (
        pl.read_csv(CONCEPT_PATH, separator="\t")
        .filter(
            pl.col("vocabulary_id").eq("ICD10CM")
            & pl.col("concept_code").str.len_chars().eq(3)
        )
        .select(name="concept_name", code="concept_code", type=pl.lit("BOOL"))
    )


def make_cohort() -> Cohort:
    cohort = Cohort(
        name="UKB-WB-100k-reduced-anon",
        root_directory=str(settings.cohort_paths[-1]),
    )
    with Session(engine) as session:
        session.add(cohort)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            cohort = session.exec(
                select(Cohort).where(Cohort.name == cohort.name)
            ).one()
        else:
            session.refresh(cohort)
    return cohort


def gather_icd_codes() -> pl.DataFrame:
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return load_icd_codes_athena().filter(
        pl.col("code").is_in(set(metadata["feature_names"]))
    )


def insert_icd_codes(cohort: Cohort, df: pl.DataFrame) -> None:
    features = [
        Feature(cohort=cohort, code=code, name=name, type=type)
        for code, name, type in zip(df["code"], df["name"], df["type"])
    ]
    features = sorted(features, key=lambda f: f.name)
    with Session(engine) as session:
        session.add_all(features)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            features = session.exec(
                select(Feature).where(Feature.cohort == cohort)
            ).all()
            features = sorted(features, key=lambda f: f.name)
        else:
            for feature in features:
                session.refresh(feature)


def build() -> None:
    if not pathlib.Path(settings.sqlite_db).exists():
        init_db()
    cohort = make_cohort()
    df = gather_icd_codes()
    insert_icd_codes(cohort, df)
