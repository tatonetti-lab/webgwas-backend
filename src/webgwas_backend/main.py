import logging
from collections.abc import Generator, Sequence
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import polars as pl
import webgwas.phenotype_definitions
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine
from sqlmodel import Session, select
from webgwas.phenotype_definitions import Field, KnowledgeBase

from webgwas_backend.config import init_logging, settings
from webgwas_backend.database import engine, init_db
from webgwas_backend.models import (
    Cohort,
    CohortResponse,
    FeatureResponse,
    PhenotypeFitQuality,
    PhenotypeSummary,
    ValidPhenotype,
    ValidPhenotypeResponse,
    WebGWASRequestID,
    WebGWASResponse,
    WebGWASResult,
)
from webgwas_backend.phenotype_summary import (
    get_phenotype_summary as get_phenotype_summary_impl,
)
from webgwas_backend.worker import Worker

logger = logging.getLogger(__name__)

init_db()

worker: Worker | None = None
fit_quality: list[PhenotypeFitQuality] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    init_logging()
    global worker
    worker = Worker(settings)
    fit_quality_vals = [
        (float(f"{x:.3f}"), float(f"{y:.3f}"))
        for x, y in pl.read_parquet(settings.fit_quality_file)
        .to_pandas()
        .values.tolist()
    ]
    global fit_quality
    fit_quality = [PhenotypeFitQuality(p=y, g=x) for x, y in fit_quality_vals]
    yield


@lru_cache(maxsize=1)
def get_fit_quality() -> list[PhenotypeFitQuality]:
    if fit_quality is None:
        raise HTTPException(status_code=500, detail="Fit quality not loaded")
    return fit_quality


@lru_cache(maxsize=1)
def get_worker() -> Worker:
    return Worker(settings)


def get_engine() -> Engine:
    return engine


def get_session(engine: Annotated[Engine, Depends(get_engine)]) -> Generator[Session]:
    with Session(engine) as session:
        yield session


@lru_cache(maxsize=1)
def get_root_data_directory() -> Path:
    return Path("prod_data")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_cohort(
    *,
    session: Annotated[Session, Depends(get_session)],
    cohort_id: int,
) -> Cohort:
    cohort = session.get(Cohort, cohort_id)
    if cohort is None:
        raise HTTPException(status_code=404, detail=f"Cohort `{cohort_id}` not found")
    return cohort


@app.get(
    "/api/cohorts",
    response_model=list[CohortResponse],
    response_model_exclude_none=True,
)
def get_cohorts(session: Annotated[Session, Depends(get_session)]) -> Sequence[Cohort]:
    cohorts = session.exec(select(Cohort)).all()
    return cohorts


@app.get(
    "/api/features",
    response_model=list[FeatureResponse],
    response_model_exclude_none=True,
)
def get_nodes(cohort: Annotated[Cohort, Depends(validate_cohort)]):
    return cohort.features


@app.put(
    "/api/phenotype",
    response_model=ValidPhenotypeResponse,
    response_model_exclude_none=True,
)
def validate_phenotype(
    *,
    cohort: Annotated[Cohort, Depends(validate_cohort)],
    phenotype_definition: str,
) -> ValidPhenotype:
    try:
        nodes = webgwas.phenotype_definitions.parse_string_definition(
            phenotype_definition
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error parsing phenotype: {e}"
        ) from e

    try:
        fields = [
            Field.model_validate(f, from_attributes=True) for f in cohort.features
        ]
        knowledge_base = KnowledgeBase.default(fields)
        nodes = webgwas.phenotype_definitions.validate_nodes(nodes, knowledge_base)
        webgwas.phenotype_definitions.type_check_nodes(nodes)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error validating phenotype: {e}"
        ) from e
    return ValidPhenotype(
        phenotype_definition=phenotype_definition,
        valid_nodes=nodes,
        is_valid=True,
        message="Valid phenotype",
    )


@app.get(
    "/api/phenotype_summary",
    response_model=PhenotypeSummary,
    response_model_exclude_none=True,
)
def get_phenotype_summary(
    *,
    cohort: Annotated[Cohort, Depends(validate_cohort)],
    phenotype_definition: Annotated[ValidPhenotype, Depends(validate_phenotype)],
    n_samples: int = 1000,
    fit_quality: Annotated[list[PhenotypeFitQuality], Depends(get_fit_quality)],
    root_data_directory: Annotated[Path, Depends(get_root_data_directory)],
) -> PhenotypeSummary:
    return get_phenotype_summary_impl(
        cohort, phenotype_definition, fit_quality, root_data_directory
    ).subsample(n_samples)


@app.post(
    "/api/igwas", response_model=WebGWASResponse, response_model_exclude_none=True
)
def post_igwas(
    cohort: Annotated[Cohort, Depends(validate_cohort)],
    phenotype_definition: Annotated[ValidPhenotype, Depends(validate_phenotype)],
    worker: Annotated[Worker, Depends(get_worker)],
) -> WebGWASResponse:
    new_request = WebGWASRequestID(
        cohort=cohort,
        phenotype_definition=phenotype_definition,
    )
    worker.submit(new_request)
    return WebGWASResponse(request_id=new_request.id, status="queued")


@app.get(
    "/api/igwas/results/{request_id}",
    response_model=WebGWASResult,
    response_model_exclude_none=True,
)
def get_igwas_results(
    request_id: str, worker: Annotated[Worker, Depends(get_worker)]
) -> WebGWASResult:
    return worker.get_results(request_id)


@app.get("/api/static/fit_quality")
def get_fit_quality_endpoint(
    fit_quality: Annotated[list[PhenotypeFitQuality], Depends(get_fit_quality)],
) -> list[PhenotypeFitQuality]:
    return fit_quality
