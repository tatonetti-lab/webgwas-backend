import logging
from collections.abc import Generator
from typing import Annotated

import webgwas.phenotype_definitions
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from webgwas.phenotype_definitions import Field, KnowledgeBase

from webgwas_backend.config import settings
from webgwas_backend.database import engine, init_db
from webgwas_backend.models import (
    Cohort,
    CohortResponse,
    FeatureResponse,
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

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

init_db()
worker_group = Worker(settings)


def get_worker() -> Worker:
    return worker_group


def get_session() -> Generator[Session]:
    with Session(engine) as session:
        yield session


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_cohort(
    *, session: Annotated[Session, Depends(get_session)], cohort_id: int
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
def get_cohorts(session: Annotated[Session, Depends(get_session)]):
    return session.exec(select(Cohort)).all()


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
    cohort: Annotated[Cohort, Depends(validate_cohort)],
    phenotype_definition: Annotated[ValidPhenotype, Depends(validate_phenotype)],
    n_samples: int = 1000,
) -> PhenotypeSummary:
    return get_phenotype_summary_impl(cohort, phenotype_definition).subsample(n_samples)


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
