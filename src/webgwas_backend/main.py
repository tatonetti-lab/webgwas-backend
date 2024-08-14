import logging
import threading
from contextlib import asynccontextmanager
from queue import Queue

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
    ValidPhenotype,
    ValidPhenotypeResponse,
    WebGWASRequestID,
    WebGWASResponse,
    WebGWASResult,
)
from webgwas_backend.s3_client import get_s3_client
from webgwas_backend.worker import Worker

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
init_db()
s3 = get_s3_client(settings.dry_run, settings.s3_bucket)

job_queue = Queue()
queued_request_ids = set()
results = dict()


def get_session():
    with Session(engine) as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = Worker(job_queue, queued_request_ids, results, s3, settings)
    t = threading.Thread(target=worker.run)
    t.daemon = True
    t.start()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_cohort(
    *, session: Session = Depends(get_session), cohort_id: int
) -> Cohort:
    cohort = session.get(Cohort, cohort_id)
    if cohort is None:
        raise HTTPException(status_code=404, detail=f"Cohort `{cohort_id}` not found")
    return cohort


@app.get("/api/cohorts", response_model=list[CohortResponse])
def get_cohorts(session: Session = Depends(get_session)):
    return session.exec(select(Cohort)).all()


@app.get("/api/features", response_model=list[FeatureResponse])
def get_nodes(cohort: Cohort = Depends(validate_cohort)):
    return cohort.features


@app.put("/api/phenotype", response_model=ValidPhenotypeResponse)
def validate_phenotype(
    *,
    cohort: Cohort = Depends(validate_cohort),
    phenotype_definition: str,
) -> ValidPhenotype:
    try:
        nodes = webgwas.phenotype_definitions.parse_string_definition(
            phenotype_definition
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing phenotype: {e}")

    try:
        fields = [
            Field.model_validate(f, from_attributes=True) for f in cohort.features
        ]
        knowledge_base = KnowledgeBase.default(fields)
        nodes = webgwas.phenotype_definitions.validate_nodes(nodes, knowledge_base)
        webgwas.phenotype_definitions.type_check_nodes(nodes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error validating phenotype: {e}")
    return ValidPhenotype(phenotype_definition=phenotype_definition, valid_nodes=nodes)


@app.post("/api/igwas", response_model=WebGWASResponse)
def post_igwas(
    cohort: Cohort = Depends(validate_cohort),
    phenotype_definition: ValidPhenotype = Depends(validate_phenotype),
) -> WebGWASResponse:
    new_request = WebGWASRequestID(
        cohort=cohort,
        phenotype_definition=phenotype_definition,
    )
    job_queue.put(new_request)
    queued_request_ids.add(new_request.id)
    return WebGWASResponse(request_id=new_request.id, status="queued")


@app.get("/api/igwas/status/{request_id}", response_model=WebGWASResponse)
def get_igwas_status(request_id: str) -> WebGWASResponse:
    queued = request_id in queued_request_ids
    if queued:
        return WebGWASResponse(request_id=request_id, status="queued")
    result_exists = request_id in results
    if result_exists:
        return results[request_id]
    raise HTTPException(status_code=404, detail="Request not found")


@app.get("/api/igwas/results/{request_id}", response_model=WebGWASResult)
def get_igwas_results(request_id: str) -> WebGWASResult:
    result = results.get(request_id)
    queued = request_id in queued_request_ids
    if result is None and not queued:
        raise HTTPException(status_code=404, detail="Request not found")
    elif result is None and queued:
        raise HTTPException(status_code=202, detail="Request is queued")
    else:
        assert result is not None
        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error_msg)
    return result
