import functools
import logging
from contextlib import asynccontextmanager
from queue import Queue
from typing import Annotated

import boto3
import webgwas
import webgwas.parser
from botocore.config import Config
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from webgwas_backend.config import Settings
from webgwas_backend.data_client import DataClient, GWASCohort
from webgwas_backend.models import (
    PhenotypeNode,
    WebGWASRequest,
    WebGWASRequestID,
    WebGWASResponse,
    WebGWASResult,
)
from webgwas_backend.s3_client import S3ProdClient
from webgwas_backend.worker import Worker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

s3 = boto3.client("s3", config=Config(signature_version="v4"))
job_queue = Queue()
queued_request_ids = set()
results = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading

    worker = Worker(
        job_queue,
        queued_request_ids,
        results,
        get_settings(),
        get_data_client(get_settings()),
        get_s3_client(get_settings()),
    )
    t = threading.Thread(target=worker.run)
    t.daemon = True
    t.start()
    yield


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@functools.lru_cache(maxsize=1)
def get_settings():
    return Settings.from_json_file("settings.json")


@cached(cache=LRUCache(maxsize=1), key=lambda settings: hashkey(True))
def get_data_client(settings: Annotated[Settings, Depends(get_settings)]):
    return DataClient.from_paths(settings.cohort_paths)


@cached(cache=LRUCache(maxsize=1), key=lambda settings: hashkey(True))
def get_s3_client(settings: Annotated[Settings, Depends(get_settings)]):
    return S3ProdClient(s3_client=s3, bucket=settings.s3_bucket)


def validate_cohort(
    data_client: Annotated[DataClient, Depends(get_data_client)], cohort_name: str
) -> GWASCohort:
    try:
        result = data_client.validate_cohort(cohort_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating cohort: {e}")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Cohort `{cohort_name}` not found")
    return result


@app.get("/api/cohorts", response_model=list[str])
def get_cohorts(data_client: Annotated[DataClient, Depends(get_data_client)]):
    return data_client.get_cohorts()


@app.get("/api/nodes", response_model=list[PhenotypeNode])
def get_nodes(cohort: Annotated[GWASCohort, Depends(validate_cohort)]):
    root_node = PhenotypeNode(
        id=0, type="operator", name="Root", min_arity=1, max_arity=1
    )
    feature_nodes = [
        PhenotypeNode(id=i + 1, type="field", name=feature, min_arity=0, max_arity=0)
        for i, feature in enumerate(cohort.feature_names)
    ]
    operator_nodes = [
        PhenotypeNode(
            id=i + len(feature_nodes) + 1,
            type="operator",
            name=name,
            min_arity=2,
            max_arity=None,
        )
        for i, name in enumerate(webgwas.parser.OperatorNode)
    ]
    return [root_node] + feature_nodes + operator_nodes


@app.get("/api/fields", response_model=list[str])
def get_fields(cohort: Annotated[GWASCohort, Depends(validate_cohort)]):
    return cohort.feature_names


@app.post("/api/igwas", response_model=WebGWASResponse)
def post_igwas(
    data_client: Annotated[DataClient, Depends(get_data_client)],
    request: WebGWASRequest,
) -> WebGWASResponse:
    cohort = validate_cohort(data_client=data_client, cohort_name=request.cohort_name)
    new_request = WebGWASRequestID(
        phenotype_definition=request.phenotype_definition,
        cohort=cohort,
    )
    job_queue.put(new_request)
    queued_request_ids.add(new_request.request_id)
    return WebGWASResponse(request_id=new_request.request_id, status="queued")


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
