import functools
import time
from contextlib import asynccontextmanager
from typing import Annotated

import pytest
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from webgwas_backend.config import Settings
from webgwas_backend.data_client import DataClient
from webgwas_backend.main import (
    app,
    get_data_client,
    get_s3_client,
    get_settings,
    job_queue,
    results,
)
from webgwas_backend.models import WebGWASResponse, WebGWASResult
from webgwas_backend.s3_client import S3MockClient
from webgwas_backend.worker import Worker


@functools.lru_cache(maxsize=1)
def get_settings_override():
    return Settings.from_json_file("test_data/settings.json")


@cached(cache=LRUCache(maxsize=1), key=lambda settings: hashkey(True))
def get_data_client_override(
    settings: Annotated[Settings, Depends(get_settings_override)],
):
    return DataClient.from_paths(cohort_paths=settings.cohort_paths)


@cached(cache=LRUCache(maxsize=1), key=lambda settings: hashkey(True))
def get_s3_client_override(
    settings: Annotated[Settings, Depends(get_settings_override)],
):
    return S3MockClient(bucket=settings.s3_bucket)


@asynccontextmanager
async def lifespan_override(app: FastAPI):
    import threading

    worker = Worker(
        job_queue,
        results,
        get_settings_override(),
        get_data_client_override(get_settings_override()),
        get_s3_client_override(get_settings_override()),
    )

    t = threading.Thread(target=worker.run)
    t.daemon = True
    t.start()
    yield


@pytest.fixture
def client():
    app.dependency_overrides[get_settings] = get_settings_override
    app.dependency_overrides[get_data_client] = get_data_client_override
    app.dependency_overrides[get_s3_client] = get_s3_client_override
    app.router.lifespan_context = lifespan_override

    with TestClient(app) as client:
        yield client


def test_get_cohorts(client):
    response = client.get("/api/cohorts")
    assert response.status_code == 200, response.json()
    assert response.json() == ["cohort1", "cohort2"]


def test_get_fields(client):
    response = client.get("/api/fields?cohort_name=cohort1")
    assert response.status_code == 200, response.json()
    assert response.json() == ["feature1", "feature2", "feature3"]


@pytest.mark.parametrize(
    "phenotype_definition,cohort",
    [
        ('"feature1" "feature2" `AND`', "cohort1"),
        ('"feature1" "feature2" `OR`', "cohort1"),
        ('"feature1" "feature2" `ADD`', "cohort1"),
        ('"feature1" "feature2" `AND` "feature3" `OR`', "cohort1"),
    ],
)
def test_post_igwas(client, phenotype_definition, cohort):
    response = client.post(
        "/api/igwas",
        json={
            "phenotype_definition": phenotype_definition,
            "cohort_name": cohort,
        },
    )
    assert response.status_code == 200, response.json()
    validated = WebGWASResponse.model_validate(response.json())
    assert validated.status == "queued"
    time.sleep(0.5)
    status_response = client.get(f"/api/igwas/status/{validated.request_id}")
    result_response = client.get(f"/api/igwas/results/{validated.request_id}")
    assert status_response.status_code == 200, status_response.json()
    assert result_response.status_code == 200, result_response.json()
    validated_status = WebGWASResponse.model_validate(status_response.json())
    assert validated_status.status == "done"
    assert validated_status.request_id == validated.request_id
    validated_result = WebGWASResult.model_validate(result_response.json())
    assert validated_result.request_id == validated.request_id
    assert validated_result.url is not None


def test_get_fields_invalid_cohort(client):
    response = client.get("/api/fields?cohort_name=invalid_cohort")
    assert response.status_code == 404, response.json()
    assert response.json() == {"detail": "Cohort `invalid_cohort` not found"}


@pytest.mark.parametrize(
    "phenotype_definition",
    [
        '"feature1" "feature2" `AND`',
        "invalid_phenotype_definition",
    ],
)
def test_post_igwas_invalid_cohort(client, phenotype_definition):
    response = client.post(
        "/api/igwas",
        json={
            "phenotype_definition": phenotype_definition,
            "cohort_name": "invalid_cohort",
        },
    )
    assert response.status_code == 404, response.json()
    assert response.json() == {"detail": "Cohort `invalid_cohort` not found"}


def test_post_igwas_invalid_phenotype_definition(client):
    response = client.post(
        "/api/igwas",
        json={
            "phenotype_definition": "invalid_phenotype_definition",
            "cohort_name": "cohort1",
        },
    )
    assert response.status_code == 200, response.json()
    request_id = response.json().get("request_id")
    assert request_id is not None
    time.sleep(0.5)
    response = client.get(f"/api/igwas/status/{request_id}")
    assert response.status_code == 200, response.json()
    js = response.json()
    response = WebGWASResponse.model_validate(js)
    assert response.status == "error"
    response = client.get(f"/api/igwas/results/{request_id}")
    assert response.status_code == 500, response.json()
    js = response.json()
    assert js.get("detail") == (
        "400: Error parsing phenotype: Unknown char 'i' in "
        "'invalid_phenotype_definition', START state"
    )
