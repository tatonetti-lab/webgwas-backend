import functools
from typing import Annotated

import pytest
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from fastapi import Depends
from fastapi.testclient import TestClient

from webgwas_backend.config import Settings
from webgwas_backend.data_client import DataClient
from webgwas_backend.main import app, get_data_client, get_s3_client, get_settings
from webgwas_backend.s3_client import S3MockClient

client = TestClient(app)


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


app.dependency_overrides[get_settings] = get_settings_override
app.dependency_overrides[get_data_client] = get_data_client_override
app.dependency_overrides[get_s3_client] = get_s3_client_override


def test_get_cohorts():
    response = client.get("/api/cohorts")
    assert response.status_code == 200, response.json()
    assert response.json() == ["cohort1", "cohort2"]


def test_get_fields():
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
def test_post_igwas(phenotype_definition, cohort):
    response = client.post(
        "/api/igwas",
        json={
            "phenotype_definition": phenotype_definition,
            "cohort_name": cohort,
        },
    )
    assert response.status_code == 200, response.json()
    assert set(response.json().keys()) == {"request_id", "url"}


def test_get_fields_invalid_cohort():
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
def test_post_igwas_invalid_cohort(phenotype_definition):
    response = client.post(
        "/api/igwas",
        json={
            "phenotype_definition": phenotype_definition,
            "cohort_name": "invalid_cohort",
        },
    )
    assert response.status_code == 404, response.json()
    assert response.json() == {"detail": "Cohort `invalid_cohort` not found"}


def test_post_igwas_invalid_phenotype_definition():
    response = client.post(
        "/api/igwas",
        json={
            "phenotype_definition": "invalid_phenotype_definition",
            "cohort_name": "cohort1",
        },
    )
    assert response.status_code == 400, response.json()
    assert response.json() == {
        "detail": (
            "Error parsing phenotype: Unknown char 'i' in "
            "'invalid_phenotype_definition', START state"
        )
    }
