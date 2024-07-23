from typing import Annotated

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from webgwas_fastapi.config import Settings
from webgwas_fastapi.data_client import DataClient
from webgwas_fastapi.main import app, get_data_client, get_s3_client, get_settings
from webgwas_fastapi.s3_client import S3MockClient

client = TestClient(app)


def get_settings_override():
    return Settings(  # type: ignore
        _env_file=(  # type: ignore
            "/Users/zietzm/Documents/projects/webgwas-frontend/"
            "webgwas-fastapi/test_data/settings.json"
        ),
        _env_file_encoding="utf-8",  # type: ignore
    )


def get_data_client_override(
    settings: Annotated[Settings, Depends(get_settings_override)],
):
    return DataClient.model_validate(settings.data_client, from_attributes=True)


def get_s3_client_override(
    settings: Annotated[Settings, Depends(get_settings_override)],
):
    return S3MockClient()  # type: ignore


app.dependency_overrides[get_settings] = get_settings_override
app.dependency_overrides[get_data_client] = get_data_client_override
app.dependency_overrides[get_s3_client] = get_s3_client_override


def test_get_cohorts():
    response = client.get("/api/cohorts")
    assert response.status_code == 200, response.json()
    assert response.json() == ["cohort1", "cohort2"]


def test_get_fields():
    response = client.get("/api/fields?cohort=cohort1")
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
            "cohort": cohort,
        },
    )
    assert response.status_code == 200, response.json()
    assert set(response.json().keys()) == {"request_id", "url"}


def test_get_fields_invalid_cohort():
    response = client.get("/api/fields?cohort=invalid_cohort")
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
            "cohort": "invalid_cohort",
        },
    )
    assert response.status_code == 404, response.json()
    assert response.json() == {"detail": "Cohort `invalid_cohort` not found"}


def test_post_igwas_invalid_phenotype_definition():
    response = client.post(
        "/api/igwas",
        json={
            "phenotype_definition": "invalid_phenotype_definition",
            "cohort": "cohort1",
        },
    )
    assert response.status_code == 400, response.json()
    assert response.json() == {
        "detail": (
            "Error parsing phenotype: Unknown char 'i' in "
            "'invalid_phenotype_definition', START state"
        )
    }
