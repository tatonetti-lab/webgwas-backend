import json

import pytest
from fastapi.testclient import TestClient

from webgwas_fastapi.data_client import DataClient
from webgwas_fastapi.main import app, get_data_client, get_s3_client
from webgwas_fastapi.s3_client import S3MockClient

client = TestClient(app)

with open(
    "/Users/zietzm/Documents/projects/webgwas-frontend/"
    "webgwas-fastapi/test_data/config.json"
) as f:
    settings = json.load(f)

mock_data_client = DataClient.model_validate(settings["data_client"])
mock_s3_client = S3MockClient()

app.dependency_overrides[get_data_client] = lambda: mock_data_client
app.dependency_overrides[get_s3_client] = lambda: mock_s3_client


def test_get_cohorts():
    response = client.get("/api/cohorts")
    assert response.status_code == 200
    assert response.json() == ["cohort1", "cohort2"]


def test_get_fields():
    response = client.get("/api/fields?cohort=cohort1")
    assert response.status_code == 200
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
    assert response.status_code == 200
    assert set(response.json().keys()) == {"request_id", "url"}


def test_get_fields_invalid_cohort():
    response = client.get("/api/fields?cohort=invalid_cohort")
    assert response.status_code == 404
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
    assert response.status_code == 404
    assert response.json() == {"detail": "Cohort `invalid_cohort` not found"}


def test_post_igwas_invalid_phenotype_definition():
    response = client.post(
        "/api/igwas",
        json={
            "phenotype_definition": "invalid_phenotype_definition",
            "cohort": "cohort1",
        },
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Error parsing phenotype: Unknown char 'i' in "
            "'invalid_phenotype_definition', START state"
        )
    }
