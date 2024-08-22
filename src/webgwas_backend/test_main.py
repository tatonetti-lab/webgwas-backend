import pathlib
import tempfile
import time

import pandas as pd
import pytest
import webgwas.regression
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine
from webgwas.phenotype_definitions import NodeType

from webgwas_backend.main import app, get_session
from webgwas_backend.models import (
    Cohort,
    Feature,
    SQLModel,
    WebGWASResponse,
    WebGWASResult,
)


def setup_db(session: Session, rootdir: pathlib.Path):
    cohort = Cohort(
        name="TEST_COHORT",
        root_directory=rootdir.as_posix(),
        num_covar=2,
    )
    session.add(cohort)
    features = [
        Feature(
            cohort=cohort,
            code="feature1",
            name="Feature 1",
            type=NodeType.REAL,
        ),
        Feature(
            cohort=cohort,
            code="feature2",
            name="Feature 2",
            type=NodeType.REAL,
        ),
        Feature(
            cohort=cohort,
            code="feature3",
            name="Feature 3",
            type=NodeType.REAL,
        ),
    ]
    session.add_all(features)
    session.commit()


def setup_test_data(session: Session, rootdir: pathlib.Path):
    setup_db(session, rootdir)
    phenotype_data = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature3": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    phenotype_data.to_parquet(rootdir.joinpath("phenotype_data.parquet"))
    phenotype_data.cov().to_csv(rootdir.joinpath("phenotypic_covariance.csv"))
    left_inverse = webgwas.regression.compute_left_inverse(phenotype_data)
    left_inverse.T.to_parquet(rootdir.joinpath("phenotype_left_inverse.parquet"))
    # Create GWAS result files
    gwas_results = {
        "feature1": pd.DataFrame(
            {
                "ID": [1, 2, 3, 4, 5],
                "BETA": [1.0, 2.0, 3.0, 4.0, 5.0],
                "SE": [1.0, 2.0, 3.0, 4.0, 5.0],
                "OBS_CT": [1, 2, 3, 4, 5],
            }
        ),
        "feature2": pd.DataFrame(
            {
                "ID": [1, 2, 3, 4, 5],
                "BETA": [1.0, 2.0, 3.0, 4.0, 5.0],
                "SE": [1.0, 2.0, 3.0, 4.0, 5.0],
                "OBS_CT": [1, 2, 3, 4, 5],
            }
        ),
        "feature3": pd.DataFrame(
            {
                "ID": [1, 2, 3, 4, 5],
                "BETA": [1.0, 2.0, 3.0, 4.0, 5.0],
                "SE": [1.0, 2.0, 3.0, 4.0, 5.0],
                "OBS_CT": [1, 2, 3, 4, 5],
            }
        ),
    }
    gwas_dir = rootdir.joinpath("gwas")
    gwas_dir.mkdir(parents=True, exist_ok=True)
    for feature, df in gwas_results.items():
        df.to_csv(
            gwas_dir.joinpath(f"{feature}.tsv.zst"),
            sep="\t",
            index=False,
        )


def get_session_override():
    with tempfile.TemporaryDirectory(delete=False) as rootdir:
        rootdir = pathlib.Path(rootdir)
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            setup_test_data(session, rootdir)
            yield session


@pytest.fixture
def client():
    app.dependency_overrides[get_session] = get_session_override
    with TestClient(app) as client:
        yield client


def test_get_cohorts(client):
    response = client.get("/api/cohorts")
    assert response.status_code == 200, response.json()
    assert response.json() == [{"id": 1, "name": "TEST_COHORT"}]


def test_get_fields(client):
    response = client.get("/api/features?cohort_id=1")
    assert response.status_code == 200, response.json()


def test_validate_phenotype(client: TestClient):
    response = client.put(
        "/api/phenotype",
        params={
            "phenotype_definition": '"feature1" "feature2" `ADD`',
            "cohort_id": 1,
        },
    )
    assert response.status_code == 200, response.json()


@pytest.mark.parametrize(
    "phenotype_definition",
    [
        ('"feature1" "feature2" `ADD`'),
        ('"feature1" "feature2" `SUB`'),
        ('"feature1" "feature2" `MUL`'),
        ('"feature1" "feature2" `DIV` "feature3" `ADD`'),
    ],
)
def test_post_igwas(client, phenotype_definition):
    response = client.post(
        "/api/igwas",
        params={
            "phenotype_definition": phenotype_definition,
            "cohort_id": 1,
        },
    )
    assert response.status_code == 200, response.json()
    validated = WebGWASResponse.model_validate(response.json())
    assert validated.status == "queued"
    time.sleep(0.1)
    for _ in range(10):
        status_response = client.get(f"/api/igwas/status/{validated.request_id}")
        assert status_response.status_code == 200, status_response.json()
        validated_status = WebGWASResponse.model_validate(status_response.json())
        assert validated_status.status == "done"
        assert validated_status.request_id == validated.request_id
        if validated_status.status == "done":
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("Timed out waiting for IGWAS to complete")

    result_response = client.get(f"/api/igwas/results/{validated.request_id}")
    assert result_response.status_code == 200, result_response.json()
    validated_result = WebGWASResult.model_validate(result_response.json())
    assert validated_result.request_id == validated.request_id
    assert validated_result.url is not None


def test_get_fields_invalid_cohort(client):
    response = client.get("/api/features?cohort_id=2")
    assert response.status_code == 404, response.json()
    assert response.json() == {"detail": "Cohort `2` not found"}


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
        params={
            "phenotype_definition": phenotype_definition,
            "cohort_id": 2,
        },
    )
    assert response.status_code == 404, response.json()
    assert response.json() == {"detail": "Cohort `2` not found"}


def test_post_igwas_invalid_phenotype_definition(client):
    response = client.post(
        "/api/igwas",
        params={
            "phenotype_definition": "invalid_phenotype_definition",
            "cohort_id": 1,
        },
    )
    assert response.status_code == 400, response.json()
    assert response.json() == {
        "detail": "Error parsing phenotype: Unknown item invalid_phenotype_definition"
    }
