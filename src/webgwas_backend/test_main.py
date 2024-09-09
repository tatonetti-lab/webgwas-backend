import pathlib
import tempfile
import time

import pandas as pd
import polars as pl
import pytest
import webgwas.regression
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine
from webgwas.phenotype_definitions import NodeType

from webgwas_backend.config import IndirectGWASSettings, Settings
from webgwas_backend.main import app, get_session, get_worker
from webgwas_backend.models import (
    Cohort,
    Feature,
    SQLModel,
    WebGWASResponse,
    WebGWASResult,
)
from webgwas_backend.worker import TestWorker


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
    pl.DataFrame(
        {
            "variant_id": ["1", "2", "3", "4", "5"],
            "feature1": [
                {"beta": 1.1, "degrees_of_freedom": 1.0, "genotype_variance": 1.0},
                {"beta": 2.2, "degrees_of_freedom": 2.0, "genotype_variance": 2.0},
                {"beta": 3.3, "degrees_of_freedom": 3.0, "genotype_variance": 3.0},
                {"beta": 4.4, "degrees_of_freedom": 4.0, "genotype_variance": 4.0},
                {"beta": 5.5, "degrees_of_freedom": 5.0, "genotype_variance": 5.0},
            ],
            "feature2": [
                {"beta": 1.5, "degrees_of_freedom": 1.0, "genotype_variance": 1.0},
                {"beta": 2.5, "degrees_of_freedom": 2.0, "genotype_variance": 2.0},
                {"beta": 3.5, "degrees_of_freedom": 3.0, "genotype_variance": 3.0},
                {"beta": 4.5, "degrees_of_freedom": 4.0, "genotype_variance": 4.0},
                {"beta": 5.5, "degrees_of_freedom": 5.0, "genotype_variance": 5.0},
            ],
            "feature3": [
                {"beta": 1.5, "degrees_of_freedom": 1.0, "genotype_variance": 1.0},
                {"beta": 2.5, "degrees_of_freedom": 2.0, "genotype_variance": 2.0},
                {"beta": 3.5, "degrees_of_freedom": 3.0, "genotype_variance": 3.0},
                {"beta": 4.5, "degrees_of_freedom": 4.0, "genotype_variance": 4.0},
                {"beta": 5.5, "degrees_of_freedom": 5.0, "genotype_variance": 5.0},
            ],
        },
        schema={
            "variant_id": pl.Utf8,
            "feature1": pl.Struct(
                {
                    "beta": pl.Float32,
                    "degrees_of_freedom": pl.Int32,
                    "genotype_variance": pl.Float32,
                }
            ),
            "feature2": pl.Struct(
                {
                    "beta": pl.Float32,
                    "degrees_of_freedom": pl.Int32,
                    "genotype_variance": pl.Float32,
                }
            ),
            "feature3": pl.Struct(
                {
                    "beta": pl.Float32,
                    "degrees_of_freedom": pl.Int32,
                    "genotype_variance": pl.Float32,
                }
            ),
        },
    ).write_parquet(rootdir.joinpath("gwas.parquet"))


def get_session_override():
    with tempfile.TemporaryDirectory() as rootdir:
        rootdir = pathlib.Path(rootdir)
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            setup_test_data(session, rootdir)
            yield session


@pytest.fixture
def client():
    settings = Settings(
        log_level="DEBUG",
        dry_run=True,
        s3_bucket="TEST",
        s3_result_path="TEST",
        sqlite_db=":memory:",
        n_workers=1,
        fit_quality_file=pathlib.Path(":memory:"),
        root_data_directory=pathlib.Path(":memory:"),
        indirect_gwas=IndirectGWASSettings(),
        cache_capacity=1,
    )
    worker = TestWorker(settings)

    def get_worker_override():
        return worker

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_worker] = get_worker_override
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
def test_post_igwas(client: TestClient, phenotype_definition: str):
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
    for _ in range(20):
        status_response = client.get(f"/api/igwas/results/{validated.request_id}")
        assert status_response.status_code in {200, 202}, status_response.json()
        validated_status = WebGWASResult.model_validate(status_response.json())
        assert validated_status.request_id == validated.request_id
        match validated_status.status:
            case "done":
                break
            case "queued":
                time.sleep(0.1)
            case "error":
                raise ValueError(f"Unexpected status: {validated_status}")
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
