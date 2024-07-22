import functools
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import pandas as pd
import psutil
import webgwas.igwas
import webgwas.parser
import webgwas.regression
from botocore.exceptions import ClientError
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from webgwas_fastapi.data_client import DataClient
from webgwas_fastapi.s3_client import S3Client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

NUM_COVAR = 12
NUM_THREADS = psutil.cpu_count()
CHUNK_SIZE = 1_000_000
CAPACITY = 25
COMPRESS = True
QUIET = False

DATA_DIR = Path(__file__).parent.joinpath("data")
RESULT_DIR = DATA_DIR.joinpath("igwas_results")

CACHE_SIZE = 100
S3_BUCKET = "webgwas-results"


def get_data_client():
    raise NotImplementedError("get_data_client not implemented")


def get_s3_client():
    raise NotImplementedError("get_s3_client not implemented")


def validate_cohort(
    data_client: Annotated[DataClient, Depends(get_data_client)], cohort: str
):
    result = data_client.validate_cohort(cohort)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Cohort `{cohort}` not found")
    return result


class WebGWASRequest(BaseModel):
    """Request for running an IGWAS.

    Defined so that users can pass phenotype definitions in a JSON body instead of
    a query string in the URL.
    """

    phenotype_definition: str
    cohort: str


class WebGWASResult(BaseModel):
    """Result of an IGWAS."""

    request_id: str
    url: str


@app.get("/api/cohorts", response_model=list[str])
def get_cohorts(data_client: Annotated[DataClient, Depends(get_data_client)]):
    return data_client.get_cohorts()


@app.get("/api/fields", response_model=list[str])
def get_fields(
    data_client: Annotated[DataClient, Depends(get_data_client)],
    cohort: Annotated[str, Depends(validate_cohort)],
):
    return data_client.get_features(cohort)


@app.post("/api/igwas", response_model=WebGWASResult)
def post_igwas(
    data_client: Annotated[DataClient, Depends(get_data_client)],
    s3_client: Annotated[S3Client, Depends(get_s3_client)],
    request: WebGWASRequest,
) -> WebGWASResult:
    validate_cohort(data_client=data_client, cohort=request.cohort)
    return handle_igwas(
        data_client=data_client,
        s3_client=s3_client,
        phenotype_definition=request.phenotype_definition,
        cohort=request.cohort,
    )


run_igwas = functools.partial(
    webgwas.igwas.igwas_files,
    num_covar=NUM_COVAR,
    chunksize=CHUNK_SIZE,
    variant_id="ID",
    beta="BETA",
    std_error="SE",
    sample_size="OBS_CT",
    num_threads=NUM_THREADS,
    capacity=CAPACITY,
    compress=COMPRESS,
    quiet=QUIET,
)


@cached(
    cache=LRUCache(maxsize=CACHE_SIZE),
    key=lambda data_client, s3_client, phenotype_definition, cohort: hashkey(
        phenotype_definition, cohort
    ),
)
def handle_igwas(
    data_client: Annotated[DataClient, Depends(get_data_client)],
    s3_client: Annotated[S3Client, Depends(get_s3_client)],
    phenotype_definition: str,
    cohort: str,
) -> WebGWASResult:
    request_id = str(uuid.uuid4())
    # TODO: Use this in logs

    # Parse the phenotype definition
    try:
        parser = webgwas.parser.RPNParser(phenotype_definition)
    except webgwas.parser.ParserException as e:
        raise HTTPException(status_code=400, detail=f"Error parsing phenotype: {e}")

    # Load data, assign the target phenotype
    features_df, cov_path, gwas_paths = data_client.get_data_cov_gwas_unchecked(cohort)

    # Assign the target phenotype
    try:
        target_phenotype = features_df.apply(
            lambda row: parser.apply_definition(row), axis=1
        )
        assert isinstance(target_phenotype, pd.Series)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error applying phenotype definition: {e}"
        )

    # Regress the target phenotype against the feature phenotypes
    try:
        beta_series = webgwas.regression.regress(target_phenotype, features_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in regression: {e}")

    del features_df  # Free up memory

    # Indirect GWAS
    with (
        tempfile.NamedTemporaryFile(suffix=".csv") as beta_file,
        tempfile.NamedTemporaryFile(suffix=".tsv.zst") as output_file,
    ):
        (
            beta_series.round(5)
            .rename(request_id)
            .to_frame()
            .rename_axis(index="feature")
            .to_csv(beta_file)
        )
        beta_file.flush()
        try:
            run_igwas(beta_file.name, cov_path, gwas_paths, output_file.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in IGWAS: {e}")

        # Upload the result to S3
        output_file_path = f"{request_id}.tsv.zst"
        try:
            s3_client.upload_file(output_file.name, S3_BUCKET, output_file_path)
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

    return WebGWASResult(
        request_id=request_id, url=f"https://{S3_BUCKET}/{output_file_path}"
    )
