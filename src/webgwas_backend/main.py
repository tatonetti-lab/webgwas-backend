import functools
import logging
import pathlib
import tempfile
import uuid
from typing import Annotated

import boto3
import webgwas.igwas
import webgwas.parser
import webgwas.regression
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from fastapi import Depends, FastAPI, HTTPException
from pandas import Series
from pydantic import BaseModel, Field

from webgwas_backend.config import Settings
from webgwas_backend.data_client import DataClient, GWASCohort
from webgwas_backend.s3_client import S3Client, S3ProdClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()
s3 = boto3.client("s3")


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


class WebGWASRequest(BaseModel):
    """Request for GWAS summary statistics"""

    phenotype_definition: str = Field(
        ...,
        description=(
            "Phenotype definition in reverse polish notation. "
            "See https://github.com/zietzm/webgwas#phenotype-definitions"
        ),
    )
    cohort_name: str = Field(
        ...,
        description=(
            "Cohort for which to run the GWAS. "
            "See https://github.com/zietzm/webgwas#cohorts"
        ),
    )


class WebGWASResult(BaseModel):
    """Result of a successful GWAS"""

    request_id: str = Field(..., description="Unique identifier for the request.")
    url: str = Field(..., description="URL to the result file. ")


@app.get("/api/cohorts", response_model=list[str])
def get_cohorts(data_client: Annotated[DataClient, Depends(get_data_client)]):
    return data_client.get_cohorts()


@app.get("/api/fields", response_model=list[str])
def get_fields(cohort: Annotated[GWASCohort, Depends(validate_cohort)]):
    return cohort.feature_names


@app.post("/api/igwas", response_model=WebGWASResult)
def post_igwas(
    settings: Annotated[Settings, Depends(get_settings)],
    data_client: Annotated[DataClient, Depends(get_data_client)],
    s3_client: Annotated[S3Client, Depends(get_s3_client)],
    request: WebGWASRequest,
) -> WebGWASResult:
    cohort = validate_cohort(data_client=data_client, cohort_name=request.cohort_name)
    return handle_igwas(
        settings=settings,
        data_client=data_client,
        s3_client=s3_client,
        phenotype_definition=request.phenotype_definition,
        cohort=cohort,
    )


@cached(
    cache=LRUCache(maxsize=get_settings().lru_cache_size),
    key=lambda settings, data_client, s3_client, phenotype_definition, cohort: hashkey(
        phenotype_definition, cohort.cohort_name
    ),
)
def handle_igwas(
    settings: Annotated[Settings, Depends(get_settings)],
    data_client: Annotated[DataClient, Depends(get_data_client)],
    s3_client: Annotated[S3Client, Depends(get_s3_client)],
    phenotype_definition: str,
    cohort: GWASCohort,
) -> WebGWASResult:
    request_id = str(uuid.uuid4())  # TODO: Use this in logs

    # Parse the phenotype definition
    try:
        parser = webgwas.parser.RPNParser(phenotype_definition)
    except webgwas.parser.ParserException as e:
        raise HTTPException(status_code=400, detail=f"Error parsing phenotype: {e}")

    # Load data, assign the target phenotype
    features_df, cov_path, gwas_paths = data_client.get_data_cov_gwas_unchecked(
        cohort.cohort_name
    )

    # Assign the target phenotype
    try:
        target_phenotype = features_df.apply(
            lambda row: parser.apply_definition(row), axis=1
        )
        assert isinstance(target_phenotype, Series)
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
    with tempfile.TemporaryDirectory() as temp_dir:
        beta_file_path = pathlib.Path(temp_dir).joinpath(f"{request_id}.csv").as_posix()
        output_file_path = pathlib.Path(temp_dir).joinpath(f"{request_id}.tsv.zst")

        (
            beta_series.round(5)
            .rename(request_id)
            .to_frame()
            .rename_axis(index="feature")
            .to_csv(beta_file_path)
        )
        try:
            webgwas.igwas.igwas_files(
                projection_matrix_path=beta_file_path,
                covariance_matrix_path=cov_path.as_posix(),
                gwas_result_paths=[p.as_posix() for p in gwas_paths],
                output_file_path=output_file_path.as_posix(),
                num_covar=cohort.num_covar,
                chunksize=settings.indirect_gwas.chunk_size,
                variant_id="ID",
                beta="BETA",
                std_error="SE",
                sample_size="OBS_CT",
                num_threads=settings.indirect_gwas.num_threads,
                capacity=settings.indirect_gwas.capacity,
                compress=settings.indirect_gwas.compress,
                quiet=settings.indirect_gwas.quiet,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in indirect GWAS: {e}")

        # Upload the result to S3
        try:
            s3_client.upload_file(output_file_path.as_posix(), output_file_path.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

    try:
        presigned_url = s3_client.get_presigned_url(output_file_path.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting presigned URL: {e}")

    return WebGWASResult(request_id=request_id, url=presigned_url)
