import logging
import pathlib
import tempfile

import pandas as pd
import webgwas.igwas
import webgwas.phenotype_definitions
import webgwas.regression
from fastapi import HTTPException
from pandas import Series

from webgwas_backend.models import WebGWASRequestID, WebGWASResult
from webgwas_backend.s3_client import get_s3_client

logger = logging.getLogger("uvicorn")


def get_igwas_coef(
    request: WebGWASRequestID, root_data_directory: pathlib.Path
) -> Series:
    directory = root_data_directory.joinpath(request.cohort.root_directory)

    # Load feature data
    features_path = directory.joinpath("phenotype_data.parquet")
    exists = features_path.exists()
    logger.info(f"Loading data from {features_path} (exists: {exists})")
    try:
        features_df = pd.read_parquet(features_path)
    except Exception as e:
        logger.error(f"Error loading data from {features_path}: {e}")
        raise e

    # Assign the target phenotype
    logger.info("Applying phenotype definition to data")
    target_phenotype = webgwas.phenotype_definitions.apply_definition_pandas(
        nodes=request.phenotype_definition.valid_nodes, df=features_df
    )
    assert isinstance(target_phenotype, Series)
    del features_df  # Free up memory
    logger.debug(f"Target phenotype: {target_phenotype}")

    # Load left inverse
    logger.info("Loading left inverse")
    left_inverse_path = directory.joinpath("phenotype_left_inverse.parquet")
    left_inverse_df = pd.read_parquet(left_inverse_path).T
    logger.debug(f"Left inverse: {left_inverse_df}")

    # Regress the target phenotype against the feature phenotypes
    logger.info("Regressing phenotype against features")
    beta_series = webgwas.regression.regress_left_inverse(
        target_phenotype, left_inverse_df
    )
    del left_inverse_df  # Free up memory
    return beta_series.round(5).rename(request.id).rename_axis(index="feature")


def handle_igwas(
    request: WebGWASRequestID,
    root_data_directory: pathlib.Path,
    dry_run: bool,
    s3_bucket: str,
    s3_region: str,
) -> WebGWASResult:
    beta_series = get_igwas_coef(request, root_data_directory).drop(
        "const", errors="ignore"
    )
    cov_path = request.cohort.get_root_path(root_data_directory).joinpath(
        "phenotypic_covariance.csv"
    )
    covariance_matrix = pd.read_csv(cov_path, index_col=0)
    gwas_path = request.cohort.get_root_path(root_data_directory).joinpath(
        "gwas.parquet"
    )
    assert gwas_path.exists(), f"GWAS file not found: {gwas_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Running Indirect GWAS")
        output_file_path = pathlib.Path(temp_dir).joinpath(f"{request.id}.tsv.zst")
        try:
            webgwas.igwas.igwas_prod(
                projection_vector=beta_series,
                covariance_matrix=covariance_matrix,
                gwas_result_path=gwas_path.as_posix(),
                output_file_path=output_file_path.as_posix(),
                num_covar=request.cohort.num_covar,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error in indirect GWAS: {e}"
            ) from e

        # Upload the result to S3
        logger.info(
            f"Uploading result to S3. Dry: {dry_run}; {s3_region} - {s3_bucket}"
        )
        logger.debug(f"Output file: {output_file_path}")
        try:
            s3_client = get_s3_client(dry_run, s3_bucket, s3_region)
        except Exception as e:
            logger.error(f"Error getting S3 client: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error getting S3 client: {e}"
            ) from e
        logger.debug("Doing upload now")
        try:
            s3_client.upload_file(output_file_path.as_posix(), output_file_path.name)
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error uploading file: {e}"
            ) from e

    logger.info("Getting presigned URL")
    presigned_url = s3_client.get_presigned_url(output_file_path.name)
    logger.debug(f"Presigned URL: {presigned_url}")
    return WebGWASResult(request_id=request.id, url=presigned_url, status="done")
