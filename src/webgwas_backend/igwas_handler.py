import logging
import pathlib
import tempfile

import pandas as pd
import polars as pl
import webgwas.igwas
import webgwas.phenotype_definitions
import webgwas.regression
from fastapi import HTTPException
from pandas import Series

from webgwas_backend.config import Settings
from webgwas_backend.models import WebGWASRequestID, WebGWASResult
from webgwas_backend.s3_client import S3Client

logger = logging.getLogger("uvicorn")


def get_igwas_coef(request: WebGWASRequestID) -> Series:
    directory = pathlib.Path(request.cohort.root_directory)

    # Load feature data
    logger.info("Loading data")
    features_path = directory.joinpath("phenotype_data.parquet")
    features_df = pl.read_parquet(features_path).to_pandas()

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
    request: WebGWASRequestID, s3_client: S3Client, settings: Settings
) -> WebGWASResult:
    beta_series = get_igwas_coef(request)
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Writing beta file")
        beta_file_path = pathlib.Path(temp_dir).joinpath(f"{request.id}.csv").as_posix()
        beta_series.to_frame().to_csv(beta_file_path)
        logger.debug(f"Beta file written to {beta_file_path}")
        logger.info("Running Indirect GWAS")
        output_file_path = pathlib.Path(temp_dir).joinpath(f"{request.id}.tsv.zst")
        cov_path = (
            pathlib.Path(request.cohort.root_directory)
            .joinpath("phenotypic_covariance.csv")
            .as_posix()
        )
        gwas_paths = [p.as_posix() for p in request.cohort.get_gwas_paths()]
        try:
            webgwas.igwas.igwas_files(
                projection_matrix_path=beta_file_path,
                covariance_matrix_path=cov_path,
                gwas_result_paths=gwas_paths,
                output_file_path=output_file_path.as_posix(),
                num_covar=request.cohort.num_covar,
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
        logger.info("Uploading result to S3")
        s3_client.upload_file(output_file_path.as_posix(), output_file_path.name)

    logger.info("Getting presigned URL")
    presigned_url = s3_client.get_presigned_url(output_file_path.name)
    logger.debug(f"Presigned URL: {presigned_url}")
    return WebGWASResult(request_id=request.id, url=presigned_url, status="done")
