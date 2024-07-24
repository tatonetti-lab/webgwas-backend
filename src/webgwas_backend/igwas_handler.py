import pathlib
import tempfile

import webgwas.igwas
import webgwas.parser
import webgwas.regression
from fastapi import HTTPException
from pandas import Series

from webgwas_backend.config import Settings
from webgwas_backend.data_client import DataClient, GWASCohort
from webgwas_backend.models import WebGWASResult
from webgwas_backend.s3_client import S3Client


def handle_igwas(
    settings: Settings,
    data_client: DataClient,
    s3_client: S3Client,
    request_id: str,
    phenotype_definition: str,
    cohort: GWASCohort,
) -> WebGWASResult:
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

    return WebGWASResult(request_id=request_id, url=presigned_url, status="done")
