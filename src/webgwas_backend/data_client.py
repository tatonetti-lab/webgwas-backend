from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, DirectoryPath, FilePath

logger = logging.getLogger("uvicorn")


class GWASCohort(BaseModel):
    """Wrapper for a set of GWAS results in a cohort

    The cohort directory is assumed to be laid out as follows:

    ```
    cohort_name/
        metadata.json
        phenotypic_covariance.csv
        phenotype_data.csv.zst
        phenotype_left_inverse.csv.zst
        gwas/
            feature1.tsv.zst
            feature2.tsv.zst
            ...
    ```

    The phenotypic covariance matrix is assumed to be adjusted for the same
    covariates as were used in the GWAS.

    The phenotype data matrix is NOT assumed to be the same as the GWAS data
    matrix--it is just assumed to have similar statistical properties and to
    have the same features.

    The phenotype left inverse also does not need to have been computed on the
    original features, just on the phenotype data matrix above. For reference,
    given a phenotype matrix `X`, the phenotype left inverse is `inv(X'X)X'`.
    This matrix should be stored as the transpose of the phenotype left inverse
    (i.e. store as `(inv(X'X)X')'`).

    `metadata.json` file must contain the following fields:
        cohort_name: string with the same name as the directory
        feature_names: list of strings with the names of the features that
            appear in the phenotypic covariance matrix, in the GWAS files, and
            in the phenotype data matrix.
        gwas_extension: string with the extension of the GWAS files (e.g. ".tsv")
        num_covar: number of covariates in the GWAS and phenotypic covariance
            matrix

    GWAS files should be named like `<feature_name><gwas_extension>`, and there
    should be one file for each feature. These should be tab-delimited files
    with the following columns (at least):
        - ID: variant ID
        - BETA: effect size estimate
        - SE: standard error of the effect size estimate
        - OBS_CT: sample size for that variant
    """

    _covariance_filename: ClassVar[str] = "phenotypic_covariance.csv"
    _data_filename: ClassVar[str] = "phenotype_data.csv.zst"
    _left_inverse_filename: ClassVar[str] = "phenotype_left_inverse.csv.zst"
    _gwas_directory: ClassVar[str] = "gwas"

    cohort_name: str
    feature_names: list[str]
    root: DirectoryPath
    covariance_path: FilePath
    data_path: FilePath
    left_inverse_path: FilePath
    gwas_paths: list[FilePath]
    num_covar: int

    @classmethod
    def from_path(cls, root: DirectoryPath) -> GWASCohort:
        """Load a cohort from a directory.

        Automatically validates that all files exist (via model_validate) and
        that the features are consistent between the covariance matrix, the
        data matrix, the GWAS files, and the metadata file.
        """
        metadata_path = root.joinpath("metadata.json")
        if not metadata_path.exists():
            raise ValueError(
                f"Invalid cohort directory. {metadata_path} does not exist."
            )
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["root"] = root
        metadata["covariance_path"] = root.joinpath(cls._covariance_filename)
        metadata["data_path"] = root.joinpath(cls._data_filename)
        metadata["left_inverse_path"] = root.joinpath(cls._left_inverse_filename)
        metadata["gwas_paths"] = [
            root.joinpath(cls._gwas_directory).joinpath(
                f"{feature}{metadata['gwas_extension']}"
            )
            for feature in metadata["feature_names"]
        ]
        cov_df = pd.read_csv(metadata["covariance_path"], index_col=0)
        if set(cov_df.columns.tolist()) != set(metadata["feature_names"]) or set(
            cov_df.index.tolist()
        ) != set(metadata["feature_names"]):
            raise ValueError(
                f"Invalid cohort directory. Covariance matrix does not "
                f"contain all features. Expected {metadata['feature_names']}, "
                f"found {cov_df.columns} and {cov_df.index}"
            )
        data_df = pd.read_csv(metadata["data_path"], nrows=0)
        if not set(data_df.columns) == set(metadata["feature_names"]):
            raise ValueError(
                f"Invalid cohort directory. Data matrix does not "
                f"contain all features. Expected {metadata['feature_names']}, "
                f"found {data_df.columns}"
            )
        left_inverse_df = pd.read_csv(metadata["left_inverse_path"], nrows=0)
        if not set(left_inverse_df.columns) == set(metadata["feature_names"]):
            raise ValueError(
                f"Invalid cohort directory. Data matrix does not "
                f"contain all features. Expected {metadata['feature_names']}, "
                f"found {left_inverse_df.columns}"
            )
        return cls.model_validate(metadata)


class DataClient(BaseModel):
    name_to_cohort: dict[str, GWASCohort]

    @classmethod
    def from_paths(cls, cohort_paths: list[DirectoryPath]) -> DataClient:
        name_to_cohort = {}
        for cohort_path in cohort_paths:
            name_to_cohort[cohort_path.name] = GWASCohort.from_path(cohort_path)
        return cls(name_to_cohort=name_to_cohort)

    def validate_cohort(self, cohort: str) -> GWASCohort | None:
        return self.name_to_cohort.get(cohort)

    def get_cohorts(self) -> list[str]:
        return sorted(self.name_to_cohort.keys())

    def get_features(self, cohort: str) -> list[str] | None:
        cohort_obj = self.name_to_cohort.get(cohort)
        if cohort_obj is None:
            return None
        return cohort_obj.feature_names

    def get_data(self, cohort: str) -> pd.DataFrame | None:
        cohort_obj = self.name_to_cohort.get(cohort)
        if cohort_obj is None:
            return None
        return pd.read_csv(cohort_obj.data_path)

    def get_left_inverse(self, cohort: str) -> pd.DataFrame | None:
        cohort_obj = self.name_to_cohort.get(cohort)
        if cohort_obj is None:
            return None
        return pd.read_csv(cohort_obj.left_inverse_path).T

    def get_covariance_path(self, cohort: str) -> Path | None:
        cohort_obj = self.name_to_cohort.get(cohort)
        if cohort_obj is None:
            return None
        return cohort_obj.covariance_path

    def get_gwas_paths(self, cohort: str) -> list[Path] | None:
        cohort_obj = self.name_to_cohort.get(cohort)
        if cohort_obj is None:
            return None
        return cohort_obj.gwas_paths

    def get_data_cov_gwas_unchecked(
        self, cohort: str
    ) -> tuple[pd.DataFrame, Path, list[Path]]:
        cohort_obj = self.name_to_cohort[cohort]
        return (
            pd.read_csv(cohort_obj.data_path),
            cohort_obj.covariance_path,
            cohort_obj.gwas_paths,
        )
