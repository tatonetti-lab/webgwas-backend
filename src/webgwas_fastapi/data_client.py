from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class DataClient(BaseModel):
    data_path: Path
    cohorts: list[str]
    cohort_to_features: dict[str, list[str]]
    cohort_to_data: dict[str, str]
    cohort_to_covariance: dict[str, Path]
    cohort_to_gwas: dict[str, list[Path]]

    def validate_cohort(self, cohort: str) -> str | None:
        if cohort not in self.cohorts:
            return None
        return cohort

    def get_cohorts(self) -> list[str]:
        return sorted(self.cohorts)

    def get_features(self, cohort: str) -> list[str] | None:
        return self.cohort_to_features.get(cohort)

    def get_data(self, cohort: str) -> pd.DataFrame | None:
        path = self.cohort_to_data.get(cohort)
        if path is None:
            return None
        return pd.read_csv(self.data_path.joinpath(path))

    def get_covariance_path(self, cohort: str) -> Path | None:
        result = self.cohort_to_covariance.get(cohort)
        if result is None:
            return None
        return self.data_path.joinpath(result)

    def get_gwas_paths(self, cohort: str) -> list[Path] | None:
        result = self.cohort_to_gwas.get(cohort)
        if result is None:
            return None
        return [self.data_path.joinpath(path) for path in result]

    def get_data_cov_gwas_unchecked(
        self, cohort: str
    ) -> tuple[pd.DataFrame, Path, list[Path]]:
        return (
            pd.read_csv(self.data_path.joinpath(self.cohort_to_data[cohort])),
            self.data_path.joinpath(self.cohort_to_covariance[cohort]),
            [self.data_path.joinpath(x) for x in self.cohort_to_gwas[cohort]],
        )
