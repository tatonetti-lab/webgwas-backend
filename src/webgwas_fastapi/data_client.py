from abc import ABC, abstractmethod

import pandas as pd


class DataClient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def validate_cohort(self, cohort: str) -> str | None:
        pass

    @abstractmethod
    def get_cohorts(self) -> list[str]:
        pass

    @abstractmethod
    def get_features(self, cohort: str) -> list[str] | None:
        pass

    @abstractmethod
    def get_data(self, cohort: str) -> pd.DataFrame | None:
        pass

    @abstractmethod
    def get_covariance_path(self, cohort: str) -> str | None:
        pass

    @abstractmethod
    def get_gwas_paths(self, cohort: str) -> list[str] | None:
        pass

    @abstractmethod
    def get_data_cov_gwas_unchecked(
        self, cohort: str
    ) -> tuple[pd.DataFrame, str, list[str]]:
        pass


class DataProdClient(DataClient):
    def __init__(self):
        super().__init__()

    def get_features(self, cohort: str) -> list[str] | None:
        raise NotImplementedError("Load features for cohort not implemented")

    def get_data(self, cohort: str) -> pd.DataFrame | None:
        raise NotImplementedError("Load data for cohort not implemented")


class DataMockClient(DataClient):
    def __init__(self):
        super().__init__()
        self.cohorts = {"cohort1", "cohort2"}
        self.cohort_to_features = {
            "cohort1": ["feature1", "feature2", "feature3"],
            "cohort2": ["feature1", "feature2", "feature3"],
        }
        self.cohort_to_data = {
            "cohort1": pd.DataFrame(
                [
                    {"feature1": 0, "feature2": 1, "feature3": 1},
                    {"feature1": 1, "feature2": 0, "feature3": 0},
                    {"feature1": 1, "feature2": 1, "feature3": 0},
                ]
            ),
            "cohort2": pd.DataFrame(
                [
                    {"feature1": 0, "feature2": 1, "feature3": 1},
                    {"feature1": 1, "feature2": 0, "feature3": 0},
                    {"feature1": 1, "feature2": 1, "feature3": 0},
                ]
            ),
        }
        self.data_path = "/Users/zietzm/Documents/projects/webgwas-frontend/webgwas-fastapi/test_data/"
        self.cohort_to_covariance = {
            "cohort1": "covariance_matrix_1.csv",
            "cohort2": "covariance_matrix_2.csv",
        }
        self.cohort_to_gwas = {
            "cohort1": ["feature1.tsv", "feature2.tsv", "feature3.tsv"],
            "cohort2": ["feature1.tsv", "feature2.tsv", "feature3.tsv"],
        }
        self.initialize()

    def initialize(self):
        features = pd.Index(["feature1", "feature2", "feature3"])
        covariance_matrix_1 = pd.DataFrame(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            index=features,
            columns=features,
        )
        covariance_matrix_1.to_csv(self.data_path + "covariance_matrix_1.csv")
        covariance_matrix_2 = pd.DataFrame(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            index=features,
            columns=features,
        )
        covariance_matrix_2.to_csv(self.data_path + "covariance_matrix_2.csv")
        gwas_result = pd.DataFrame(
            [
                ["1", 0.1, 0.2, 100],
                ["2", 0.2, 0.3, 100],
                ["3", 0.3, 0.4, 100],
            ],
            columns=pd.Index(["ID", "BETA", "SE", "OBS_CT"]),
        )
        gwas_result.to_csv(self.data_path + "feature1.tsv", index=False, sep="\t")
        gwas_result.to_csv(self.data_path + "feature2.tsv", index=False, sep="\t")
        gwas_result.to_csv(self.data_path + "feature3.tsv", index=False, sep="\t")

    def validate_cohort(self, cohort: str) -> str | None:
        if cohort not in self.cohorts:
            return None
        return cohort

    def get_cohorts(self) -> list[str]:
        return sorted(self.cohorts)

    def get_features(self, cohort: str) -> list[str] | None:
        return self.cohort_to_features.get(cohort)

    def get_data(self, cohort: str) -> pd.DataFrame | None:
        return self.cohort_to_data.get(cohort)

    def get_covariance_path(self, cohort: str) -> str | None:
        result = self.cohort_to_covariance.get(cohort)
        if result is None:
            return None
        return self.data_path + result

    def get_gwas_paths(self, cohort: str) -> list[str] | None:
        result = self.cohort_to_gwas.get(cohort)
        if result is None:
            return None
        return [self.data_path + path for path in result]

    def get_data_cov_gwas_unchecked(
        self, cohort: str
    ) -> tuple[pd.DataFrame, str, list[str]]:
        return (
            self.cohort_to_data[cohort],
            self.data_path + self.cohort_to_covariance[cohort],
            [self.data_path + x for x in self.cohort_to_gwas[cohort]],
        )
