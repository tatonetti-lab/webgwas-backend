import logging
import pathlib
from typing import Annotated, Optional

import pandas as pd
import polars as pl
import typer
import webgwas.mdav
import webgwas.regression
from pydantic import BaseModel
from rich.logging import RichHandler
from rich.progress import track
from sqlmodel import Session, select
from webgwas.phenotype_definitions import Field

from webgwas_backend.database import db_exists, engine, init_db
from webgwas_backend.models import Cohort, Feature

logger = logging.getLogger(__name__)


class InputFiles(BaseModel):
    phenotype_path: pathlib.Path | None = None
    phenotype_file_separator: str = "\t"
    phenotype_person_id_col: str = "eid"
    covariate_path: pathlib.Path | None = None
    covariate_file_separator: str = "\t"
    covariate_person_id_col: str = "eid"
    gwas_paths: list[pathlib.Path] | None = None
    gwas_file_extension: str = ".tsv.zst"


class StepsCompleted(BaseModel):
    phenotypes_covariates: bool = False
    gwas: bool = False
    feature_map: bool = False

    def all(self):
        return all([self.phenotypes_covariates, self.gwas, self.feature_map])


class CohortFiles:
    def __init__(self, name: str, root: pathlib.Path):
        self.name = name
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.gwas_root = root.joinpath("gwas")
        self.phenotype_path = root.joinpath("phenotype_data.parquet")
        self.left_inverse_path = root.joinpath("phenotype_left_inverse.parquet")
        self.covariance_path = root.joinpath("phenotypic_covariance.csv")
        self.features: list[str] | None = None
        self.feature_code_to_info: dict[str, Field] | None = None
        self.num_covar: int | None = None
        self.inputs: InputFiles = InputFiles()
        self.steps_completed: StepsCompleted = StepsCompleted()

    def register_phenotypes_covariates(
        self,
        phenotype_path: pathlib.Path,
        covariate_path: pathlib.Path,
        phenotype_file_separator: str = "\t",
        covariate_file_separator: str = "\t",
        phenotype_person_id_col: str = "eid",
        covariate_person_id_col: str = "eid",
    ) -> None:
        self.num_covar = len(
            pl.read_csv(covariate_path, separator=covariate_file_separator, n_rows=0)
            .drop(covariate_person_id_col)
            .columns
        )
        logger.info(f"Found {self.num_covar} covariates")
        feature_names = (
            pl.read_csv(phenotype_path, separator=phenotype_file_separator, n_rows=0)
            .drop(phenotype_person_id_col)
            .columns
        )
        logger.info(f"Found {len(feature_names)} features in phenotype file")
        if self.features is None:
            self.features = list(feature_names)
        else:
            self.features = list(set(self.features).intersection(feature_names))
            if len(self.features) == 0:
                raise ValueError(f"No features found: {self.features[:3]}")
        logger.info(f"Result in {len(self.features)} features")
        self.inputs.phenotype_path = phenotype_path
        self.inputs.covariate_path = covariate_path
        self.inputs.phenotype_file_separator = phenotype_file_separator
        self.inputs.phenotype_person_id_col = phenotype_person_id_col
        self.inputs.covariate_file_separator = covariate_file_separator
        self.inputs.covariate_person_id_col = phenotype_person_id_col

    def register_gwas(
        self,
        original_gwas_root: pathlib.Path,
        extension: str = ".tsv.zst",
    ) -> None:
        gwas_paths = sorted(original_gwas_root.glob(f"*{extension}"))
        logger.info(f"Found {len(gwas_paths)} GWAS files")
        gwas_features = [p.name.replace(extension, "") for p in gwas_paths]
        if self.features is None:
            self.features = gwas_features
        else:
            gwas_features_set = sorted(set(gwas_features).intersection(self.features))
            if len(gwas_features_set) == 0:
                raise ValueError(
                    f"No GWAS features found: {self.features[:3]} not in "
                    f"{gwas_features[:3]}"
                )
            self.inputs.gwas_paths = [
                p
                for p in gwas_paths
                if p.name.replace(extension, "") in gwas_features_set
            ]
            if len(self.inputs.gwas_paths) == 0:
                raise ValueError(
                    f"No GWAS files found: {self.features[:3]} not in "
                    f"{gwas_features[:3]}"
                )
            self.inputs.gwas_file_extension = extension

    def process_phenotypes_covariates(
        self,
        k_anonymity: int = 10,
    ) -> None:
        if self.inputs.phenotype_path is None:
            raise ValueError("No phenotype file specified")
        if self.inputs.covariate_path is None:
            raise ValueError("No covariate file specified")
        phenotype_df = pl.read_csv(
            self.inputs.phenotype_path,
            separator=self.inputs.phenotype_file_separator,
        ).rename({self.inputs.phenotype_person_id_col: "eid"})
        covariate_df = pl.read_csv(
            self.inputs.covariate_path,
            separator=self.inputs.covariate_file_separator,
        ).rename({self.inputs.covariate_person_id_col: "eid"})
        covariate_names = covariate_df.drop("eid").columns
        merged_df = phenotype_df.join(covariate_df, on="eid", how="inner")
        X = merged_df.select(covariate_names).to_pandas().assign(const=1.0)
        Y = merged_df.select(self.features).to_pandas()
        logger.info("Residualizing phenotypes")
        residualized_Y = webgwas.regression.residualize(Y, X)
        del X  # Free up memory
        logger.info("Computing covariance matrix")
        covariance_matrix = residualized_Y.cov()
        del residualized_Y  # Free up memory
        logger.info("Writing covariance matrix")
        covariance_matrix.to_csv(self.covariance_path)
        del covariance_matrix  # Free up memory
        if k_anonymity > 0:
            logger.info("Anonymizing phenotypes")
            anon_array = webgwas.mdav.mdav(Y.values[:10000], k_anonymity)
            anon_df = pd.DataFrame(anon_array, columns=Y.columns)
            del anon_array, Y  # Free up memory
            logger.info("Writing anonymized phenotypes")
            anon_df.to_parquet(self.phenotype_path)
            Y = anon_df
        else:
            logger.info("Writing phenotypes")
            Y.to_parquet(self.phenotype_path)
        logger.info("Computing left inverse")
        left_inverse = webgwas.regression.compute_left_inverse(Y)
        del Y  # Free up memory
        logger.info("Writing left inverse")
        left_inverse.T.to_parquet(self.left_inverse_path)
        self.steps_completed.phenotypes_covariates = True

    def process_gwas(
        self,
        variant_id: str = "ID",
        beta: str = "BETA",
        std_error: str = "SE",
        sample_size: str = "OBS_CT",
        separator: str = "\t",
        keep_n_variants: int | None = None,
    ) -> None:
        if self.inputs.gwas_paths is None:
            raise ValueError("No GWAS files specified")
        self.gwas_root.mkdir(parents=True, exist_ok=True)
        logger.info("Reading GWAS files")
        schema_overrides = {
            variant_id: pl.Utf8,
            beta: pl.Float64,
            std_error: pl.Float64,
            sample_size: pl.Int64,
        }
        gwas_features = [
            p.name.replace(self.inputs.gwas_file_extension, "")
            for p in self.inputs.gwas_paths
        ]
        for gwas_path, feature_name in track(
            zip(self.inputs.gwas_paths, gwas_features),
            total=len(self.inputs.gwas_paths),
            description="Ingesting GWAS files",
        ):
            output_path = self.gwas_root.joinpath(feature_name).with_suffix(".tsv.zst")
            pl.read_csv(
                gwas_path,
                separator=separator,
                schema_overrides=schema_overrides,
                n_rows=keep_n_variants,
            ).write_csv(output_path, separator="\t")
        self.steps_completed.gwas = True

    def register_feature_map(self, feature_code_to_info: dict[str, Field]) -> None:
        self.feature_code_to_info = feature_code_to_info
        if self.features is None:
            self.features = sorted(self.feature_code_to_info.keys())
        else:
            self.features = list(
                set(self.features).intersection(self.feature_code_to_info.keys())
            )
        self.steps_completed.feature_map = True

    def register_feature_map_file(
        self, feature_map_path: pathlib.Path, separator: str
    ) -> None:
        self.register_feature_map(
            {
                k: Field.model_validate(
                    {
                        "name": row["name"],
                        "code": row["code"],
                        "type": row["type"],
                    }
                )
                for k, row in pl.read_csv(feature_map_path, separator=separator)
                .rows_by_key("code", named=True, include_key=True, unique=True)
                .items()
            }
        )

    def write_database(self, session: Session) -> None:
        if not self.steps_completed.all():
            raise ValueError("Not all steps completed")
        if self.features is None:
            raise ValueError("Features not registered")
        if self.feature_code_to_info is None:
            raise ValueError("Feature map not registered")
        cohort = Cohort.model_validate(
            {
                "name": self.name,
                "root_directory": self.root.absolute().as_posix(),
                "num_covar": self.num_covar,
            }
        )
        session.add(cohort)
        session.commit()
        session.refresh(cohort)
        for feature_code in sorted(self.features):
            info = self.feature_code_to_info[feature_code]
            if info.name is None:
                raise ValueError(f"Feature {feature_code} has no name")
            if info.type is None:
                raise ValueError(f"Feature {feature_code} has no type")
            feature = Feature(
                cohort=cohort,
                code=feature_code,
                name=info.name,
                type=info.type,
            )
            session.add(feature)
        session.commit()

    def validate(self, session: Session) -> None:
        cohort = session.exec(select(Cohort).where(Cohort.name == self.name)).one()
        if cohort is None:
            raise ValueError(f"Cohort {self.name} not found")
        features = session.exec(select(Feature).where(Feature.cohort == cohort)).all()
        feature_codes = [f.code for f in features]
        if len(features) == 0:
            raise ValueError(f"Cohort {self.name} has no features")
        phenotype_df = pl.read_parquet(self.phenotype_path, n_rows=1)
        phenotype_features = set(phenotype_df.columns)
        assert phenotype_features == set(feature_codes), [
            f for f in phenotype_features if f not in feature_codes
        ] + [f for f in feature_codes if f not in phenotype_features]
        covariance_df = pd.read_csv(self.covariance_path, index_col=0)
        assert set(covariance_df.columns) == set(feature_codes)
        assert set(covariance_df.index) == set(feature_codes)
        left_inverse_df = pl.read_parquet(self.left_inverse_path, n_rows=1).to_pandas()
        left_inverse_features = set(left_inverse_df.columns)
        assert left_inverse_features == set(feature_codes)


def cohort_table_exists() -> bool:
    return engine.dialect.has_table(engine.connect(), "cohort")


def cohort_already_exists(cohort_name: str) -> bool:
    if db_exists():
        if cohort_table_exists():
            with Session(engine) as session:
                cohort = session.exec(
                    select(Cohort).where(Cohort.name == cohort_name)
                ).one_or_none()
                if cohort is not None:
                    return True
    return False


def register_cohort(
    *,
    cohort_name: str,
    output_root: pathlib.Path,
    original_phenotype_path: Annotated[pathlib.Path, typer.Option()],
    original_covariate_path: Annotated[pathlib.Path, typer.Option()],
    original_gwas_root: Annotated[pathlib.Path, typer.Option()],
    feature_map_path: Annotated[pathlib.Path, typer.Option()],
    phenotype_file_separator: str = "\t",
    covariate_file_separator: str = "\t",
    phenotype_person_id_col: str = "eid",
    covariate_person_id_col: str = "eid",
    k_anonymity: int = 10,
    gwas_file_extension: str = ".tsv.zst",
    variant_id: str = "ID",
    beta: str = "BETA",
    std_error: str = "SE",
    sample_size: str = "OBS_CT",
    gwas_separator: str = "\t",
    keep_n_variants: Optional[int] = None,
) -> None:
    if cohort_already_exists(cohort_name):
        raise ValueError(f"Cohort {cohort_name} already exists")
    logger.info("Creating cohort directory")
    cohort = CohortFiles(cohort_name, output_root)
    logger.info("Registering phenotypes and covariates")
    cohort.register_phenotypes_covariates(
        phenotype_path=original_phenotype_path,
        covariate_path=original_covariate_path,
        phenotype_file_separator=phenotype_file_separator,
        covariate_file_separator=covariate_file_separator,
        phenotype_person_id_col=phenotype_person_id_col,
        covariate_person_id_col=covariate_person_id_col,
    )
    logger.info("Registering GWAS files")
    cohort.register_gwas(
        original_gwas_root=original_gwas_root,
        extension=gwas_file_extension,
    )
    logger.info("Registering feature map")
    cohort.register_feature_map_file(feature_map_path, phenotype_file_separator)
    logger.info("Processing phenotypes and covariates")
    cohort.process_phenotypes_covariates(k_anonymity=k_anonymity)
    logger.info("Processing GWAS files")
    cohort.process_gwas(
        variant_id=variant_id,
        beta=beta,
        std_error=std_error,
        sample_size=sample_size,
        separator=gwas_separator,
        keep_n_variants=keep_n_variants,
    )
    if not db_exists() or not cohort_table_exists():
        logger.info("Initializing database")
        init_db()

    logger.info("Writing cohort and features to database")
    with Session(engine) as session:
        cohort.write_database(session)
        logger.info("Done, validating cohort")
        cohort.validate(session)
    logger.info(f"Done, {cohort.name} registered successfully")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler()]
    )
    typer.run(register_cohort)
    return 0