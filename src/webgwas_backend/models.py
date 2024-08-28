import uuid
from pathlib import Path
from typing import Literal

from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint
from webgwas.phenotype_definitions import Node, NodeType


class CohortResponse(SQLModel):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)


class Cohort(CohortResponse, SQLModel, table=True):
    root_directory: str = Field(unique=True)
    features: list["Feature"] = Relationship(back_populates="cohort")
    num_covar: int

    def get_gwas_paths(self) -> list[Path]:
        return [
            Path(self.root_directory)
            .joinpath("gwas")
            .joinpath(f"{feature.code}.tsv.zst")
            for feature in self.features
        ]


class FeatureResponse(SQLModel):
    id: int | None = Field(default=None, primary_key=True)
    code: str
    name: str
    type: NodeType


class Feature(FeatureResponse, SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("cohort_id", "code", "name", name="unique_feature"),
    )
    cohort_id: int | None = Field(default=None, foreign_key="cohort.id")
    cohort: Cohort | None = Relationship(back_populates="features")


class ValidPhenotypeResponse(SQLModel):
    is_valid: bool
    message: str
    phenotype_definition: str


class ValidPhenotype(ValidPhenotypeResponse, SQLModel):
    valid_nodes: list[Node]


class WebGWASRequest(SQLModel):
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


class WebGWASRequestID(SQLModel):
    """Internal use only"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phenotype_definition: ValidPhenotype
    cohort: Cohort


class WebGWASResponse(SQLModel):
    """Response to a request for GWAS summary statistics"""

    request_id: str = Field(..., description="Unique identifier for the request.")
    status: Literal["queued", "done", "error"] = Field(
        ..., description="Status of the request."
    )


class WebGWASResult(WebGWASResponse):
    """Result of a successful GWAS"""

    error_msg: str | None = Field(
        default=None, description="Error message if status is 'error'."
    )
    url: str | None = Field(default=None, description="URL to the result file.")


class ApproximatePhenotypeValues(SQLModel):
    t: float = Field(description="True value of the feature")
    a: float = Field(description="Approximate value of the feature")
    n: int = Field(description="Number of samples with this value")


class PhenotypeSummary(SQLModel):
    """Summary of a phenotype definition"""

    phenotype_definition: str = Field(
        ...,
        description=("Phenotype definition in reverse polish notation. "),
    )
    cohort_name: str = Field(
        ...,
        description=("Cohort for which to run the GWAS. "),
    )
    phenotypes: list[ApproximatePhenotypeValues] = Field(
        ...,
        description=(
            "True and approximate phenotype values for each sample in the "
            "anonymized cohort"
        ),
    )
    rsquared: float = Field(..., description="R-squared of the phenotype definition")

    def subsample(self, n_samples: int) -> "PhenotypeSummary":
        phenotypes = [
            ApproximatePhenotypeValues(
                t=p.t,
                a=p.a,
                n=p.n,
            )
            for p in self.phenotypes[:n_samples]
        ]
        return PhenotypeSummary(
            phenotype_definition=self.phenotype_definition,
            cohort_name=self.cohort_name,
            phenotypes=phenotypes,
            rsquared=self.rsquared,
        )
