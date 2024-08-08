import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import DirectoryPath
from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint
from webgwas.phenotype_definitions import Node, NodeType


class CohortResponse(SQLModel):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)


class Cohort(CohortResponse, SQLModel, table=True):
    root_directory: DirectoryPath = Field(unique=True)
    features: list["Feature"] = Relationship(back_populates="cohort")

    def get_gwas_paths(self) -> list[Path]:
        return [
            Path(self.root_directory).joinpath(f"{feature.code}.tsv.zst")
            for feature in self.features
        ]

    def get_num_covar(self) -> int:
        with open(Path(self.root_directory).joinpath("metadata.json")) as f:
            metadata = json.load(f)
        return metadata["num_covar"]


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


class WebGWASResult(SQLModel):
    """Result of a successful GWAS"""

    request_id: str = Field(..., description="Unique identifier for the request.")
    status: str = Field(..., description="Status of the request.")
    error_msg: str | None = Field(
        default=None, description="Error message if status is 'error'."
    )
    url: str | None = Field(default=None, description="URL to the result file.")
