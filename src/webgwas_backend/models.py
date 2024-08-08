import uuid
from typing import Literal

from pydantic import DirectoryPath
from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint
from webgwas.phenotype_definitions import NodeType


class Cohort(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    root_directory: DirectoryPath = Field(unique=True)

    features: list["Feature"] = Relationship(back_populates="cohort")


class Feature(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("cohort_id", "code", "name", name="unique_feature"),
    )

    id: int | None = Field(default=None, primary_key=True)
    code: str
    name: str
    type: NodeType

    cohort_id: int | None = Field(default=None, foreign_key="cohort.id")
    cohort: Cohort | None = Relationship(back_populates="features")


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

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phenotype_definition: str
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
