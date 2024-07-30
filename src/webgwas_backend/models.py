import uuid
from typing import Literal

from pydantic import BaseModel, Field

from webgwas_backend.data_client import GWASCohort


class PhenotypeNode(BaseModel):
    """A node in the phenotype definition tree"""

    id: int = Field(..., description="Unique identifier for the node")
    type: Literal["field", "operator", "constant"]
    name: str = Field(..., description="Name of the node")
    min_arity: int | None = Field(
        0, serialization_alias="minArity", description="Minimum number of operands"
    )
    max_arity: int | None = Field(
        0, serialization_alias="maxArity", description="Maximum number of operands"
    )


class WebGWASRequest(BaseModel):
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


class WebGWASRequestID(BaseModel):
    """Internal use only"""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phenotype_definition: str
    cohort: GWASCohort


class WebGWASResponse(BaseModel):
    """Response to a request for GWAS summary statistics"""

    request_id: str = Field(..., description="Unique identifier for the request.")
    status: Literal["queued", "done", "error"] = Field(
        ..., description="Status of the request."
    )


class WebGWASResult(BaseModel):
    """Result of a successful GWAS"""

    request_id: str = Field(..., description="Unique identifier for the request.")
    status: str = Field(..., description="Status of the request.")
    error_msg: str | None = Field(
        default=None, description="Error message if status is 'error'."
    )
    url: str | None = Field(default=None, description="URL to the result file.")
