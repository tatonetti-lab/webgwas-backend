import uuid

from pydantic import BaseModel, Field

from webgwas_backend.data_client import GWASCohort


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
    status: str = Field(..., description="Status of the request.")


class WebGWASResult(BaseModel):
    """Result of a successful GWAS"""

    request_id: str = Field(..., description="Unique identifier for the request.")
    status: str = Field(..., description="Status of the request.")
    error_msg: str | None = Field(
        default=None, description="Error message if status is 'error'."
    )
    url: str | None = Field(default=None, description="URL to the result file.")
