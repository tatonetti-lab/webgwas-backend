import collections
import logging
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import sklearn.metrics
import webgwas.phenotype_definitions
import webgwas.regression
from pandas import Series

from webgwas_backend.models import (
    ApproximatePhenotypeValues,
    Cohort,
    PhenotypeFitQuality,
    PhenotypeSummary,
    ValidPhenotype,
)

logger = logging.getLogger(__name__)


def get_phenotype_summary(
    cohort: Cohort,
    phenotype_definition: ValidPhenotype,
    fit_quality: list[PhenotypeFitQuality],
    root_data_directory: pathlib.Path,
) -> PhenotypeSummary:
    directory = cohort.get_root_path(root_data_directory)

    # Load feature data
    logger.debug("Loading data")
    features_path = directory.joinpath("phenotype_data.parquet")
    features_df = pl.read_parquet(features_path).to_pandas()

    # Assign the target phenotype
    logger.debug("Applying phenotype definition to data")
    target_phenotype = webgwas.phenotype_definitions.apply_definition_pandas(
        nodes=phenotype_definition.valid_nodes, df=features_df
    )
    assert isinstance(target_phenotype, Series)
    logger.debug(f"Target phenotype: {target_phenotype}")

    # Load left inverse
    logger.debug("Loading left inverse")
    left_inverse_path = directory.joinpath("phenotype_left_inverse.parquet")
    left_inverse_df = pd.read_parquet(left_inverse_path).T
    logger.debug(f"Left inverse: {left_inverse_df}")

    # Regress the target phenotype against the feature phenotypes
    logger.debug("Regressing phenotype against features")
    beta_series = (
        webgwas.regression.regress_left_inverse(target_phenotype, left_inverse_df)
        .round(5)
        .rename(phenotype_definition.phenotype_definition)
        .rename_axis(index="feature")
    )
    yhat = features_df.assign(const=1.0) @ beta_series
    rsquared = sklearn.metrics.r2_score(target_phenotype, yhat)
    assert isinstance(rsquared, float)
    phenotype_counts = collections.Counter(zip(target_phenotype, yhat, strict=True))
    phenotypes = [
        ApproximatePhenotypeValues(
            t=float(
                np.format_float_positional(true_value, precision=3, fractional=False)
            ),
            a=float(
                np.format_float_positional(approx_value, precision=3, fractional=False)
            ),
            n=value,
        )
        for (true_value, approx_value), value in phenotype_counts.items()
    ]
    result = PhenotypeSummary(
        phenotype_definition=phenotype_definition.phenotype_definition,
        cohort_name=cohort.name,
        phenotypes=phenotypes,
        rsquared=rsquared,
        fit_quality=fit_quality,
    )
    return result
