import logging

from webgwas_backend.cli.build_ukb_reduced import build as build_ukb_reduced

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> int:
    logger.info("Building UKB-WB-100k-reduced-anon")
    build_ukb_reduced()
    logger.info("Done")
    return 0
