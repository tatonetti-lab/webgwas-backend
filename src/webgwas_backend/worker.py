import logging
import time
from queue import Queue

from webgwas_backend.config import Settings
from webgwas_backend.data_client import DataClient
from webgwas_backend.igwas_handler import handle_igwas
from webgwas_backend.models import WebGWASRequestID, WebGWASResult
from webgwas_backend.s3_client import S3Client

logger = logging.getLogger(__name__)


def worker_function(
    job_queue: Queue[WebGWASRequestID],
    queued_request_ids: set[str],
    results: dict[str, WebGWASResult],
    settings: Settings,
    data_client: DataClient,
    s3_client: S3Client,
):
    while True:
        request = job_queue.get()

        try:
            result = handle_igwas(
                settings=settings,
                data_client=data_client,
                s3_client=s3_client,
                phenotype_definition=request.phenotype_definition,
                cohort=request.cohort,
                request_id=request.request_id,
            )
            time.sleep(10)
            results[request.request_id] = result
        except Exception as e:
            msg = f"{e}"
            logger.error(msg)
            results[request.request_id] = WebGWASResult(
                request_id=request.request_id, status="error", error_msg=msg
            )

        job_queue.task_done()
        queued_request_ids.remove(request.request_id)


class Worker:
    def __init__(
        self,
        job_queue: Queue,
        queued_request_ids: set[str],
        results: dict[str, WebGWASResult],
        settings: Settings,
        data_client: DataClient,
        s3_client: S3Client,
    ):
        self.job_queue = job_queue
        self.queued_request_ids = queued_request_ids
        self.results = results
        self.settings = settings
        self.data_client = data_client
        self.s3_client = s3_client

    def run(self):
        worker_function(
            self.job_queue,
            self.queued_request_ids,
            self.results,
            self.settings,
            self.data_client,
            self.s3_client,
        )
