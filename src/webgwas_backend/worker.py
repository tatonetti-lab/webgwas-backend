import logging
from queue import Queue

from webgwas_backend.config import Settings
from webgwas_backend.igwas_handler import handle_igwas
from webgwas_backend.models import WebGWASRequestID, WebGWASResult
from webgwas_backend.s3_client import S3Client

logger = logging.getLogger("uvicorn")


class Worker:
    def __init__(
        self,
        job_queue: Queue[WebGWASRequestID],
        queued_request_ids: set[str],
        results: dict[str, WebGWASResult],
        s3_client: S3Client,
        settings: Settings,
    ):
        self.job_queue = job_queue
        self.queued_request_ids = queued_request_ids
        self.results = results
        self.s3_client = s3_client
        self.settings = settings

    def run(self):
        logger.info("Starting worker")
        while True:
            request = self.job_queue.get()
            logger.info(f"Got request: {request}")
            try:
                result = handle_igwas(request, self.s3_client)
                self.results[request.id] = result
                logger.info(f"Finished request: {request.id}")
            except Exception as e:
                msg = f"{e}"
                logger.error(msg)
                self.results[request.id] = WebGWASResult(
                    request_id=request.id, status="error", error_msg=msg
                )
            self.job_queue.task_done()

            self.queued_request_ids.remove(request.id)
