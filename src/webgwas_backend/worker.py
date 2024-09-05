import logging
import pathlib
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager

from fastapi import HTTPException

from webgwas_backend.config import Settings
from webgwas_backend.igwas_handler import handle_igwas
from webgwas_backend.models import WebGWASRequestID, WebGWASResult

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings: Settings):
        self.s3_dry_run = settings.dry_run
        self.s3_bucket = settings.s3_bucket
        self.s3_result_path = settings.s3_result_path
        self.root_data_directory = settings.root_data_directory

        self.manager = Manager()
        self.lock = self.manager.Lock()
        self.results: dict[str, Future[WebGWASResult]] = dict()
        self.executor = ProcessPoolExecutor(max_workers=settings.n_workers)

    def submit(self, request: WebGWASRequestID):
        logger.info(f"Submitting request: {request}")
        with self.lock:
            self.results[request.id] = self.executor.submit(
                self.handle_request,
                request,
                self.root_data_directory,
                self.s3_dry_run,
                self.s3_bucket,
                self.s3_result_path,
            )
        logger.info(f"Queued request: {request.id}")

    @staticmethod
    def handle_request(
        request: WebGWASRequestID,
        root_data_directory: pathlib.Path,
        dry_run: bool,
        s3_bucket: str,
        s3_result_path: str,
    ):
        try:
            return handle_igwas(
                request, root_data_directory, dry_run, s3_bucket, s3_result_path
            )
        except Exception as e:
            return WebGWASResult(
                request_id=request.id, status="error", error_msg=f"{e}"
            )

    def get_results(self, request_id: str) -> WebGWASResult:
        with self.lock:
            future = self.results.get(request_id)
            if future is None:
                raise HTTPException(status_code=404, detail="Request not found")
            if future.running():
                return WebGWASResult(request_id=request_id, status="queued")
            if future.done():
                return future.result()
        raise HTTPException(status_code=500, detail=f"Internal error: {future}")

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.manager.shutdown()


class TestWorker(Worker):
    __test__ = False

    def __init__(self, settings: Settings):
        self.s3_dry_run = settings.dry_run
        self.s3_bucket = settings.s3_bucket
        self.id_to_result: dict[str, WebGWASResult] = {}

    def submit(self, request: WebGWASRequestID):
        logger.info(f"Submitting request: {request}")
        result = self.handle_request(
            request,
            self.root_data_directory,
            self.s3_dry_run,
            self.s3_bucket,
        )
        self.id_to_result[request.id] = result
        logger.info(f"Queued request: {request.id}")

    def get_results(self, request_id: str) -> WebGWASResult:
        return self.id_to_result[request_id]
