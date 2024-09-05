import concurrent.futures
import logging
from contextlib import asynccontextmanager

from boto3.session import boto3
from botocore.config import Config
from fastapi import FastAPI

from webgwas_backend.s3_client import get_s3_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


worker = None


class TestWorker:
    def __init__(self):
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)

    def submit(self):
        self.executor.submit(self.handle_request)

    @staticmethod
    def handle_request():
        logger.info("Starting test")
        s3_client = boto3.client(
            "s3",
            region_name="us-west-1",
            verify=True,
            config=Config(signature_version="v4"),
        )
        logger.info("Uploading file")
        s3_client.upload_file("pyproject.toml", "webgwas", "pyproject.toml")
        logger.info("Done")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global worker
    worker = TestWorker()
    yield
    worker.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    try:
        assert worker is not None, "Worker not initialized"
        worker.submit()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
