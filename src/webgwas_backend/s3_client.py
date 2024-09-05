import logging
import os
import pathlib
import subprocess
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class S3Client(ABC):
    @abstractmethod
    def upload_file(self, local_path, key):
        pass

    @abstractmethod
    def get_presigned_url(self, key) -> str:
        pass


class S3ProdClient(S3Client):
    def __init__(self, bucket: str = "webgwas"):
        self.bucket = bucket

    def upload_file(self, local_path, key):
        result = subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                local_path,
                f"s3://{self.bucket}/{key}",
            ],
            capture_output=True,
        )
        logger.debug(f"Upload result: {result}")
        result.check_returncode()

    def get_presigned_url(self, key) -> str:
        presigned_url_result = subprocess.run(
            [
                "aws",
                "s3",
                "presign",
                f"s3://{self.bucket}/{key}",
                "--expires-in",
                "3600",
            ],
            capture_output=True,
        )
        logger.debug(f"Presigned URL result: {presigned_url_result}")
        presigned_url_result.check_returncode()
        return presigned_url_result.stdout.decode("utf-8").strip()


class S3MockClient(S3Client):
    def __init__(self, bucket: str = "webgwas"):
        self.temp_dir = pathlib.Path(os.getcwd()).joinpath("temp_data")
        self.temp_dir.mkdir(exist_ok=True)
        self.files = {}
        self.bucket = bucket

    def upload_file(self, local_path, key):
        start_path = pathlib.Path(local_path)
        end_path = self.temp_dir.joinpath(key)
        end_path.parent.mkdir(exist_ok=True)
        start_path.rename(self.temp_dir / key)
        bucket_keys = self.files.setdefault(self.bucket, {})
        bucket_keys[key] = self.temp_dir / key

    def get_presigned_url(self, key):
        return self.files[self.bucket][key].as_posix()


def get_s3_client(dry_run: bool, bucket: str):
    if dry_run:
        return S3MockClient(bucket=bucket)
    return S3ProdClient(bucket=bucket)
