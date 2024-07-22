import pathlib
from abc import ABC, abstractmethod

import boto3


class S3Client(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def upload_file(self, local_path, bucket, key):
        pass


class S3ProdClient(S3Client):
    def __init__(self):
        super().__init__()
        self.s3_client = boto3.client("s3")

    def upload_file(self, local_path, bucket, key):
        self.s3_client.upload_file(local_path, bucket, key)


class S3MockClient(S3Client):
    def __init__(self):
        super().__init__()
        self.data_dir = pathlib.Path(
            "/Users/zietzm/Documents/projects/webgwas-frontend/webgwas-fastapi/"
            "test_data/webgwas-results"
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}

    def upload_file(self, local_path, bucket, key):
        pathlib.Path(local_path).rename(self.data_dir / key)
        bucket_keys = self.files.setdefault(bucket, {})
        bucket_keys[key] = self.data_dir / key
