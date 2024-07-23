import pathlib
import tempfile
from abc import ABC, abstractmethod


class S3Client(ABC):
    @abstractmethod
    def upload_file(self, local_path, key):
        pass


class S3ProdClient(S3Client):
    def __init__(self, s3_client, bucket: str = "webgwas"):
        self.s3_client = s3_client
        self.bucket = bucket

    def upload_file(self, local_path, key):
        self.s3_client.upload_file(local_path, self.bucket, key)


class S3MockClient(S3Client):
    def __init__(self, bucket: str = "webgwas"):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = pathlib.Path(self.temp_dir.name)
        self.files = {}
        self.bucket = bucket

    def upload_file(self, local_path, key):
        pathlib.Path(local_path).rename(self.data_dir / key)
        bucket_keys = self.files.setdefault(self.bucket, {})
        bucket_keys[key] = self.data_dir / key
