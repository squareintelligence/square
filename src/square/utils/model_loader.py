from __future__ import annotations

from io import BytesIO
import os
from typing import Any

import boto3
import joblib
from botocore.client import BaseClient


def load_joblib_model_from_s3(model_path: str, s3_client: BaseClient | None = None) -> Any:
    bucket = "square-intelligence-metric-models"
    prefix = "models"

    key = f"{prefix}/{model_path.lstrip('/')}" if prefix else model_path.lstrip("/")
    if not key:
        raise ValueError("Model path cannot be empty")

    region = os.getenv("AWS_REGION")
    client = s3_client or boto3.client("s3", region_name=region)
    buffer = BytesIO()
    client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)

    return joblib.load(buffer)
