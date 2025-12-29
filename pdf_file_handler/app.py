import os
from io import BytesIO

import boto3
from markitdown import MarkItDown

ARTIFACT_BUCKET = os.environ.get("BUCKET_NAME", "default-bucket-name")
s3_client = boto3.client("s3")

md = MarkItDown()


def lambda_handler(event, context):
    file_key = event["key"]
    id = event["id"]

    obj = s3_client.get_object(Bucket=ARTIFACT_BUCKET, Key=file_key)

    file_bytes = obj["Body"].read()

    stream = BytesIO(file_bytes)

    result = md.convert_stream(stream)

    out_key = f"{id}/markdown/markdown.md"

    s3_client.put_object(
        Bucket=ARTIFACT_BUCKET,
        Key=out_key,
        Body=result.text_content,
    )

    return out_key
