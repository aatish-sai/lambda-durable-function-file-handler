from collections.abc import Sequence
from aws_durable_execution_sdk_python import (
    durable_execution,
    DurableContext,
    durable_step,
    StepContext,
)
from aws_durable_execution_sdk_python.config import (
    BatchedInput,
    CompletionConfig,
    ItemBatcher,
    MapConfig,
)
from markitdown import MarkItDown
import boto3
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

s3 = boto3.client("s3")


ARTIFACT_BUCKET = "your-artifact-bucket-name"


@durable_execution
def lambda_handler(event: dict, context: DurableContext):
    key = ""
    id = ""

    file_meta = context.step(detect_file_type(
        key, id), name="detect_file_type")

    markdown_key, markdown_id = context.step(
        to_markdown(file_meta), name="to_markdown")

    chunks = context.step(split_text(
        markdown_key, markdown_id), name="split_text")

    config = MapConfig(
        max_concurrency=5,
        item_batcher=ItemBatcher(max_items_per_batch=1),
        completion_config=CompletionConfig.all_successful(),
    )

    results = context.map(
        inputs=chunks,
        func=embed_and_store,
        name="embed_and_store",
        config=config,
    )

    return results


@durable_step
def detect_file_type(step_context: StepContext, key: str, id: str):
    key_lower = key.lower()

    if key_lower.endswith(".pdf"):
        return {"key": key, "id": id, "type": "pdf"}
    return {}


@durable_step
def to_markdown(step_context: StepContext, file_meta: dict):
    key = file_meta["key"]
    id = file_meta["id"]

    obj = s3.get_object(Bucket=ARTIFACT_BUCKET, Key=key)
    file_bytes = obj["Body"].read()

    md = MarkItDown()

    result = md.convert_stream(file_bytes)

    out_key = f"{id}/markdown/markdown.md"

    s3.put_object(
        Bucket=ARTIFACT_BUCKET,
        Key=out_key,
        Body=result.text_content,
    )

    return out_key, id


@durable_step
def split_text(step_context: StepContext, markdown_key: str, id: str):
    obj = s3.get_object(Bucket=ARTIFACT_BUCKET, Key=markdown_key)
    markdown = obj["Body"].read().decode("utf-8")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    )
    docs = header_splitter.split_text(markdown)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunk_keys = []

    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)

        for idx, c in enumerate(chunks):
            key = f"{id}/chunks/{idx}.txt"

            s3.put_object(
                Bucket=ARTIFACT_BUCKET,
                Key=key,
                Body=c,
                ContentType="text/plain",
            )

            chunk_keys.append({"key": key, "id": id})
    return chunk_keys


def embed_and_store(
    step_context: DurableContext,
    batch: BatchedInput[None, dict],
    index: int,
    items: Sequence[dict],
):
    result = []
    for item in batch.items:
        obj = s3.get_object(Bucket=ARTIFACT_BUCKET, Key=item["key"])
        chunk = obj["Body"].read().decode("utf-8")

        print(chunk)

        result.append({})

    return result
