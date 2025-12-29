from aws_durable_execution_sdk_python import (
    durable_execution,
    DurableContext,
    durable_step,
    StepContext,
)
from aws_durable_execution_sdk_python.config import (
    CompletionConfig,
    ItemBatcher,
    MapConfig,
)
import boto3
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import os

s3 = boto3.client("s3")


ARTIFACT_BUCKET = os.environ.get("BUCKET_NAME", "my-artifact-bucket")
PDF_TO_MARKDOWN_ARN = os.environ.get("PDF_TO_MARKDOWN_ARN", "")
DOC_TO_MARKDOWN_ARN = os.environ.get("PDF_TO_MARKDOWN_ARN", "")


@durable_execution
def lambda_handler(event: dict, context: DurableContext):
    key = event["key"]
    id = f"id-{key}"

    file_type = context.step(detect_file_type(
        key, id), name="detect_file_type")

    file_meta = {"type": file_type, "key": key, "id": id}

    if file_type == "pdf":
        markdown_key = context.invoke(
            PDF_TO_MARKDOWN_ARN, file_meta, name="pdf_to_markdown"
        )
    elif file_type == "doc":
        markdown_key = context.invoke(
            DOC_TO_MARKDOWN_ARN, file_meta, name="doc_to_markdown"
        )
    else:
        return {}

    chunks = context.step(split_text(markdown_key, id), name="split_text")

    config = MapConfig(
        max_concurrency=5,
        item_batcher=ItemBatcher(max_items_per_batch=5),
        completion_config=CompletionConfig.all_successful(),
    )

    results = context.map(
        inputs=chunks,
        func=embed_and_store,
        name="embed_and_store",
        config=config,
    )

    context.logger.info(results)

    return {
        "message": "Execution completed successfully",
    }


@durable_step
def detect_file_type(step_context: StepContext, key: str, id: str):
    key_lower = key.lower()

    if key_lower.endswith(".pdf"):
        return "pdf"
    return {}


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
    batch: dict,
    index: int,
    items: list[dict],
):
    step_context.logger.info(
        f"Processing batch {index} with {len(items)} items.")
    step_context.logger.info(f"batch: {batch}")

    if not batch:
        return {"message": "No data to process"}
    obj = s3.get_object(Bucket=ARTIFACT_BUCKET, Key=batch["key"])
    chunk = obj["Body"].read().decode("utf-8")

    print(chunk)

    return {"message": "Processed successfully"}
