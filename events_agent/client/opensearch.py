import os

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

_client = None


def get_opensearch_client() -> OpenSearch:
    global _client

    """Get or create an OpenSearch client with AWS authentication."""
    if _client:
        return _client

    # Use boto3's default credential chain (mimics TypeScript's defaultProvider)
    session = boto3.Session()
    credentials = session.get_credentials()
    if not credentials:
        raise ValueError("No AWS credentials found. Ensure AWS CLI is configured or use an IAM role.")

    aws_auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        "us-east-1",
        "es",
        session_token=credentials.token if credentials.token else None,
    )

    _client = OpenSearch(
        hosts=[{"host": os.getenv("OPENSEARCH_URL").replace("https://", ""), "port": 443}],
        http_auth=aws_auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    if not _client.ping():
        raise ConnectionError("Failed to connect to OpenSearch at " + os.getenv("OPENSEARCH_URL"))

    return _client
