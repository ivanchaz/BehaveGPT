import gzip
import os


def is_credential_not_available() -> bool:
    credential_path = os.getenv("APPLICATION_DEFAULT_CREDENTIALS")
    return credential_path is None or credential_path == ""


def decompress(content: bytes) -> str:
    decompressed = gzip.decompress(content)
    return decompressed.decode("utf-8")
