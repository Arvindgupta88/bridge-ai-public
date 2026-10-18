import hashlib
from pathlib import Path


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate a SHA256 hash for a file.
    Args:
        file_path (Path): The path to the file for which to calculate the hash.
    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    sha = hashlib.sha256()

    with file_path.open("rb") as f:

        while True:

            block = f.read(8192)

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()