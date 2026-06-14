import struct

import numpy as np

from client import client


def _embed_text_to_floats(text: str) -> list[float]:
    """
    Returns a python list of float32 values.
    The model is asked to normalize the vector (unit length) - this makes
    cosine similarity equivalent to a simple dot-product, which many SQLite
    vector extensions expect.
    """

    # get the raw embedding from OpenAI (list of Python floats)
    raw_vec: list[float] = embed(text)

    #  convert to a numpy array so we can normalize and cast safely
    vec = np.asarray(raw_vec, dtype=np.float32)

    # L2-normalise (avoid division-by-zero for the zero-vector)
    norm = np.linalg.norm(vec)
    if norm > 0.0:
        vec = vec / norm

    # return a plain python list (still float32 values)
    return vec.tolist() # -> List[float]


def serialize_f32(vector: list[float]) -> bytes:
    """
    Pack a Python list of floats into the exact binary layout
    that SQLite vector extensions understand: little-endian, 4-byte IEEE-754.
    Equivalent to `sqlite_vec.serialize_f32` or numpy's .tobytes().
    """
    # struct.pack format: "<" = little‑endian, repeat "f" for each element
    fmt = f"<{len(vector)}f"
    return struct.pack(fmt, *vector)


def embed(text: str) -> list[float]:
    """Return the OpenAI embedding for a single string."""
    response = client.embeddings.create(
        model="text-embedding-3-small",   # change model if you like
        input=[text]                      # API expects a list, even for one item
    )
    return response.data[0].embedding   # already a plain Python list of floats


def cosine_similarity_np(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity using NumPy.

    Parameters
    ----------
    a, b : list[float]
        Two equal‑length embedding vectors.

    Returns
    -------
    float
        Cosine similarity in the range [-1, 1].
    """
    # Convert to NumPy arrays (float64 by default)
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)

    # Dot product
    dot = np.dot(va, vb)

    # L2 norms
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    # Guard against zero‑length vectors (should not happen with OpenAI embeddings)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
