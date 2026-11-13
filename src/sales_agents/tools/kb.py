"""Product knowledge: the fictional seller's docs and a small keyword search.

The corpus is ten short passages, so a keyword search with inverse document
frequency is enough. A vector store can replace `search_product_docs` later
without changing its signature.
"""

import math
import re
from functools import lru_cache
from pathlib import Path

DOCS_PATH = Path(__file__).parents[3] / "datasets" / "product_docs.md"

STOPWORDS = frozenset(
    "a an the and or of to in on for is are it its do does how what our we you your with from at by as be "
    "this that much many can will i my me us".split()
)


def stem(word: str) -> str:
    """Crude stemmer: drop a plural or -ed/-ing ending, then a trailing e."""
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def words(text: str) -> list[str]:
    """Lowercase word stems without stopwords."""
    found = re.findall(r"[a-z0-9]+", text.lower().replace("’", "'"))
    return [stem(w) for w in found if w not in STOPWORDS]


@lru_cache
def load_docs(path: Path = DOCS_PATH) -> list[str]:
    return [p.strip() for p in path.read_text().split("\n\n") if p.strip()]


def search_product_docs(query: str, k: int = 3) -> list[str]:
    """The k passages that best match the query. Returns [] when nothing matches."""
    docs = load_docs()
    passage_words = [set(words(d)) for d in docs]
    scores = []
    for doc, vocab in zip(docs, passage_words):
        score = 0.0
        for term in set(words(query)):
            if term in vocab:
                df = sum(term in other for other in passage_words)
                score += math.log(1 + len(docs) / df)
        scores.append((score, doc))
    ranked = sorted((s for s in scores if s[0] > 0), key=lambda s: s[0], reverse=True)
    return [doc for _, doc in ranked[:k]]
