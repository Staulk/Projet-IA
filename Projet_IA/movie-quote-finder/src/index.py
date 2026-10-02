"""Index sémantique (embeddings) construit une seule fois et mis en cache sur disque.

Utilise un petit modèle sentence-transformers (léger, tourne bien sur CPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .data import QuoteEntry, load_quotes

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# NB: le dataset est en anglais, mais ce modèle multilingue permet aussi de poser
# la question en français (ou dans d'autres langues) tout en matchant correctement.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
META_PATH = CACHE_DIR / "quotes_meta.json"

_model = None  # chargé à la demande (paresseux), pour ne pas payer le coût si non utilisé


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _save_meta(quotes: List[QuoteEntry]) -> None:
    data = [
        {"quote": q.quote, "movie": q.movie, "type": q.type, "year": q.year, "normalized": q.normalized}
        for q in quotes
    ]
    META_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_meta() -> List[QuoteEntry]:
    data = json.loads(META_PATH.read_text(encoding="utf-8"))
    return [QuoteEntry(**row) for row in data]


def build_index(force: bool = False) -> tuple[List[QuoteEntry], np.ndarray]:
    """Construit (ou recharge depuis le cache) la liste des citations + leurs embeddings."""
    if not force and EMBEDDINGS_PATH.exists() and META_PATH.exists():
        quotes = _load_meta()
        embeddings = np.load(EMBEDDINGS_PATH)
        if len(quotes) == embeddings.shape[0]:
            return quotes, embeddings
        # cache incohérent (dataset changé) -> on reconstruit

    quotes = load_quotes()
    model = get_model()
    texts = [q.quote for q in quotes]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # -> cosine similarity = simple produit scalaire
        convert_to_numpy=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    _save_meta(quotes)
    return quotes, embeddings


if __name__ == "__main__":
    quotes, embeddings = build_index(force=True)
    print(f"Index construit : {len(quotes)} citations, embeddings shape={embeddings.shape}")
    print(f"Sauvegardé dans {EMBEDDINGS_PATH} et {META_PATH}")
