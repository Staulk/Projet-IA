"""Chargement et normalisation du dataset de répliques de films.

Source : https://huggingface.co/datasets/ygorgeurts/movie-quotes
Colonnes : quote (str), movie (str), type (str), year (int)
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import List

DATASET_ID = "ygorgeurts/movie-quotes"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class QuoteEntry:
    quote: str
    movie: str
    type: str
    year: int
    normalized: str  # version nettoyée, utilisée pour le fuzzy matching


def normalize_text(text: str) -> str:
    """Nettoie un texte pour la comparaison fuzzy : minuscules, sans ponctuation,
    espaces multiples réduits. Garde les lettres/chiffres/espaces uniquement.
    """
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_quotes() -> List[QuoteEntry]:
    """Télécharge (et met en cache localement via la lib `datasets`) le dataset,
    puis retourne la liste des répliques normalisées.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")

    entries: List[QuoteEntry] = []
    for row in ds:
        quote = (row.get("quote") or "").strip()
        movie = (row.get("movie") or "").strip()
        if not quote or not movie:
            continue
        entries.append(
            QuoteEntry(
                quote=quote,
                movie=movie,
                type=(row.get("type") or "").strip(),
                year=int(row["year"]) if row.get("year") is not None else 0,
                normalized=normalize_text(quote),
            )
        )
    return entries


if __name__ == "__main__":
    quotes = load_quotes()
    print(f"{len(quotes)} répliques chargées.")
    for q in quotes[:5]:
        print(f"- [{q.year}] {q.movie}: {q.quote!r}")
