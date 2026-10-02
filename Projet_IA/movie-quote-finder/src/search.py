"""Recherche hybride : fuzzy matching (fautes de frappe) + embeddings sémantiques
(questions floues / paraphrasées).

Stratégie :
- on calcule un score fuzzy (0-1) ET un score sémantique (0-1) pour chaque citation
- ces deux scores ont un "plancher" différent et non nul : deux phrases sans aucun
  rapport obtiennent déjà ~0.55 de similarité cosinus en moyenne (anisotropie des
  embeddings), et un score fuzzy non négligeable dès qu'elles partagent des mots
  courants. Un seuil absolu est donc trompeur.
- on normalise chaque score en z-score par rapport à SA PROPRE distribution sur
  cette question précise (score candidat - moyenne des 732 scores) / écart-type.
  Ça répond à la question "à quel point ce candidat sort du lot", peu importe où
  se situe le plancher.
- le score final retenu est le max des deux z-scores -> une citation très proche
  mot-à-mot (même mal orthographiée) remonte grâce au fuzzy, une citation
  reformulée / décrite de façon floue remonte grâce au sémantique
- la méthode qui a "gagné" pour chaque résultat sert à afficher un niveau de confiance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rapidfuzz import fuzz, process

from .data import QuoteEntry, normalize_text
from .index import get_model


@dataclass
class SearchResult:
    quote: str
    movie: str
    year: int
    type: str
    score: float  # score brut (0-1) de la méthode gagnante, pour affichage
    z_score: float  # écarts-type au-dessus de la moyenne des 732 scores, pour cette question
    method: str  # "fuzzy" | "semantic"
    confidence: str  # "élevée" | "moyenne" | "faible"


def _confidence_label(z: float) -> str:
    # Calibré empiriquement : un vrai bon match sort typiquement à 6+ écarts-type
    # au-dessus du bruit ambiant ; un match plausible mais incertain vers 3-6 ;
    # en dessous, le candidat ne se distingue pas vraiment du reste du dataset.
    if z >= 6:
        return "élevée"
    if z >= 3:
        return "moyenne"
    return "faible"


def _zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def _fuzzy_scores(query_norm: str, quotes: List[QuoteEntry]) -> np.ndarray:
    choices = [q.normalized for q in quotes]
    # token_sort_ratio résiste bien aux mots dans le désordre / fautes légères
    matches = process.cdist([query_norm], choices, scorer=fuzz.token_sort_ratio)[0]
    return np.asarray(matches, dtype=np.float32) / 100.0


def _semantic_scores(query: str, embeddings: np.ndarray) -> np.ndarray:
    model = get_model()
    query_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    # embeddings et query_emb sont normalisés -> produit scalaire = cosine similarity
    sims = embeddings @ query_emb[0]
    # ramène de [-1, 1] à [0, 1] pour rester cohérent avec les scores fuzzy
    return (sims + 1.0) / 2.0


def hybrid_search(
    query: str,
    quotes: List[QuoteEntry],
    embeddings: np.ndarray,
    top_k: int = 3,
) -> List[SearchResult]:
    query_norm = normalize_text(query)

    fuzzy_scores = _fuzzy_scores(query_norm, quotes)
    semantic_scores = _semantic_scores(query, embeddings)

    fuzzy_z = _zscore(fuzzy_scores)
    semantic_z = _zscore(semantic_scores)

    combined_z = np.maximum(fuzzy_z, semantic_z)
    methods = np.where(fuzzy_z >= semantic_z, "fuzzy", "semantic")

    top_indices = np.argsort(-combined_z)[:top_k]

    results: List[SearchResult] = []
    for idx in top_indices:
        q = quotes[idx]
        method = str(methods[idx])
        raw_score = float(fuzzy_scores[idx] if method == "fuzzy" else semantic_scores[idx])
        z = float(combined_z[idx])
        results.append(
            SearchResult(
                quote=q.quote,
                movie=q.movie,
                year=q.year,
                type=q.type,
                score=raw_score,
                z_score=z,
                method=method,
                confidence=_confidence_label(z),
            )
        )
    return results
