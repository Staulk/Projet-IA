"""Couche LLM optionnelle, via Ollama (100% local).

Rôle du LLM : NE PAS deviner le film avec ses connaissances internes (risque
d'hallucination, hors du dataset). Il sert uniquement à choisir/expliquer parmi
les candidats déjà trouvés par la recherche hybride (fuzzy + sémantique), et à
rédiger une réponse naturelle.

Installer Ollama : https://ollama.com/download
Puis récupérer un modèle léger, ex : `ollama pull llama3.2`
"""

from __future__ import annotations

from typing import List

import requests

from .search import SearchResult

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


def is_ollama_available(timeout: float = 1.0) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _build_prompt(question: str, candidates: List[SearchResult]) -> str:
    lines = [
        "Tu es un assistant expert en citations de films. Tu dois identifier de quel "
        "film provient une réplique, en te basant UNIQUEMENT sur les candidats listés "
        "ci-dessous, qui viennent d'une base de données de citations. N'utilise AUCUNE "
        "autre connaissance et n'invente rien : si aucun candidat ne correspond "
        "vraiment, dis-le clairement.",
        "",
        f'Question de l\'utilisateur : "{question}"',
        "",
        "Candidats trouvés dans la base (du plus probable au moins probable) :",
    ]
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f'{i}. "{c.quote}" — {c.movie} ({c.year}) '
            f"[score de correspondance: {c.score:.2f}, méthode: {c.method}]"
        )
    lines += [
        "",
        "Réponds en français, en 2-3 phrases maximum : indique quel candidat "
        "correspond le mieux (titre + année) et pourquoi, ou indique qu'aucun ne "
        "correspond de façon fiable.",
    ]
    return "\n".join(lines)


def ask_llm(question: str, candidates: List[SearchResult], model: str = DEFAULT_MODEL) -> str:
    prompt = _build_prompt(question, candidates)
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()
