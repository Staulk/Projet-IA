"""À exécuter une seule fois (ou après mise à jour du dataset) pour construire
et mettre en cache l'index sémantique.

    python build_index.py
"""

from src.index import build_index

if __name__ == "__main__":
    quotes, embeddings = build_index(force=True)
    print(f"Index construit : {len(quotes)} citations, embeddings shape={embeddings.shape}")
