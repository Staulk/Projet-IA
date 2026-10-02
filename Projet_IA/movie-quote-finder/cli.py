"""CLI interactive : pose une question sur une réplique de film, obtiens le(s)
film(s) candidat(s), avec une couche LLM locale optionnelle (Ollama) pour une
réponse en langage naturel.

    python cli.py
"""

from src.index import build_index
from src.llm import ask_llm, is_ollama_available
from src.search import hybrid_search


def main() -> None:
    print("Chargement de l'index (première fois : téléchargement du dataset + calcul des "
          "embeddings, peut prendre une minute)...")
    quotes, embeddings = build_index()
    print(f"Index prêt : {len(quotes)} répliques indexées.\n")

    llm_available = is_ollama_available()
    if llm_available:
        print("Ollama détecté -> les réponses seront reformulées par un LLM local.\n")
    else:
        print("Ollama non détecté -> affichage brut des 5 meilleurs candidats "
              "(installe/lance Ollama plus tard pour une réponse en langage naturel, "
              "voir src/llm.py).\n")

    print("Tape ta question (ou 'quit' pour sortir).\n")

    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        # On récupère un filet plus large (top 10) pour la couche LLM : sur une
        # question très reformulée, la bonne réponse n'est pas toujours dans le
        # tout premier rang de la recherche fuzzy/sémantique, mais un LLM qui
        # raisonne sur 10 candidats a de bien meilleures chances de la repérer
        # que le pur score de similarité.
        wide_results = hybrid_search(question, quotes, embeddings, top_k=10)
        # Sans LLM pour trancher, on affiche 5 candidats (au lieu de 3) : ça laisse
        # plus de chances à l'œil humain de repérer le bon film sur une question floue.
        results = wide_results[:5]

        print("\nCandidats trouvés (top 5) :")
        for r in results:
            print(f"  - {r.movie} ({r.year}) — confiance {r.confidence} "
                  f"[score={r.score:.2f}, méthode={r.method}]")
            print(f"    citation : \"{r.quote}\"")

        if llm_available:
            print("\nRéponse : ", end="", flush=True)
            try:
                answer = ask_llm(question, wide_results)
                print(answer)
            except Exception as exc:  # noqa: BLE001
                print(f"(erreur LLM : {exc})")
        print()


if __name__ == "__main__":
    main()
