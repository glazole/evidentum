from __future__ import annotations

import os
import sys
import types
import unittest

from sqlalchemy import JSON

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

if "pgvector.sqlalchemy" not in sys.modules:
    pgvector_module = types.ModuleType("pgvector")
    pgvector_sqlalchemy_module = types.ModuleType("pgvector.sqlalchemy")
    pgvector_sqlalchemy_module.Vector = lambda dim=None: JSON()  # type: ignore[attr-defined]
    sys.modules["pgvector"] = pgvector_module
    sys.modules["pgvector.sqlalchemy"] = pgvector_sqlalchemy_module

from app.services import retriever
from app.services.retriever import RetrievalHit


def make_hit(
    chunk_id: int,
    *,
    distance: float = 0.2,
    fts_rank: float | None = None,
    retrieval_source: list[str] | None = None,
    section_title: str = "Лечение",
    chunk_text: str = "антикоагулянтная терапия при фибрилляции предсердий",
) -> RetrievalHit:
    semantic_score = None if retrieval_source == ["fts"] else retriever.score_from_distance(distance)
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=1,
        chunk_index=chunk_id,
        source_id="ru_af",
        source_name="Клинические рекомендации",
        title="Фибрилляция предсердий",
        region="RU",
        year=2024,
        url=None,
        file_path=None,
        section_title=section_title,
        chunk_text=chunk_text,
        char_count=len(chunk_text),
        score=semantic_score or retriever.score_from_fts_rank(fts_rank),
        distance=distance,
        retrieval_source=retrieval_source or ["raw_query"],
        semantic_score=semantic_score,
        fts_rank=fts_rank,
    )


class HybridRetrieverTests(unittest.TestCase):
    def test_decompose_query_keeps_original_and_subquestions(self) -> None:
        variants = retriever.decompose_query(
            "Как лечить ФП у беременных и какие антикоагулянты противопоказаны?"
        )

        self.assertEqual(variants[0], "Как лечить ФП у беременных и какие антикоагулянты противопоказаны?")
        self.assertIn("Как лечить ФП у беременных", variants)
        self.assertIn("какие антикоагулянты противопоказаны", variants)

    def test_merge_combines_vector_and_fts_signals(self) -> None:
        vector_hit = make_hit(10, distance=0.25, retrieval_source=["raw_query"])
        fts_hit = make_hit(10, distance=1.0, fts_rank=0.8, retrieval_source=["fts"])

        merged = retriever.merge_and_rerank_hits(
            [vector_hit],
            [fts_hit],
            top_k=5,
            query="антикоагулянтная терапия при фибрилляции предсердий",
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(set(merged[0].retrieval_source or []), {"raw_query", "fts"})
        self.assertAlmostEqual(merged[0].semantic_score or 0.0, retriever.score_from_distance(0.25))
        self.assertEqual(merged[0].fts_rank, 0.8)
        self.assertIsNotNone(merged[0].citation_confidence)

    def test_noise_section_is_penalized(self) -> None:
        treatment = make_hit(1, section_title="Лечение")
        references = make_hit(2, section_title="Список литературы")

        self.assertGreater(
            retriever.rerank_score(treatment, ["лечение"], "лечение"),
            retriever.rerank_score(references, ["лечение"], "лечение"),
        )


if __name__ == "__main__":
    unittest.main()
