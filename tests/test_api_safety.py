from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import JSON

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

if "pgvector.sqlalchemy" not in sys.modules:
    pgvector_module = types.ModuleType("pgvector")
    pgvector_sqlalchemy_module = types.ModuleType("pgvector.sqlalchemy")
    pgvector_sqlalchemy_module.Vector = lambda dim=None: JSON()  # type: ignore[attr-defined]
    sys.modules["pgvector"] = pgvector_module
    sys.modules["pgvector.sqlalchemy"] = pgvector_sqlalchemy_module

if "python_multipart" not in sys.modules:
    python_multipart_module = types.ModuleType("python_multipart")
    python_multipart_module.__version__ = "0.0.20"
    sys.modules["python_multipart"] = python_multipart_module
if "multipart.multipart" not in sys.modules:
    multipart_module = types.ModuleType("multipart")
    multipart_module.__version__ = "0.0.20"
    multipart_inner_module = types.ModuleType("multipart.multipart")
    multipart_inner_module.parse_options_header = lambda value: (value, {})  # type: ignore[attr-defined]
    sys.modules["multipart"] = multipart_module
    sys.modules["multipart.multipart"] = multipart_inner_module

from app.api import routes


class UploadSafetyTests(unittest.TestCase):
    def test_safe_upload_destination_stays_inside_raw_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(routes, "DATA_RAW_DIR", Path(tmp)):
                dest = routes._safe_upload_destination("../../secret.pdf")

        self.assertEqual(dest.suffix, ".pdf")
        self.assertNotIn("..", dest.parts)
        self.assertEqual(dest.name, "secret.pdf")

    def test_safe_upload_destination_rejects_unsupported_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(routes, "DATA_RAW_DIR", Path(tmp)):
                with self.assertRaises(HTTPException) as ctx:
                    routes._safe_upload_destination("payload.exe")

        self.assertEqual(ctx.exception.status_code, 400)

    def test_save_upload_file_enforces_size_limit(self) -> None:
        class FakeUpload:
            file = io.BytesIO(b"abcde")

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "upload.pdf"
            with patch.object(routes, "UPLOAD_MAX_BYTES", 4):
                with self.assertRaises(HTTPException) as ctx:
                    routes._save_upload_file(FakeUpload(), dest)  # type: ignore[arg-type]

            self.assertEqual(ctx.exception.status_code, 413)
            self.assertFalse(dest.exists())

    def test_save_upload_file_rejects_existing_file(self) -> None:
        class FakeUpload:
            file = io.BytesIO(b"abc")

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "upload.pdf"
            dest.write_bytes(b"old")

            with self.assertRaises(HTTPException) as ctx:
                routes._save_upload_file(FakeUpload(), dest)  # type: ignore[arg-type]

            self.assertEqual(ctx.exception.status_code, 409)
            self.assertEqual(dest.read_bytes(), b"old")


class AdminAndAnswerTests(unittest.TestCase):
    def test_admin_token_required(self) -> None:
        with patch.object(routes, "ADMIN_API_TOKEN", "secret"):
            with self.assertRaises(HTTPException) as ctx:
                routes.require_admin_token("wrong")

        self.assertEqual(ctx.exception.status_code, 401)

    def test_answer_route_passes_document_id(self) -> None:
        captured: dict[str, int | None] = {}

        def fake_answer_question(*args, **kwargs):
            captured["document_id"] = kwargs.get("document_id")
            return object()

        with (
            patch.object(routes, "answer_question", side_effect=fake_answer_question),
            patch.object(routes, "format_for_ui", return_value={"answer": "ok", "sources": []}),
            patch("app.services.evaluator._build_context_text", return_value=""),
            patch("app.services.evaluator.save_query_log", return_value=1),
            patch("app.services.evaluator.run_judge", return_value=None),
        ):
            routes.answer_route(
                routes.AnswerRequest(question="q", document_id=42),
                BackgroundTasks(),
            )

        self.assertEqual(captured["document_id"], 42)


if __name__ == "__main__":
    unittest.main()
