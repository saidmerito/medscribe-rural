# tests/test_pipeline.py
"""
Unit tests for MedScribe Rural pipeline components.
Run with: pytest tests/ -v
"""

import json
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock


# ─── OCR TESTS ───────────────────────────────────────────────────────────────

class TestImagePreprocessing:
    def test_preprocess_raises_on_missing_file(self):
        from ocr.extractor import preprocess_image
        with pytest.raises(ValueError, match="Could not load image"):
            preprocess_image("/nonexistent/path.jpg")


# ─── AI STRUCTURER TESTS ─────────────────────────────────────────────────────

class TestStructurer:

    MOCK_RECORD = {
        "patient_id": "DJ-2024-001",
        "date": "2024-03-15",
        "age": 7,
        "sex": "F",
        "chief_complaint": "fever and vomiting",
        "diagnosis": "Acute Watery Diarrhea",
        "icd10_code": "A09",
        "treatment": "ORS + Zinc",
        "outcome": "discharged",
        "confidence": 0.92,
        "flags": [],
        "raw_text": "Patient 7F fievre vomissements ORS"
    }

    def test_structure_records_adds_facility(self):
        from ai.structurer import structure_records

        mock_response = {
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "save_medical_records",
                        "arguments": {"records": [self.MOCK_RECORD.copy()]}
                    }
                }]
            }
        }

        with patch("ai.structurer._call_ollama", return_value=mock_response):
            records = structure_records("some ocr text", facility_name="Centre Ali Sabieh")

        assert len(records) == 1
        assert records[0]["facility"] == "Centre Ali Sabieh"
        assert "extracted_at" in records[0]

    def test_low_confidence_record_gets_flagged(self):
        from ai.structurer import structure_records

        low_conf_record = {**self.MOCK_RECORD, "confidence": 0.50, "flags": []}
        mock_response = {
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "save_medical_records",
                        "arguments": {"records": [low_conf_record]}
                    }
                }]
            }
        }

        with patch("ai.structurer._call_ollama", return_value=mock_response):
            records = structure_records("blurry text", facility_name="Test")

        assert "low_overall_confidence" in records[0]["flags"]

    def test_empty_ocr_text_returns_empty(self):
        from ai.structurer import structure_records
        with patch("ai.structurer._call_ollama") as mock:
            records = structure_records("   ")
        mock.assert_not_called()
        assert records == []


# ─── DATABASE TESTS ──────────────────────────────────────────────────────────

class TestDatabase:

    @pytest.fixture(autouse=True)
    def use_temp_db(self, tmp_path, monkeypatch):
        """Use a temporary database for each test."""
        monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
        import config
        config.DB_PATH = str(tmp_path / "test.db")
        import db.database as db_module
        db_module.DB_PATH = str(tmp_path / "test.db")
        db_module.init_db()

    def test_insert_and_retrieve_records(self):
        from db.database import insert_records, get_records

        records = [{
            "patient_id": "TEST-001",
            "facility": "Test Center",
            "date": "2024-03-15",
            "age": 25,
            "sex": "M",
            "chief_complaint": "fever",
            "diagnosis": "Malaria",
            "icd10_code": "B54",
            "treatment": "Artemether",
            "outcome": "discharged",
            "confidence": 0.95,
            "flags": [],
            "raw_text": "test",
            "extracted_at": "2024-03-15T10:00:00"
        }]

        n = insert_records(records)
        assert n == 1

        retrieved = get_records(days=30)
        assert len(retrieved) == 1
        assert retrieved[0]["patient_id"] == "TEST-001"

    def test_get_stats_returns_correct_count(self):
        from db.database import insert_records, get_stats

        records = [
            {
                "patient_id": f"P-{i}", "facility": "F", "date": "2024-03-15",
                "age": 20+i, "sex": "F", "chief_complaint": "cough",
                "diagnosis": "Pneumonia", "icd10_code": "J18",
                "treatment": "Amoxicillin", "outcome": "discharged",
                "confidence": 0.88, "flags": [], "raw_text": "",
                "extracted_at": "2024-03-15T10:00:00"
            } for i in range(5)
        ]
        insert_records(records)
        stats = get_stats()
        assert stats["total_records"] == 5

    def test_insert_empty_list(self):
        from db.database import insert_records
        n = insert_records([])
        assert n == 0


# ─── REPORT TESTS ────────────────────────────────────────────────────────────

class TestReports:

    def test_csv_export_returns_none_when_no_data(self):
        from reports.generator import generate_csv_export
        with patch("reports.generator.get_records", return_value=[]):
            result = generate_csv_export()
        assert result is None
