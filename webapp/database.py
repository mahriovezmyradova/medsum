"""
Supabase session storage for the MedASR web application.

Table schema (run once in the Supabase SQL editor):
----------------------------------------------------
create table public.sessions (
    id          uuid primary key default gen_random_uuid(),
    created_at  timestamptz not null default now(),
    doctor_id   text,
    duration_s  float,
    asr_model   text,
    ai_enabled  boolean,
    ai_model    text,
    summarizer  text,
    raw_transcript      text,
    enhanced_transcript text,
    summary_raw         text,
    summary_enhanced    text,
    rouge1_raw          float,
    rouge1_enhanced     float,
    medical_pres_raw    float,
    medical_pres_enh    float,
    ai_improvement_score float
);

-- Enable Row-Level Security if needed:
-- alter table public.sessions enable row level security;
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Optional

logger = logging.getLogger(__name__)


class SessionDB:
    """
    Thin wrapper around the Supabase Python client.

    Falls back to a no-op when credentials are not configured so the app
    still works without a database connection.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ):
        self._client = None
        url = url or os.getenv("SUPABASE_URL")
        key = key or os.getenv("SUPABASE_KEY")

        if url and key:
            try:
                from supabase import create_client
                self._client = create_client(url, key)
                logger.info("Supabase client initialised (%s)", url)
            except ImportError:
                logger.warning("supabase-py not installed – DB storage disabled. Run: pip install supabase")
            except Exception as exc:
                logger.warning("Supabase connection failed: %s", exc)
        else:
            logger.info("SUPABASE_URL / SUPABASE_KEY not set – running without DB persistence")

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── write ─────────────────────────────────────────────────────────────────

    def save_session(
        self,
        doctor_id: str,
        pipeline_result,          # PipelineResult from src/pipeline/pipeline.py
        audio_duration_s: float,
    ) -> Optional[str]:
        """
        Persist one consultation session.  Returns the new row UUID or None.
        """
        if not self.available:
            return None

        row = {
            "doctor_id": doctor_id,
            "duration_s": round(audio_duration_s, 2),
            "asr_model": pipeline_result.asr_model,
            "ai_enabled": pipeline_result.ai_enabled,
            "ai_model": pipeline_result.ai_model,
            "summarizer": pipeline_result.summarizer,
            "raw_transcript": pipeline_result.raw_transcription,
            "enhanced_transcript": pipeline_result.enhanced_transcription,
            "summary_raw": pipeline_result.summary_raw,
            "summary_enhanced": pipeline_result.summary_enhanced,
            "rouge1_raw": pipeline_result.metrics_raw.get("rouge1"),
            "rouge1_enhanced": pipeline_result.metrics_enhanced.get("rouge1"),
            "medical_pres_raw": pipeline_result.metrics_raw.get("medical_preservation"),
            "medical_pres_enh": pipeline_result.metrics_enhanced.get("medical_preservation"),
            "ai_improvement_score": round(pipeline_result.ai_improvement_score, 4),
        }

        try:
            response = self._client.table("sessions").insert(row).execute()
            session_id = response.data[0]["id"] if response.data else None
            logger.info("Session saved: %s", session_id)
            return session_id
        except Exception as exc:
            logger.error("Failed to save session: %s", exc)
            return None

    # ── read ──────────────────────────────────────────────────────────────────

    def get_recent_sessions(self, doctor_id: str, limit: int = 20) -> list[dict]:
        """Return the most recent sessions for a given doctor ID."""
        if not self.available:
            return []
        try:
            resp = (
                self._client.table("sessions")
                .select("*")
                .eq("doctor_id", doctor_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as exc:
            logger.error("Failed to fetch sessions: %s", exc)
            return []

    def get_all_sessions(self, limit: int = 100) -> list[dict]:
        """Return the most recent sessions across all doctors (admin view)."""
        if not self.available:
            return []
        try:
            resp = (
                self._client.table("sessions")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return resp.data or []
        except Exception as exc:
            logger.error("Failed to fetch all sessions: %s", exc)
            return []
