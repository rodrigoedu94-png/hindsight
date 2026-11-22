"""Real-LLM check for the consolidation prompt delta introduced in a827c6651.

Verifies the three prompt clarifications added to ``consolidation/prompts.py``:
1. Passing grumbles (e.g. ``"ugh, the printer jammed again"``) are classified as ``momentary``.
2. Another occurrence of an event at another time is judged ``unrelated`` to an earlier event
   observation and recorded as a separate event rather than overwriting the earlier one.
3. Unrelated entries may omit ``observation_text`` or emit ``""`` with valid schema output.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm
from hindsight_api.engine.response_models import MemoryFact
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


@pytest.mark.asyncio
async def test_real_model_respects_momentary_grumble_and_separate_event_occurrence():
    monday = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    tuesday = datetime.now(timezone.utc) - timedelta(days=1)
    obs_id = str(uuid.uuid4())
    pool = [
        MemoryFact(
            id=obs_id,
            text="Alice ran the database backup script on Monday.",
            fact_type="observation",
            mentioned_at=monday,
        )
    ]
    memories = [
        {
            "id": uuid.uuid4(),
            "text": "Ugh, the printer jammed again!",
            "tags": [],
            "mentioned_at": tuesday,
        },
        {
            "id": uuid.uuid4(),
            "text": "Alice ran the database backup script again on Tuesday.",
            "tags": [],
            "mentioned_at": tuesday,
        },
    ]

    result = await _consolidate_batch_with_llm(
        llm_config=LLMConfig.from_env(),
        memories=memories,
        union_observations=pool,
        union_source_facts={},
        config=_get_raw_config(),
        closest={str(m["id"]): [obs_id] for m in memories},
    )

    assert result.failed is False, "the consolidation batch call failed outright"
    writes = "\n".join(
        [f"- CREATE: {c.text}" for c in result.creates]
        + [f"- UPDATE {u.observation_id}: {u.text}" for u in result.updates]
        + [f"- DELETE {d.observation_id}: {d.reason}" for d in result.deletes]
    )
    await assert_meets_criteria(
        response=writes or "(no actions)",
        criteria=(
            "1) The passing grumble about the printer jamming is NOT recorded as a lasting observation. "
            "2) Alice running the database backup script on Tuesday is recorded as a separate event "
            "(a new observation), and does NOT overwrite or delete the Monday observation."
        ),
        context=(
            "Existing observation: 'Alice ran the database backup script on Monday.'\n"
            "New facts: (a) 'Ugh, the printer jammed again!' (b) 'Alice ran the database backup script again on Tuesday.'"
        ),
    )
