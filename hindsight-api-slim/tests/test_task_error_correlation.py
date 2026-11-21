"""Task failures retain an operation id even when the poller schedules a retry."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.worker.exceptions import RetryTaskAt


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_id", ["00000000-0000-0000-0000-000000000001", None])
@pytest.mark.parametrize("retry", [False, True])
async def test_task_failure_log_has_operation_id_and_traceback(operation_id: str | None, retry: bool, caplog) -> None:
    engine = object.__new__(MemoryEngine)
    engine._audit_logger = None
    failure = RuntimeError("ORA-00060: synthetic deadlock")
    engine._handle_graph_maintenance = AsyncMock(side_effect=failure)
    payload_marker = "synthetic-payload-must-not-appear-in-log"

    with patch(
        "hindsight_api.engine.memory_engine.get_config",
        return_value=SimpleNamespace(worker_max_retries=int(retry), worker_task_retry_backoff_seconds=60),
    ):
        with pytest.raises(RetryTaskAt if retry else RuntimeError, match="synthetic deadlock"):
            await engine._dispatch_task(
                {"type": "graph_maintenance", "unused_payload": payload_marker},
                "graph_maintenance",
                operation_id,
                None,
            )

    records = [record for record in caplog.records if "Task execution failed:" in record.getMessage()]
    assert len(records) == 1
    assert "graph_maintenance" in records[0].getMessage()
    assert f"operation_id: {operation_id}" in records[0].getMessage()
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is failure
    assert payload_marker not in caplog.text
