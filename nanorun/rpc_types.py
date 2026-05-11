"""RPC protocol types for nanorun websocket communication.

This module defines the message types, method names, event names, and error codes
shared between the RPC server (remote) and client (local). Both sides import from
here to ensure protocol agreement.

Transport: JSON over WebSocket, tunneled through SSH port forwarding.
  - Remote listens on localhost:9321
  - Local forwards via: ssh -L <port>:localhost:9321 user@host
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


# =============================================================================
# Transport constants
# =============================================================================

RPC_PORT = 9321


# =============================================================================
# Message types
# =============================================================================

class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"


# =============================================================================
# RPC methods (local -> remote)
# =============================================================================

class Method(str, Enum):
    PING = "ping"
    RUN = "run"
    QUEUE_ADD = "queue_add"
    CANCEL = "cancel"
    PAUSE = "pause"
    RESUME = "resume"
    STATUS = "status"
    GPU_PROCESSES = "gpu_processes"
    QUEUE_LIST = "queue_list"
    QUEUE_CLEAR = "queue_clear"
    QUEUE_REMOVE = "queue_remove"
    QUEUE_SET = "queue_set"
    GET_MAPPING = "get_mapping"
    LIST_MAPPINGS = "list_mappings"
    GET_CRASH_LOG = "get_crash_log"
    LIST_CRASH_LOGS = "list_crash_logs"



# =============================================================================
# Events (remote -> local, unsolicited)
# =============================================================================

class Event(str, Enum):
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_RUN_ID = "experiment_run_id"
    EXPERIMENT_FINISHED = "experiment_finished"
    EXPERIMENT_FAILED = "experiment_failed"
    QUEUE_CHANGED = "queue_changed"
    HUB_SYNC_FAILED = "hub_sync_failed"


# =============================================================================
# Error codes
# =============================================================================

class ErrorCode(str, Enum):
    INVALID_METHOD = "invalid_method"
    INVALID_PARAMS = "invalid_params"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INTERNAL = "internal"


# =============================================================================
# Wire messages
# =============================================================================

_seq = 0

def _next_request_id() -> str:
    global _seq
    _seq += 1
    return f"req_{int(time.time() * 1000)}_{_seq:03d}"


@dataclass
class Request:
    method: Method
    params: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_next_request_id)

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.REQUEST,
            "id": self.id,
            "method": self.method.value,
            "params": self.params,
        })


@dataclass
class Response:
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None

    def to_json(self) -> str:
        msg: Dict[str, Any] = {"type": MessageType.RESPONSE, "id": self.id}
        if self.error is not None:
            msg["error"] = self.error
        else:
            msg["result"] = self.result or {}
        return json.dumps(msg)

    @staticmethod
    def ok(request_id: str, **result: Any) -> "Response":
        return Response(id=request_id, result=result)

    @staticmethod
    def err(request_id: str, code: ErrorCode, message: str) -> "Response":
        return Response(id=request_id, error={"code": code.value, "message": message})


@dataclass
class EventMessage:
    event: Event
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.EVENT,
            "event": self.event.value,
            "data": self.data,
            "timestamp": self.timestamp,
        })


def parse_message(raw: str) -> Request | Response | EventMessage:
    """Parse a JSON message string into the appropriate type.

    Raises ValueError on invalid messages.
    """
    data = json.loads(raw)
    msg_type = data.get("type")

    if msg_type == MessageType.REQUEST:
        return Request(
            id=data["id"],
            method=Method(data["method"]),
            params=data.get("params", {}),
        )
    elif msg_type == MessageType.RESPONSE:
        return Response(
            id=data["id"],
            result=data.get("result"),
            error=data.get("error"),
        )
    elif msg_type == MessageType.EVENT:
        return EventMessage(
            event=Event(data["event"]),
            data=data.get("data", {}),
            timestamp=data.get("timestamp", ""),
        )
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
