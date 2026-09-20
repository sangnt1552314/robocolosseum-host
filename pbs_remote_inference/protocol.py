"""Our own WebSocket message protocol for the launcher-orchestrated workflow.

Control/metadata travel as JSON text frames. Camera images are transported as
base64-encoded JPEG strings inside the JSON for version 1; the codec is isolated
in :func:`encode_jpeg` / :func:`decode_jpeg` so binary frames can replace it
later without changing the schemas.

Handshake adds a ``session_id`` so the local server can bind an incoming worker
to the session it launched via the API::

    worker  --worker_ready{session_id, model}-->  local server
    server  --worker_ack----------------------->  worker
    server  --observation---------------------->  worker
    worker  --action--------------------------->  server   (or --error-->)
"""

from __future__ import annotations

import base64
import io
from typing import Annotated, Any, Literal, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    TypeAdapter,
    model_validator,
)

FieldName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
Vector = Annotated[list[FiniteFloat], Field(min_length=1, max_length=512)]

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerReady(_Msg):
    type: Literal["worker_ready"] = "worker_ready"
    session_id: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    worker_id: str = Field(default="", max_length=128)
    action_horizon: int = Field(default=16, ge=1, le=256, strict=True)


class WorkerAck(_Msg):
    type: Literal["worker_ack"] = "worker_ack"
    session_id: str = Field(min_length=1, max_length=128)
    accepted: bool = True
    message: str = Field(default="", max_length=512)


class Observation(_Msg):
    type: Literal["observation"] = "observation"
    request_id: str = Field(min_length=1, max_length=128)
    timestamp: FiniteFloat = Field(ge=0)
    instruction: str = Field(max_length=8192)
    robot_state: dict[FieldName, Vector]
    # generic camera slot name -> base64-encoded JPEG (see encode/decode_jpeg).
    cameras: dict[FieldName, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(_Msg):
    type: Literal["action"] = "action"
    request_id: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    actions: list[Vector] = Field(min_length=1, max_length=256)
    inference_time_ms: FiniteFloat = Field(ge=0)

    @model_validator(mode="after")
    def _rectangular(self) -> "Action":
        if any(len(row) != len(self.actions[0]) for row in self.actions):
            raise ValueError(
                "actions must be a rectangular (horizon, action_dim) chunk"
            )
        return self


class WorkerError(_Msg):
    type: Literal["error"] = "error"
    request_id: str | None = None
    message: str = Field(max_length=2048)


Message = Union[WorkerReady, WorkerAck, Observation, Action, WorkerError]
_MESSAGE_ADAPTER: TypeAdapter[Message] = TypeAdapter(
    Annotated[Message, Field(discriminator="type")]
)


def parse_message(raw: str | bytes) -> Message:
    """Validate and parse an incoming JSON frame into a typed message."""
    return _MESSAGE_ADAPTER.validate_json(raw)


def dump_message(message: Message) -> str:
    """Serialise a message to a compact JSON string."""
    return message.model_dump_json()


def encode_jpeg(image: Any, quality: int = 85) -> str:
    """Encode an HWC RGB image as a base64 JPEG string."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB image, got shape {arr.shape}")
    buffer = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_jpeg(data: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string back into an HWC RGB uint8 array."""
    raw = base64.b64decode(data, validate=True)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("each compressed image must be 1 byte to 8 MiB")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("only JPEG and PNG images are supported")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("image exceeds the 4 megapixel limit")
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid compressed image") from exc
