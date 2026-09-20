"""
echoX - Real-Time Streaming WebSocket Endpoint
Accepts continuous PCM audio chunks from client microphone or stream,
buffers into 64,600-sample AASIST windows,
and returns raw frame-by-frame risk telemetry in real time.
"""

import uuid
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.services.audio_processor import StreamBuffer
from backend.app.services.detector import DETECTOR_SERVICE
from backend.app.core.config import AUDIO_CONFIG, MODEL_CONFIG

router = APIRouter(tags=["Real-Time Streaming"])


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    Receives raw 16-bit PCM bytes (16 kHz mono) or float32 arrays.
    Buffers into 64,000-sample (4.0s) windows and runs Facebook Wav2Vec 2.0 inference.
    Returns raw frame-by-frame risk telemetry.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    stream_buffer = StreamBuffer()

    try:
        await websocket.send_json({
            "event": "connected",
            "session_id": session_id,
            "sample_rate": AUDIO_CONFIG.SAMPLE_RATE,
            "target_window": AUDIO_CONFIG.SAMPLE_WINDOW,
            "model": MODEL_CONFIG.MODEL_NAME,
            "status": "ready"
        })

        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif "text" in message and message["text"]:
                if message["text"] == "ping":
                    await websocket.send_json({"event": "pong"})
                    continue
                continue
            else:
                continue

            window = stream_buffer.append_chunk(chunk)

            if window is not None:
                det_result = DETECTOR_SERVICE.detect(window)

                await websocket.send_json({
                    "event": "telemetry",
                    "session_id": session_id,
                    "risk_score": det_result["risk_score"],
                    "verdict": det_result["verdict"],
                    "action": det_result["action"],
                    "inference_time_ms": det_result["inference_time_ms"],
                    "spoof_probability": det_result["spoof_probability"],
                    "color": det_result["color"],
                    "description": det_result["description"]
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass
