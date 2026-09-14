"""
echoX - Real-Time Microphone Streaming CLI
Captures microphone audio, sends chunks to the echoX streaming engine via WebSocket,
and displays raw frame-by-frame risk telemetry.
"""

import asyncio
import json
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import websockets
except ImportError:
    websockets = None

SERVER_WS_URL = "ws://127.0.0.1:8000/ws/stream"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 4000


def print_risk_meter(score: float, verdict: str, action: str, latency_ms: float):
    width = 30
    filled = int(width * (score / 100.0))
    bar = "=" * filled + "-" * (width - filled)

    if score <= 30:
        color = "\033[92m"
    elif score <= 60:
        color = "\033[93m"
    else:
        color = "\033[91m"
    reset = "\033[0m"

    sys.stdout.write(
        f"\r{color}[{bar}] {score:5.1f}/100 | {verdict:12} | {action:13} | {latency_ms:4.1f}ms{reset}"
    )
    sys.stdout.flush()


async def stream_audio_client(use_simulation: bool = False):
    if websockets is None:
        print("Error: websockets library is required to run live_mic_stream.py")
        return

    print("=" * 70)
    print("echoX - Real-Time Live Microphone Stream")
    print(f"Connecting to {SERVER_WS_URL}...")
    print("=" * 70)

    try:
        async with websockets.connect(SERVER_WS_URL) as ws:
            resp = await ws.recv()
            conn_info = json.loads(resp)
            print(f"Connected! Session ID: {conn_info.get('session_id')}")
            print("Listening... (Press Ctrl+C to stop)\n")

            async def receive_telemetry():
                try:
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("event") == "telemetry":
                            print_risk_meter(
                                data["risk_score"],
                                data["verdict"],
                                data["action"],
                                data.get("inference_time_ms", 0.0)
                            )
                except asyncio.CancelledError:
                    pass

            recv_task = asyncio.create_task(receive_telemetry())

            try:
                audio_stream = None
                try:
                    import sounddevice as sd
                    if not use_simulation:
                        audio_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16')
                        audio_stream.start()
                except Exception:
                    audio_stream = None

                if audio_stream:
                    print("Using live hardware microphone input...")
                    while True:
                        data, _ = audio_stream.read(CHUNK_SAMPLES)
                        await ws.send(data.tobytes())
                        await asyncio.sleep(0.05)
                else:
                    print("Simulation mode: Streaming synthetic test audio...")
                    t = 0.0
                    dt = CHUNK_SAMPLES / SAMPLE_RATE
                    while True:
                        time_mod = t % 6.0
                        if time_mod < 3.0:
                            samples = (np.sin(2 * np.pi * 220 * np.linspace(t, t + dt, CHUNK_SAMPLES)) * 10000).astype(np.int16)
                        else:
                            samples = np.random.normal(0, 100, CHUNK_SAMPLES).astype(np.int16)
                        t += dt
                        await ws.send(samples.tobytes())
                        await asyncio.sleep(dt)

            finally:
                if audio_stream:
                    audio_stream.stop()
                    audio_stream.close()
                recv_task.cancel()

    except Exception as e:
        print(f"\nConnection error: {e}")
        print("Tip: Ensure the echoX backend is running with: uvicorn backend.app.main:app --port 8000")


if __name__ == "__main__":
    sim_mode = "--simulate" in sys.argv
    try:
        asyncio.run(stream_audio_client(use_simulation=sim_mode))
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
