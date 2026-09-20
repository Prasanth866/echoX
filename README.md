# echoX: Facebook Wav2Vec 2.0 Deepfake Voice & Audio Anti-Spoofing Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Presentation: Gradio](https://img.shields.io/badge/Gradio-4.20+-orange.svg)](https://gradio.app)
[![Model: Wav2Vec2](https://img.shields.io/badge/Backbone-Facebook%20Wav2Vec%202.0-blue.svg)](https://huggingface.co/facebook/wav2vec2-base)

echoX is a multi-tier platform designed to detect synthetic speech, AI voice clones, and acoustic impersonation in real time. It utilizes Facebook / Meta's Wav2Vec 2.0 neural backbone (`facebook/wav2vec2-base` fine-tuned for deepfake voice detection) paired with a modular Risk Engine, standardized 4.0-second window slicing and zero-padding (64,000 samples @ 16 kHz), live audio microphone capture, and backend compliance logging.

---

## Multi-Tier System Architecture

```text
echoX/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes_stream.py      # WebSocket endpoint for real-time raw frame streaming
│   │   │   └── routes_upload.py      # REST endpoint for file upload analysis & compliance logging
│   │   ├── core/
│   │   │   ├── config.py             # Risk thresholds (0-30, 31-60, 61-100) & Wav2Vec2 settings
│   │   │   ├── risk_engine.py        # 3-tier security policies (ALLOW, CHALLENGE_MFA, BLOCK_ALERT)
│   │   │   └── audit_engine.py       # Backend compliance audit logger
│   │   ├── models/
│   │   │   ├── wav2vec.py            # Facebook Wav2Vec 2.0 Deepfake Detector (768-D representations)
│   │   │   └── aasist.py             # Legacy architecture reference
│   │   ├── services/
│   │   │   ├── audio_processor.py    # Resampling (16kHz), 4.0s (64k samples) slicing & zero-padding
│   │   │   └── detector.py           # Facebook Wav2Vec 2.0 detector & risk scoring pipeline
│   │   └── main.py                   # FastAPI / Uvicorn server entrypoint
│   └── weights/                      # Model checkpoints
│
├── frontend/
│   ├── demo_gradio.py                # Ultra-clean 1-click pitch UI with 4 sample buttons & live mic
│   └── web_dashboard/                # Modern dark glassmorphic web operations dashboard
│
├── scripts/
│   ├── live_mic_stream.py            # Standalone CLI microphone streamer
│   ├── evaluate_dataset.py           # Computes EER (Equal Error Rate) on test samples
│   └── generate_samples.py           # Generates 4 distinct 4.0s calibration samples
│
├── tests/
│   ├── samples/
│   │   ├── sample_1_genuine_human.wav       # Authentic natural human voice
│   │   ├── sample_2_cloned_elevenlabs.wav   # AI neural vocoder voice clone
│   │   ├── sample_3_neural_tts.wav          # WaveNet / Tacotron synthetic speech
│   │   └── sample_4_voice_conversion.wav    # Cross-speaker voice conversion
│   └── test_inference.py             # Pytest test suite
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Architecture documentation & pitch guide
```

---

## Key Specifications

1. **Audio Ingestion (Standard 4.0-Second Window):**
   - Standardizes all incoming audio chunks to 16 kHz mono.
   - Enforces the fixed 64,000-sample window (exactly 4.0 seconds): shorter inputs are zero-padded; longer inputs are sliced.

2. **Facebook Wav2Vec 2.0 Neural Representations:**
   - 768-dimensional latent speech representations capturing micro-pitch jitter, vocal tract formants, and vocoder phase discontinuities.

3. **4 Preloaded Calibration Benchmarks:**
   - **Sample 1:** Bona Fide Authentic Human Voice (Low Risk, `ALLOW`).
   - **Sample 2:** ElevenLabs AI Voice Clone (High Risk, `BLOCK_ALERT`).
   - **Sample 3:** WaveNet / Tacotron Neural TTS (High Risk, `BLOCK_ALERT`).
   - **Sample 4:** Voice Conversion (High Risk, `BLOCK_ALERT`).

4. **Live Audio & Microphone Guard:**
   - Real-time spoken audio verification with browser and CLI streaming support.

5. **Compliance Audit Ledger (Backend Logic Only):**
   - Verification events, timestamps, audio SHA-256 digests, and security actions are recorded in tamper-evident compliance logs.

---

## 3-Tier Enterprise Security Policy

| Risk Score | Tier Classification | Policy Action | Operational Trigger |
| :---: | :---: | :---: | :--- |
| **0 - 30** | **Low Risk** | `ALLOW` | Verified bona fide human voice. Normal transaction flow. |
| **31 - 60** | **Medium Risk** | `CHALLENGE_MFA` | Indeterminate spectral artifacts. Secondary challenge triggered. |
| **61 - 100** | **High Risk** | `BLOCK_ALERT` | Spoofed or synthetic cloned voice detected. Immediate block and audit alert. |

---

## Quickstart Guide

### 1. Installation & Environment

```bash
cd echoX
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/generate_samples.py
```

### 2. Launch FastAPI Backend & Clean Web Dashboard

```bash
uvicorn backend.app.main:app --reload --port 8000
```
- Clean Web Dashboard: `http://127.0.0.1:8000/` (or `/dashboard`)
- API Health Check: `http://127.0.0.1:8000/health`
- OpenAPI Documentation: `http://127.0.0.1:8000/docs`
- Streaming WebSocket: `ws://127.0.0.1:8000/ws/stream`

### 3. Launch Gradio Pitch Presentation UI

```bash
python frontend/demo_gradio.py
```
Open `http://localhost:7860` for the clean presentation dashboard.

### 4. Run Automated Tests

```bash
pytest tests/test_inference.py -v
```
