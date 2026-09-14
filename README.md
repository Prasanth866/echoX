# echoX: Multi-Tier Deepfake Voice & Audio Anti-Spoofing Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Presentation: Gradio](https://img.shields.io/badge/Gradio-4.20+-orange.svg)](https://gradio.app)

echoX is a multi-tier platform designed to detect synthetic speech, AI voice clones, and acoustic impersonation in real time. It pairs an AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) neural backbone with a modular Risk Engine, direct window slicing and zero-padding, raw frame-by-frame streaming telemetry, and backend compliance logging.

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
│   │   │   ├── config.py             # Risk thresholds (0-30, 31-60, 61-100) & model settings
│   │   │   ├── risk_engine.py        # Fusion & 3-tier security policies
│   │   │   └── audit_engine.py       # Backend compliance audit logger
│   │   ├── models/
│   │   │   ├── aasist.py             # AASIST PyTorch architecture (SincConv + HS-GAL)
│   │   │   └── wav2vec.py            # Secondary acoustic feature representation backbone
│   │   ├── services/
│   │   │   ├── audio_processor.py    # Resampling (16kHz), direct slicing & zero-padding
│   │   │   └── detector.py           # Model loader, hardcoded probs[0] & inference pipeline
│   │   └── main.py                   # FastAPI / Uvicorn server entrypoint
│   └── weights/
│       └── AASIST.pth                # Pretrained model weights
│
├── frontend/
│   ├── demo_gradio.py                # Fast, interactive 1-click presentation UI
│   └── web_dashboard/                # Operations monitor preview
│
├── notebooks/
│   ├── 01_feature_extraction.ipynb   # Spectrograms, sinc-filters, raw wave inspection
│   ├── 02_model_inference_test.ipynb # Latency benchmarking & confidence distributions
│   └── 03_risk_fusion_demo.ipynb     # Decision boundaries & frame-by-frame risk scoring
│
├── scripts/
│   ├── live_mic_stream.py            # Standalone CLI microphone streamer
│   ├── evaluate_dataset.py           # Computes EER (Equal Error Rate) on test samples
│   └── generate_samples.py           # Generates genuine & cloned test WAV calibration samples
│
├── tests/
│   ├── samples/
│   │   ├── genuine_human.wav         # Calibrated genuine human voice sample
│   │   └── cloned_elevenlabs.wav     # Calibrated AI cloned sample with vocoder artifacts
│   └── test_inference.py             # Pytest test suite
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Architecture diagram & pitch guide
```

---

## Key Prototype Specifications

1. **Audio Ingestion (Slicing / Zero-Padding Directly):**
   - Standardizes all incoming audio chunks to 16 kHz mono.
   - Enforces the fixed 64,600-sample AASIST window: shorter inputs are directly zero-padded to 64,600 samples; longer inputs are directly sliced.

2. **Streaming Output (Raw Frame-by-Frame Score):**
   - Evaluates streaming audio frames independently and emits the raw frame-by-frame risk score in real time.

3. **AASIST Logits (Hardcoded probs[0] Index):**
   - Directly indexes model output probabilities using `probs[0]` for spoof risk calculation.

4. **Audit Compliance (Backend Logic Only):**
   - Verification events, timestamps, payload digests, and security actions are recorded internally in backend compliance logs.

---

## 3-Tier Enterprise Security Policy

| Risk Score | Tier Classification | Policy Action | Operational Trigger |
| :---: | :---: | :---: | :--- |
| **0 - 30** | **Low Risk** | `ALLOW` | Verified bona fide human voice. Normal transaction flow. |
| **31 - 60** | **Medium Risk** | `CHALLENGE_MFA` | Indeterminate spectral artifacts. Secondary challenge triggered. |
| **61 - 100** | **High Risk** | `BLOCK_ALERT` | Spoofed or synthetic cloned voice detected. Immediate block and audit alert. |

---

## Quickstart Guide

### 1. Installation

```bash
# Enter directory
cd echoX

# Create virtual environment and install dependencies
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Generate calibration test audio samples
python scripts/generate_samples.py
```

### 2. Launch FastAPI Backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```
- API Health: `http://127.0.0.1:8000/health`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- WebSocket Stream: `ws://127.0.0.1:8000/ws/stream`

### 3. Launch Gradio Pitch Presentation UI

```bash
python frontend/demo_gradio.py
```
Open `http://localhost:7860` to access the interactive presentation dashboard.

### 4. Run Automated Test Suite

```bash
pytest tests/test_inference.py -v
```

---

## Evaluator Pitch Demo Flow

1. **Static Test (File Inspector Tab):**
   - Click "Load Cloned AI Sample (ElevenLabs)" -> Click "Analyze Audio Sample".
   - Result: Score indicates High Risk (>85/100) with `BLOCK_ALERT` action.
2. **Live Microphone Test (Live Mic Tab):**
   - Speak into the laptop microphone -> Click "Verify Spoken Audio".
   - Result: Confirmed Low Risk (<30/100) with `ALLOW` status.
3. **Live Impersonation Test:**
   - Play a synthetic AI voice recording from a phone into the microphone -> Demonstrate the dynamic score spiking into the High Risk zone in real time.
