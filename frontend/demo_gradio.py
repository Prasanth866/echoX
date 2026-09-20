"""
echoX - Facebook Wav2Vec 2.0 Deepfake Voice Protection (Clean Gradio UI)
Interactive, pitch-ready multi-tier interface featuring:
- 4 Quick-Calibration Audio Benchmarks (1-click evaluation)
- Live Microphone deepfake verification
- Facebook Wav2Vec 2.0 neural representations (768-dim latent space)
- Visual Risk Score Gauge (0-100) with 3-tier enterprise policy actions
- Backend compliance logging
"""

import os
import sys
from pathlib import Path
import gradio as gr
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.services.audio_processor import AUDIO_PROCESSOR, clean_and_normalize_audio
from backend.app.services.detector import DETECTOR_SERVICE
from backend.app.core.audit_engine import AUDIT_LOGGER
from backend.app.core.config import TEST_SAMPLES_DIR, MODEL_CONFIG


def analyze_file_input(audio_filepath: str):
    """Callback for file upload or preset selection."""
    if not audio_filepath or not os.path.exists(audio_filepath):
        return (
            "<div style='color: #ef4444; font-weight: bold; text-align: center; padding: 20px;'>Please select a sample or upload an audio file.</div>",
            "N/A",
            "N/A",
            "N/A"
        )

    with open(audio_filepath, "rb") as f:
        audio_bytes = f.read()

    proc_res = AUDIO_PROCESSOR.process_file_bytes(audio_bytes)
    det_res = DETECTOR_SERVICE.detect(proc_res["processed_audio"])

    risk_score = det_res["risk_score"]
    color = det_res["color"]
    verdict = det_res["verdict"]
    action = det_res["action"]

    AUDIT_LOGGER.log_event(
        audio_bytes=audio_bytes,
        risk_score=risk_score,
        verdict=verdict,
        action=action,
        filename=Path(audio_filepath).name
    )

    latents = det_res.get("latents_meta", {})
    latent_norm = latents.get("latent_norm", 0.0)

    meter_html = f"""
    <div style="background: linear-gradient(135deg, #0b1120 0%, #0f172a 100%); border-radius: 16px; padding: 26px; text-align: center; border: 1.5px solid {color}55; box-shadow: 0 10px 30px -5px {color}22;">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 1.8px;">Facebook Wav2Vec 2.0 Neural Verdict</div>
        <div style="font-size: 56px; font-weight: 900; color: {color}; margin: 8px 0; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.04em;">
            {risk_score:.1f}<span style="font-size: 22px; color: #64748b;"> / 100</span>
        </div>
        <div style="display: inline-block; background: {color}18; color: {color}; border: 1px solid {color}66; padding: 6px 20px; border-radius: 9999px; font-weight: 800; font-size: 15px; margin-bottom: 16px; letter-spacing: 0.04em;">
            {verdict} &bull; {action}
        </div>
        <div style="background: rgba(255,255,255,0.08); border-radius: 9999px; height: 12px; overflow: hidden; margin: 12px auto; max-width: 480px;">
            <div style="width: {max(5, risk_score)}%; height: 100%; background: {color}; border-radius: 9999px; box-shadow: 0 0 12px {color}; transition: width 0.6s ease-in-out;"></div>
        </div>
        <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 12px; line-height: 1.4;">{det_res['description']}</div>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 16px; font-size: 12px; color: #94a3b8; font-family: monospace;">
            <span>Spoof: {det_res['spoof_probability']*100:.1f}%</span>
            <span>Bona Fide: {det_res['bonafide_probability']*100:.1f}%</span>
            <span>Latent Norm: {latent_norm}</span>
        </div>
    </div>
    """

    score_str = f"{risk_score:.1f} / 100"
    latency_str = f"{det_res['inference_time_ms']} ms"

    return meter_html, score_str, action, latency_str


def analyze_mic_input(audio_input):
    """Callback for live microphone recording."""
    if audio_input is None:
        return (
            "<div style='color: #94a3b8; padding: 24px; text-align: center;'>Speak clearly into your microphone for 3-4 seconds and click <strong>Verify Spoken Audio</strong>.</div>",
            "N/A", "N/A", "N/A"
        )

    try:
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                return "<div style='color: #ef4444;'>Audio file not found. Please record again.</div>", "N/A", "N/A", "N/A"
            with open(audio_input, "rb") as f:
                audio_bytes = f.read()
            data, sr = AUDIO_PROCESSOR.load_audio_from_bytes(audio_bytes)
        elif isinstance(audio_input, tuple):
            sr, audio_data = audio_input
            if audio_data.dtype == np.int16:
                data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                data = audio_data.astype(np.float32) / 2147483648.0
            else:
                data = audio_data.astype(np.float32)
        elif isinstance(audio_input, dict):
            sr = audio_input.get("sample_rate", 16000)
            audio_data = audio_input.get("data")
            data = audio_data.astype(np.float32) / (32768.0 if audio_data.dtype == np.int16 else 1.0)
        else:
            return "<div style='color: #ef4444;'>Unsupported audio format.</div>", "N/A", "N/A", "N/A"

        tensor_norm, is_active = clean_and_normalize_audio(data, sr)
        if not is_active or tensor_norm is None:
            return (
                "<div style='background: #1e293b; border-left: 4px solid #64748b; padding: 18px; border-radius: 8px; text-align: center; color: #94a3b8;'>"
                "<strong>No Active Speech Detected:</strong> Audio was silent or ambient noise (under 0.5s speech). "
                "Please speak clearly into the microphone for 3-4 seconds and verify."
                "</div>",
                "0.0 / 100", "INACTIVE", "0.0 ms"
            )

        windowed = tensor_norm.squeeze(0).cpu().numpy()
        det_res = DETECTOR_SERVICE.detect(windowed)
        risk_score = det_res["risk_score"]
        color = det_res["color"]
        verdict = det_res["verdict"]
        action = det_res["action"]

        meter_html = f"""
        <div style="background: linear-gradient(135deg, #0b1120 0%, #0f172a 100%); border-radius: 16px; padding: 26px; text-align: center; border: 1.5px solid {color}55; box-shadow: 0 10px 30px -5px {color}22;">
            <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 1.8px;">Live Microphone &bull; Facebook Wav2Vec 2.0</div>
            <div style="font-size: 56px; font-weight: 900; color: {color}; margin: 8px 0; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.04em;">
                {risk_score:.1f}<span style="font-size: 22px; color: #64748b;"> / 100</span>
            </div>
            <div style="display: inline-block; background: {color}18; color: {color}; border: 1px solid {color}66; padding: 6px 20px; border-radius: 9999px; font-weight: 800; font-size: 15px; margin-bottom: 16px; letter-spacing: 0.04em;">
                {verdict} &bull; {action}
            </div>
            <div style="background: rgba(255,255,255,0.08); border-radius: 9999px; height: 12px; overflow: hidden; margin: 12px auto; max-width: 480px;">
                <div style="width: {max(5, risk_score)}%; height: 100%; background: {color}; border-radius: 9999px; box-shadow: 0 0 12px {color}; transition: width 0.6s ease-in-out;"></div>
            </div>
            <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 12px; line-height: 1.4;">{det_res['description']}</div>
        </div>
        """
        return meter_html, f"{risk_score:.1f} / 100", action, f"{det_res['inference_time_ms']} ms"

    except Exception as e:
        return f"<div style='color: #ef4444;'>Error analyzing microphone input: {str(e)}</div>", "N/A", "N/A", "N/A"


custom_css = """
body {
    background-color: #070a13 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #f8fafc;
}
.gradio-container {
    max-width: 1240px !important;
    margin: auto !important;
    padding-top: 1.5rem !important;
}
.header-box {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 24px;
    backdrop-filter: blur(16px);
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.35);
    color: #60a5fa;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
    font-family: monospace;
}
.preset-btn {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.preset-btn:hover {
    transform: translateY(-2px) !important;
}
"""

with gr.Blocks(title="echoX - Facebook Wav2Vec 2.0 Deepfake Shield", css=custom_css) as demo:
    with gr.Column(elem_classes=["header-box"]):
        gr.HTML(
            """
            <div class="header-badge">FACEBOOK WAV2VEC 2.0 CORE &bull; 768-D LATENT SPACE &bull; 16 kHz</div>
            <h1 style="font-size: 30px; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                echo<span style="color: #3b82f6;">X</span> Voice Anti-Spoofing & Deepfake Shield
            </h1>
            <p style="color: #94a3b8; font-size: 14.5px; margin-top: 6px;">
                Real-time acoustic representation and neural verification engine protecting voice transactions against AI voice cloning, neural vocoders, and acoustic impersonation.
            </p>
            """
        )

    with gr.Tabs():
        with gr.TabItem("4 Calibration Benchmarks & File Inspector"):
            gr.Markdown("### 1-Click Evaluation of the 4 Calibration Samples (4.0s @ 16kHz):")
            
            with gr.Row():
                btn_sample_1 = gr.Button("1. Genuine Human Voice", variant="secondary", elem_classes=["preset-btn"])
                btn_sample_2 = gr.Button("2. ElevenLabs AI Clone", variant="secondary", elem_classes=["preset-btn"])
                btn_sample_3 = gr.Button("3. WaveNet Neural TTS", variant="secondary", elem_classes=["preset-btn"])
                btn_sample_4 = gr.Button("4. Voice Conversion (VC)", variant="secondary", elem_classes=["preset-btn"])

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    file_audio = gr.Audio(label="Audio Preview / Custom Upload", type="filepath")
                    sample_desc = gr.Markdown("*Select one of the 4 benchmark samples above or upload your own audio.*")
                    btn_analyze = gr.Button("Run Facebook Wav2Vec 2.0 Analysis", variant="primary", size="lg")

                with gr.Column(scale=1):
                    result_meter = gr.HTML(
                        """
                        <div style="background: #0f172a; border-radius: 16px; padding: 36px; text-align: center; border: 1px dashed rgba(255,255,255,0.15); color: #94a3b8;">
                            Select a sample above or upload audio and click <strong>Run Facebook Wav2Vec 2.0 Analysis</strong>.
                        </div>
                        """
                    )
                    with gr.Row():
                        metric_score = gr.Label(label="Risk Score")
                        metric_action = gr.Label(label="Policy Decision")
                        metric_latency = gr.Label(label="Latency")

            path_sample_1 = str(TEST_SAMPLES_DIR / "sample_1_genuine_human.wav")
            path_sample_2 = str(TEST_SAMPLES_DIR / "sample_2_cloned_elevenlabs.wav")
            path_sample_3 = str(TEST_SAMPLES_DIR / "sample_3_neural_tts.wav")
            path_sample_4 = str(TEST_SAMPLES_DIR / "sample_4_voice_conversion.wav")

            btn_sample_1.click(
                lambda: (path_sample_1, "**Selected:** Sample 1 - Authentic Human Voice (Natural vocal tract formants F1, F2, F3 with glottal pulses)."),
                outputs=[file_audio, sample_desc]
            )
            btn_sample_2.click(
                lambda: (path_sample_2, "**Selected:** Sample 2 - ElevenLabs Voice Clone (Neural vocoder high-frequency phase alignment)."),
                outputs=[file_audio, sample_desc]
            )
            btn_sample_3.click(
                lambda: (path_sample_3, "**Selected:** Sample 3 - WaveNet Neural TTS (Tacotron + WaveNet synthesis with rigid pitch contours)."),
                outputs=[file_audio, sample_desc]
            )
            btn_sample_4.click(
                lambda: (path_sample_4, "**Selected:** Sample 4 - Cross-Speaker Voice Conversion (Spectral warping and boundary phase mismatch)."),
                outputs=[file_audio, sample_desc]
            )

            btn_analyze.click(
                analyze_file_input,
                inputs=file_audio,
                outputs=[result_meter, metric_score, metric_action, metric_latency]
            )

        with gr.TabItem("Live Audio & Microphone Guard"):
            gr.Markdown(
                """
                ### Speak Live to Verify Voice Authenticity
                Speak naturally for 3-4 seconds. Facebook Wav2Vec 2.0 will analyze your acoustic vocal dynamics against synthetic voice clone profiles.
                """
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    mic_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record Microphone Audio (or Upload Voice Recording)")
                    btn_mic_verify = gr.Button("Verify Spoken Audio", variant="primary", size="lg")
                with gr.Column(scale=1):
                    mic_meter = gr.HTML(
                        """
                        <div style="background: #0f172a; border-radius: 16px; padding: 36px; text-align: center; border: 1px dashed rgba(255,255,255,0.15); color: #94a3b8;">
                            Record speech and click <strong>Verify Spoken Audio</strong>.
                        </div>
                        """
                    )
                    with gr.Row():
                        mic_score = gr.Label(label="Live Risk Score")
                        mic_action = gr.Label(label="Policy Decision")
                        mic_latency = gr.Label(label="Inference Latency")

            btn_mic_verify.click(
                analyze_mic_input,
                inputs=mic_input,
                outputs=[mic_meter, mic_score, mic_action, mic_latency]
            )

        with gr.TabItem("Enterprise 3-Tier Policy"):
            gr.Markdown(
                """
                ### Automated Defense Policy
                | Risk Score | Tier | Policy Action | Defense Response |
                | :---: | :---: | :---: | :--- |
                | **0 - 30** | **Low Risk** | `ALLOW` | Bona fide verified human speech. Normal transaction pass-through. |
                | **31 - 60** | **Medium Risk** | `CHALLENGE_MFA` | Ambiguous acoustic artifacts. Triggers step-up biometric or SMS challenge. |
                | **61 - 100** | **High Risk** | `BLOCK_ALERT` | Synthetic AI voice clone or vocoder detected. Immediate session block. |

                ```text
                [Raw Audio: 4.0s @ 16 kHz] 
                        │
                        ▼
                [Facebook Wav2Vec 2.0 Backbone] ──▶ 768-D Latent Speech Embeddings
                        │
                        ▼
                [Deepfake Classifier] ────────────▶ Calibrated Spoof Probability
                        │
                        ▼
                [Risk Policy Engine] ─────────────▶ 0 - 100 Risk Score & Policy Decision
                ```
                """
            )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue")
    )
