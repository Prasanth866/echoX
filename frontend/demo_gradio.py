"""
echoX - Presentation Pitch Dashboard (Gradio)
Interactive, pitch-ready multi-tier interface featuring:
- 1-Click Genuine vs Cloned Audio evaluation
- Live Microphone deepfake verification
- Visual Risk Score Meter (0-100) with 3-tier policy actions
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

from backend.app.services.audio_processor import AUDIO_PROCESSOR
from backend.app.services.detector import DETECTOR_SERVICE
from backend.app.core.audit_engine import AUDIT_LOGGER
from backend.app.core.config import TEST_SAMPLES_DIR


def analyze_file_input(audio_filepath: str):
    """Callback for file upload or preset selection."""
    if not audio_filepath or not os.path.exists(audio_filepath):
        return (
            "<div style='color: #ef4444; font-weight: bold;'>Please upload or select an audio sample.</div>",
            "No data",
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

    meter_html = f"""
    <div style="background: #0f172a; border-radius: 14px; padding: 24px; text-align: center; border: 2px solid {color}; box-shadow: 0 10px 25px -5px {color}33;">
        <div style="font-size: 13px; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 1.5px;">Impersonation Risk Level</div>
        <div style="font-size: 54px; font-weight: 900; color: {color}; margin: 8px 0; font-family: monospace;">
            {risk_score:.1f}<span style="font-size: 24px; color: #64748b;"> / 100</span>
        </div>
        <div style="display: inline-block; background: {color}22; color: {color}; border: 1px solid {color}; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 16px; margin-bottom: 16px;">
            {verdict} - {action}
        </div>
        <div style="background: #334155; border-radius: 9999px; height: 14px; overflow: hidden; margin: 10px auto; max-width: 450px;">
            <div style="width: {risk_score}%; height: 100%; background: {color}; transition: width 0.6s ease-in-out;"></div>
        </div>
        <div style="font-size: 14px; color: #cbd5e1; margin-top: 10px;">{det_res['description']}</div>
    </div>
    """

    score_str = f"{risk_score:.1f} / 100"
    latency_str = f"{det_res['inference_time_ms']} ms"

    return meter_html, score_str, action, latency_str


def analyze_mic_input(audio_tuple):
    """Callback for live microphone recording."""
    if audio_tuple is None:
        return (
            "<div style='color: #94a3b8;'>Record spoken audio from your microphone to analyze.</div>",
            "N/A", "N/A", "N/A"
        )

    sr, audio_data = audio_tuple
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_float = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_float = audio_data.astype(np.float32)

    mono = AUDIO_PROCESSOR.to_mono(audio_float)
    resampled = AUDIO_PROCESSOR.resample(mono, sr)
    windowed = AUDIO_PROCESSOR.enforce_aasist_window(resampled)

    det_res = DETECTOR_SERVICE.detect(windowed)
    risk_score = det_res["risk_score"]
    color = det_res["color"]
    verdict = det_res["verdict"]
    action = det_res["action"]

    meter_html = f"""
    <div style="background: #0f172a; border-radius: 14px; padding: 24px; text-align: center; border: 2px solid {color}; box-shadow: 0 10px 25px -5px {color}33;">
        <div style="font-size: 13px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Live Microphone Verification</div>
        <div style="font-size: 54px; font-weight: 900; color: {color}; margin: 8px 0; font-family: monospace;">
            {risk_score:.1f}<span style="font-size: 24px; color: #64748b;"> / 100</span>
        </div>
        <div style="display: inline-block; background: {color}22; color: {color}; border: 1px solid {color}; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 16px; margin-bottom: 16px;">
            {verdict} - {action}
        </div>
        <div style="background: #334155; border-radius: 9999px; height: 14px; overflow: hidden; margin: 10px auto; max-width: 450px;">
            <div style="width: {risk_score}%; height: 100%; background: {color}; transition: width 0.6s ease-in-out;"></div>
        </div>
        <div style="font-size: 14px; color: #cbd5e1; margin-top: 10px;">{det_res['description']}</div>
    </div>
    """
    return meter_html, f"{risk_score:.1f} / 100", action, f"{det_res['inference_time_ms']} ms"


custom_css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.gradio-container { max-width: 1200px !important; margin: auto; }
"""

with gr.Blocks(title="echoX - Deepfake Voice Protection", css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # echoX: Deepfake Voice & Audio Anti-Spoofing Detection Platform
        ### Multi-Tier Detection Engine | AASIST Spectro-Temporal GAT | Enterprise Risk Policy
        """
    )

    with gr.Tabs():
        with gr.TabItem("File Inspector & Preset Benchmarks"):
            gr.Markdown("#### Test preloaded voice samples or upload custom audio files:")
            with gr.Row():
                with gr.Column(scale=1):
                    file_audio = gr.Audio(label="Audio Input", type="filepath")
                    with gr.Row():
                        btn_genuine = gr.Button("Load Genuine Human Sample", variant="secondary")
                        btn_cloned = gr.Button("Load Cloned AI Sample (ElevenLabs)", variant="secondary")
                    btn_analyze = gr.Button("Analyze Audio Sample", variant="primary")

                with gr.Column(scale=1):
                    result_meter = gr.HTML(label="Risk Gauge")
                    with gr.Row():
                        metric_score = gr.Label(label="Risk Score")
                        metric_action = gr.Label(label="Policy Decision")
                        metric_latency = gr.Label(label="Inference Latency")

            sample_gen_path = str(TEST_SAMPLES_DIR / "genuine_human.wav")
            sample_clone_path = str(TEST_SAMPLES_DIR / "cloned_elevenlabs.wav")

            btn_genuine.click(lambda: sample_gen_path, outputs=file_audio)
            btn_cloned.click(lambda: sample_clone_path, outputs=file_audio)

            btn_analyze.click(
                analyze_file_input,
                inputs=file_audio,
                outputs=[result_meter, metric_score, metric_action, metric_latency]
            )

        with gr.TabItem("Live Microphone Guard"):
            gr.Markdown("#### Speak live into microphone to verify authenticity vs cloned audio playback:")
            with gr.Row():
                with gr.Column(scale=1):
                    mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Microphone Stream")
                    btn_mic_verify = gr.Button("Verify Spoken Audio", variant="primary")
                with gr.Column(scale=1):
                    mic_meter = gr.HTML(label="Live Risk Telemetry")
                    with gr.Row():
                        mic_score = gr.Label(label="Live Score")
                        mic_action = gr.Label(label="Live Decision")
                        mic_latency = gr.Label(label="Latency")

            btn_mic_verify.click(
                analyze_mic_input,
                inputs=mic_input,
                outputs=[mic_meter, mic_score, mic_action, mic_latency]
            )

        with gr.TabItem("Architecture & Security Policy"):
            gr.Markdown(
                """
                ### Multi-Tier Architecture Overview
                
                ```text
                [Audio Input: Mic / PSTN / Upload]
                                |
                                v
                   [Audio Ingestion & Slicing] ---> Direct 64,600-sample padding/slicing (16 kHz)
                                |
                                v
                     [AASIST Neural Backbone]   ---> SincConv -> Res2Net -> HS-GAL Graph Attention
                                |
                                v
                      [Risk Fusion Engine]      ---> 0-100 Impersonation Risk Score (Raw frame)
                                |
                                v
                     [3-Tier Security Policy]
                     - 0-30:   ALLOW
                     - 31-60:  CHALLENGE_MFA
                     - 61-100: BLOCK_ALERT
                ```
                
                - **Direct Ingestion:** Slices or zero-pads directly to the exact 64,600-sample window required by AASIST.
                - **Streaming Output:** Provides raw frame-by-frame risk scores.
                - **AASIST Logits:** Hardcoded probs[0] index for direct spoof probability extraction.
                - **Audit Compliance:** Backend logic event logging.
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
