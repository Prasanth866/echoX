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


def analyze_mic_input(audio_input):
    """Callback for live microphone recording."""
    if audio_input is None:
        return (
            "<div style='color: #94a3b8; padding: 20px; text-align: center;'>Record spoken audio from your microphone and click verify.</div>",
            "N/A", "N/A", "N/A"
        )

    try:
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                return "<div style='color: #ef4444;'>Audio file not found. Please record again.</div>", "N/A", "N/A", "N/A"
            with open(audio_input, "rb") as f:
                audio_bytes = f.read()
            proc_res = AUDIO_PROCESSOR.process_file_bytes(audio_bytes)
            windowed = proc_res["processed_audio"]
        elif isinstance(audio_input, tuple):
            sr, audio_data = audio_input
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_float = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_float = audio_data.astype(np.float32)
            mono = AUDIO_PROCESSOR.to_mono(audio_float)
            resampled = AUDIO_PROCESSOR.resample(mono, sr)
            windowed = AUDIO_PROCESSOR.enforce_aasist_window(resampled)
        elif isinstance(audio_input, dict):
            sr = audio_input.get("sample_rate", 16000)
            audio_data = audio_input.get("data")
            audio_float = audio_data.astype(np.float32) / (32768.0 if audio_data.dtype == np.int16 else 1.0)
            mono = AUDIO_PROCESSOR.to_mono(audio_float)
            resampled = AUDIO_PROCESSOR.resample(mono, sr)
            windowed = AUDIO_PROCESSOR.enforce_aasist_window(resampled)
        else:
            return "<div style='color: #ef4444;'>Unsupported audio format.</div>", "N/A", "N/A", "N/A"

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

    except Exception as e:
        return f"<div style='color: #ef4444;'>Error analyzing microphone input: {str(e)}</div>", "N/A", "N/A", "N/A"


custom_css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.gradio-container { max-width: 1200px !important; margin: auto; }
"""

with gr.Blocks(title="echoX - Deepfake Voice Protection") as demo:
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
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Microphone Audio")
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

        with gr.TabItem("ASVspoof 2019 LA Benchmark (A01-A19)"):
            gr.Markdown("#### Evaluate against ASVspoof 2019 Logical Access (LA) Classical TTS & VC Algorithms:")
            
            asv_choices = [
                "Bona Fide - Authentic Human Voice",
                "A01 - Neural Acoustic + WaveNet (TTS)",
                "A02 - Acoustic Model + WORLD Vocoder (TTS)",
                "A03 - Linear Prediction (LPC) Vocoder (TTS)",
                "A04 - Waveform Concatenation / Unit Selection (TTS)",
                "A05 - Voice Conversion (VAE/GMM VC)",
                "A06 - Spectral Filtering VC",
                "A07 - Neural Vocoder (Unseen TTS)",
                "A08 - Low Bit-Rate Neural Codec (TTS)",
                "A09 - Direct Waveform Generation (TTS)",
                "A10 - Modern HiFi-GAN Vocoder (TTS)",
                "A11 - Sinc-Filtered Vocoder (TTS)",
                "A12 - LPC Residual Excitation (TTS)",
                "A13 - Harmonic + Noise Model (VC)",
                "A14 - Bilinear Frequency Warping (VC)",
                "A15 - CycleGAN / StarGAN (VC)",
                "A16 - High-Pitch Synthetic Speech (TTS)",
                "A17 - Low-Pitch Synthetic Speech (TTS)",
                "A18 - Diffusion-Based Vocoder (TTS)",
                "A19 - PSTN/GSM Compressed Voice (VC)"
            ]
            
            asv_sample_map = {
                "Bona Fide - Authentic Human Voice": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000000.wav"), "Natural authentic speech with human glottal vocal tract resonance."),
                "A01 - Neural Acoustic + WaveNet (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000001.wav"), "Tacotron acoustic model paired with sample-level WaveNet vocoder."),
                "A02 - Acoustic Model + WORLD Vocoder (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000002.wav"), "Transfer-function WORLD vocoder with periodic buzzing artifact."),
                "A03 - Linear Prediction (LPC) Vocoder (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000003.wav"), "Classical LPC all-pole spectral filter model."),
                "A04 - Waveform Concatenation / Unit Selection (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000004.wav"), "Concatenative unit selection with boundary phase mismatches."),
                "A05 - Voice Conversion (VAE/GMM VC)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000005.wav"), "Variational autoencoder voice conversion with residual vocoding."),
                "A06 - Spectral Filtering VC": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000006.wav"), "Transfer-function voice conversion with high-frequency tilt distortion."),
                "A07 - Neural Vocoder (Unseen TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000007.wav"), "Evaluation unseen neural vocoder with high-frequency phase artifacts."),
                "A08 - Low Bit-Rate Neural Codec (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000008.wav"), "Neural speech synthesis with sub-band vector quantization."),
                "A09 - Direct Waveform Generation (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000009.wav"), "End-to-end direct waveform neural generation."),
                "A10 - Modern HiFi-GAN Vocoder (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000010.wav"), "Tacotron 2 paired with multi-receptive field HiFi-GAN vocoder."),
                "A11 - Sinc-Filtered Vocoder (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000011.wav"), "Bandpass sinc-filtered vocoder synthesis."),
                "A12 - LPC Residual Excitation (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000012.wav"), "Linear predictive coding with glottal residual excitation."),
                "A13 - Harmonic + Noise Model (VC)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000013.wav"), "Harmonic plus noise model (HNM) voice conversion."),
                "A14 - Bilinear Frequency Warping (VC)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000014.wav"), "Bilinear frequency warping voice conversion."),
                "A15 - CycleGAN / StarGAN (VC)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000015.wav"), "Adversarial neural network voice conversion."),
                "A16 - High-Pitch Synthetic Speech (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000016.wav"), "Formant-shifted high fundamental frequency synthetic speech."),
                "A17 - Low-Pitch Synthetic Speech (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000017.wav"), "Low fundamental frequency synthesis with heavy sub-harmonics."),
                "A18 - Diffusion-Based Vocoder (TTS)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000018.wav"), "Score-based diffusion acoustic model synthesis."),
                "A19 - PSTN/GSM Compressed Voice (VC)": (str(TEST_SAMPLES_DIR / "asvspoof2019_la" / "LA_E_0000019.wav"), "Voice conversion passed through telephone bandwidth compression.")
            }
            
            with gr.Row():
                with gr.Column(scale=1):
                    asv_dropdown = gr.Dropdown(
                        choices=asv_choices,
                        value=asv_choices[0],
                        label="Select ASVspoof 2019 LA Attack or Bona Fide Sample"
                    )
                    asv_audio_player = gr.Audio(label="Audio Sample Preview", type="filepath")
                    asv_desc = gr.Textbox(label="Algorithm Technical Breakdown", interactive=False)
                    btn_asv_run = gr.Button("Evaluate Attack against AASIST Core", variant="primary")
                    
                with gr.Column(scale=1):
                    asv_meter = gr.HTML(label="ASVspoof Risk Gauge")
                    with gr.Row():
                        asv_score = gr.Label(label="Risk Score")
                        asv_action = gr.Label(label="Policy Decision")
                        asv_lat = gr.Label(label="Inference Latency")
            
            def on_asv_select(selected_name):
                path, desc = asv_sample_map.get(selected_name, ("", ""))
                return path, desc
                
            asv_dropdown.change(
                on_asv_select,
                inputs=asv_dropdown,
                outputs=[asv_audio_player, asv_desc]
            )
            
            btn_asv_run.click(
                analyze_file_input,
                inputs=asv_audio_player,
                outputs=[asv_meter, asv_score, asv_action, asv_lat]
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css
    )
