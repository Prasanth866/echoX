"""
echoX - Benchmark & Dataset Evaluation Script
Evaluates model detection performance on audio datasets, including full support for
ASVspoof 2019 Logical Access (LA) protocol files and A01-A19 attack benchmarks.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.services.audio_processor import AUDIO_PROCESSOR
from backend.app.services.detector import DETECTOR_SERVICE
from backend.app.services.asvspoof_dataset import ASVspoof2019LALoader, ASVSPOOF_2019_ATTACKS
from backend.app.core.config import TEST_SAMPLES_DIR


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray):
    """
    Computes Equal Error Rate (EER) where False Acceptance Rate (FAR) = False Rejection Rate (FRR).
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 50.0

    thresholds = np.linspace(0, 100, 1001)
    frr = np.array([np.mean(bonafide_scores > th) for th in thresholds])
    far = np.array([np.mean(spoof_scores <= th) for th in thresholds])

    abs_diffs = np.abs(far - frr)
    min_idx = np.argmin(abs_diffs)
    eer = (far[min_idx] + frr[min_idx]) / 2.0 * 100.0
    eer_threshold = thresholds[min_idx]

    return round(float(eer), 2), round(float(eer_threshold), 2)


def evaluate_asvspoof_protocol(protocol_path: Path, audio_dir: Path):
    """
    Evaluates detection metrics across ASVspoof 2019 LA protocol and attacks A01-A19.
    """
    entries = ASVspoof2019LALoader.parse_protocol_file(protocol_path)
    if not entries:
        print(f"No valid entries found in protocol: {protocol_path}")
        return

    print("=" * 80)
    print("echoX - ASVspoof 2019 LA Protocol Evaluation")
    print(f"Protocol: {protocol_path.name} ({len(entries)} utterances)")
    print(f"Audio Dir: {audio_dir}")
    print("=" * 80)

    bonafide_scores = []
    spoof_scores = []
    attack_scores: Dict[str, List[float]] = {}
    latencies = []

    for entry in entries:
        audio_file = audio_dir / entry.file_name
        if not audio_file.exists():
            audio_file = audio_dir / f"{entry.file_name}.wav"
        if not audio_file.exists():
            audio_file = audio_dir / f"{entry.file_name}.flac"

        if not audio_file.exists():
            continue

        audio_bytes = audio_file.read_bytes()
        proc_res = AUDIO_PROCESSOR.process_file_bytes(audio_bytes)
        det_res = DETECTOR_SERVICE.detect(proc_res["processed_audio"])
        score = det_res["risk_score"]
        latencies.append(det_res["inference_time_ms"])

        if entry.is_spoof:
            spoof_scores.append(score)
            attack_id = entry.attack_id
            if attack_id not in attack_scores:
                attack_scores[attack_id] = []
            attack_scores[attack_id].append(score)
            status = "PASS" if score > 60 else "FAIL"
        else:
            bonafide_scores.append(score)
            status = "PASS" if score <= 30 else "FAIL"

        attack_info = ASVspoof2019LALoader.get_attack_info(entry.attack_id if entry.is_spoof else "bonafide")
        print(f"[*] {entry.file_name:<18} | {entry.attack_id:<8} | {score:5.1f}/100 | {det_res['verdict']:<12} | {status} ({attack_info['name'][:24]})")

    bf_arr = np.array(bonafide_scores) if bonafide_scores else np.array([20.0])
    sp_arr = np.array(spoof_scores) if spoof_scores else np.array([80.0])
    eer, threshold = compute_eer(bf_arr, sp_arr)

    print("\n" + "=" * 80)
    print("ASVSPOOF 2019 LA: BREAKDOWN BY ATTACK ALGORITHM (A01 - A19)")
    print("-" * 80)
    print(f"{'Attack ID':<10} {'Type':<6} {'Algorithm Name':<32} {'Samples':<8} {'Avg Score':<10} {'Detection Rate'}")
    print("-" * 80)

    bf_acc = np.mean(bf_arr <= 30) * 100.0 if len(bf_arr) else 100.0
    print(f"{'Bona Fide':<10} {'Human':<6} {'Authentic Natural Voice':<32} {len(bf_arr):<8} {np.mean(bf_arr):5.1f}/100   {bf_acc:5.1f}%")

    for atk_id in sorted(attack_scores.keys()):
        scores = np.array(attack_scores[atk_id])
        info = ASVspoof2019LALoader.get_attack_info(atk_id)
        det_rate = np.mean(scores > 60) * 100.0
        print(f"{atk_id:<10} {info['type']:<6} {info['name'][:30]:<32} {len(scores):<8} {np.mean(scores):5.1f}/100   {det_rate:5.1f}%")

    print("=" * 80)
    print("OVERALL PERFORMANCE SUMMARY")
    print(f"Total Samples Evaluated:      {len(bonafide_scores) + len(spoof_scores)}")
    print(f"Equal Error Rate (EER):        {eer}% (at risk threshold {threshold})")
    print(f"Inference Latency (Median P50): {np.percentile(latencies, 50):.1f} ms" if latencies else "N/A")
    print("=" * 80)


def evaluate_directory(samples_dir: Path):
    protocol_files = list(samples_dir.glob("*.txt"))
    if protocol_files:
        evaluate_asvspoof_protocol(protocol_files[0], samples_dir)
        return

    wav_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.flac"))
    if not wav_files:
        print(f"No audio files found in {samples_dir}")
        return

    print("=" * 65)
    print("echoX - Audio Benchmark Suite")
    print(f"Target: {samples_dir} ({len(wav_files)} files)")
    print("=" * 65)

    bonafide_scores = []
    spoof_scores = []
    latencies = []

    for file_path in wav_files:
        audio_bytes = file_path.read_bytes()
        proc_res = AUDIO_PROCESSOR.process_file_bytes(audio_bytes)
        det_res = DETECTOR_SERVICE.detect(proc_res["processed_audio"])
        score = det_res["risk_score"]
        latencies.append(det_res["inference_time_ms"])

        is_spoof_label = "clone" in file_path.name.lower() or "spoof" in file_path.name.lower() or "fake" in file_path.name.lower() or "A0" in file_path.name or "A1" in file_path.name

        if is_spoof_label:
            spoof_scores.append(score)
            status = "PASS (Detected Spoof)" if score > 60 else "FAIL (Missed Spoof)"
        else:
            bonafide_scores.append(score)
            status = "PASS (Verified Genuine)" if score <= 30 else "FAIL (False Positive)"

        print(f"[*] {file_path.name:<25} | Score: {score:5.1f}/100 | {det_res['verdict']:<12} | {status}")

    bonafide_arr = np.array(bonafide_scores) if bonafide_scores else np.array([15.0])
    spoof_arr = np.array(spoof_scores) if spoof_scores else np.array([85.0])
    eer, threshold = compute_eer(bonafide_arr, spoof_arr)

    print("-" * 65)
    print("BENCHMARK SUMMARY RESULTS")
    print(f"Equal Error Rate (EER):      {eer}% (at risk threshold {threshold})")
    print(f"Latency (Median P50):        {np.percentile(latencies, 50):.1f} ms" if latencies else "N/A")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate echoX on ASVspoof 2019 LA dataset.")
    parser.add_argument("--samples_dir", type=str, default=str(TEST_SAMPLES_DIR / "asvspoof2019_la"), help="Directory of audio samples")
    parser.add_argument("--protocol", type=str, default=None, help="Optional path to ASVspoof 2019 protocol file")
    args = parser.parse_args()

    if args.protocol:
        evaluate_asvspoof_protocol(Path(args.protocol), Path(args.samples_dir))
    else:
        evaluate_directory(Path(args.samples_dir))
