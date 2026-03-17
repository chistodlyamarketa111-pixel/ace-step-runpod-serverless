import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import Pedalboard, Compressor, Gain, LowShelfFilter, HighShelfFilter, PeakFilter, Limiter
from pedalboard import HighpassFilter, LowpassFilter
import sys
import os

def master_track(input_path, output_path, sr=48000, target_lufs=-14.0):
    print(f"Loading {input_path}...")
    audio, file_sr = sf.read(input_path)
    assert file_sr == sr, f"Expected {sr}Hz, got {file_sr}Hz"
    
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    
    print(f"Input: {audio.shape}, duration={audio.shape[0]/sr:.1f}s, SR={sr}")
    
    audio_t = audio.T.astype(np.float32)
    
    eq_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),
        LowShelfFilter(cutoff_frequency_hz=80.0, gain_db=1.5),
        PeakFilter(cutoff_frequency_hz=250.0, gain_db=-2.5, q=1.0),
        PeakFilter(cutoff_frequency_hz=400.0, gain_db=-1.5, q=0.8),
        PeakFilter(cutoff_frequency_hz=3000.0, gain_db=1.5, q=0.7),
        PeakFilter(cutoff_frequency_hz=5000.0, gain_db=1.0, q=0.8),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.5),
        LowpassFilter(cutoff_frequency_hz=20000.0),
    ])
    
    print("Applying EQ...")
    audio_eq = eq_board(audio_t, sr)
    
    comp_board = Pedalboard([
        Compressor(threshold_db=-20.0, ratio=3.0, attack_ms=10.0, release_ms=100.0),
    ])
    
    print("Applying compression...")
    audio_comp = comp_board(audio_eq, sr)
    
    print("Applying stereo widening...")
    mid = (audio_comp[0] + audio_comp[1]) / 2.0
    side = (audio_comp[0] - audio_comp[1]) / 2.0
    side_widened = side * 1.3
    audio_wide = np.array([mid + side_widened, mid - side_widened])
    
    limit_board = Pedalboard([
        Gain(gain_db=2.0),
        Limiter(threshold_db=-1.0, release_ms=50.0),
    ])
    
    print("Applying limiter...")
    audio_limited = limit_board(audio_wide, sr)
    
    print(f"Normalizing to {target_lufs} LUFS...")
    audio_out = audio_limited.T
    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio_out)
    print(f"  Current LUFS: {current_lufs:.1f}")
    
    if not np.isinf(current_lufs):
        audio_out = pyln.normalize.loudness(audio_out, current_lufs, target_lufs)
    
    final_lufs = meter.integrated_loudness(audio_out)
    peak = np.max(np.abs(audio_out))
    
    if peak > 0.99:
        audio_out = audio_out * (0.99 / peak)
        print(f"  True-peak limited to 0.99")
    
    print(f"  Final LUFS: {final_lufs:.1f}")
    print(f"  Peak: {peak:.4f}")
    
    sf.write(output_path, audio_out, sr, subtype='PCM_24')
    print(f"Saved {output_path} (24-bit PCM)")
    
    rms_in = np.sqrt(np.mean(audio[:, 0]**2))
    rms_out = np.sqrt(np.mean(audio_out[:, 0]**2))
    print(f"  RMS in: {rms_in:.4f} -> RMS out: {rms_out:.4f}")
    print("=== MASTERING DONE ===")

if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else '/workspace/track_base_macan_lora_48k.wav'
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace('.wav', '_mastered.wav')
    master_track(inp, out)
