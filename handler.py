#!/usr/bin/env python3
"""
ACE-Step v1.5 — RunPod Serverless Handler
FULLY LAZY: Only runpod imported at startup. All heavy libs loaded on first request.
"""

import base64
import io
import os
import sys
import time
import traceback
import tempfile
import subprocess

HANDLER_VERSION = "2026-03-12-v16-fix-transformers"
print(f"[ACE-Step] ===== HANDLER STARTUP =====", flush=True)
print(f"[ACE-Step] Version: {HANDLER_VERSION}", flush=True)
print(f"[ACE-Step] Python: {sys.version}", flush=True)
print(f"[ACE-Step] PID: {os.getpid()}", flush=True)

def _ensure_transformers_version():
    min_ver = "4.57.0"
    try:
        import transformers
        from packaging.version import Version
        cur = transformers.__version__.replace(".dev0", "")
        if Version(cur) >= Version(min_ver):
            print(f"[ACE-Step] transformers {transformers.__version__} OK (>= {min_ver})", flush=True)
            return
        print(f"[ACE-Step] transformers {transformers.__version__} too old, upgrading...", flush=True)
    except ImportError:
        print(f"[ACE-Step] transformers not found, installing...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
                    f"transformers>={min_ver}", "lazy_loader"], check=True)
    import importlib
    if "transformers" in sys.modules:
        importlib.reload(sys.modules["transformers"])
    import transformers
    print(f"[ACE-Step] transformers upgraded to {transformers.__version__}", flush=True)

_ensure_transformers_version()

import runpod
print(f"[ACE-Step] runpod={runpod.__version__} OK — starting serverless worker immediately", flush=True)

torch = None
np = None
torchaudio = None
AceStepHandler = None
GenerationParams = None
GenerationConfig = None
generate_music = None
Pedalboard = None
HighpassFilter = None
LowShelfFilter = None
PeakFilter = None
HighShelfFilter = None
Compressor = None
Limiter = None
Gain = None
LowpassFilter = None
pyln = None
MASTERING_AVAILABLE = False
ENHANCE_AVAILABLE = False
_enhance_fn = None
_denoise_fn = None
_libs_loaded = False


def _lazy_import_all():
    global torch, np, torchaudio, AceStepHandler, GenerationParams, GenerationConfig, generate_music
    global Pedalboard, HighpassFilter, LowShelfFilter, PeakFilter, HighShelfFilter, Compressor, Limiter, Gain, LowpassFilter
    global pyln, MASTERING_AVAILABLE, ENHANCE_AVAILABLE, _enhance_fn, _denoise_fn, _libs_loaded

    if _libs_loaded:
        return

    print(f"[ACE-Step] ===== LAZY IMPORT START (first request) =====", flush=True)
    t0 = time.time()

    print(f"[ACE-Step] [1/6] Importing torch...", flush=True)
    import torch as _torch
    torch = _torch
    print(f"[ACE-Step] [1/6] torch={torch.__version__}, cuda={torch.version.cuda}, avail={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
        print(f"[ACE-Step] [1/6] GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f} GB", flush=True)

    print(f"[ACE-Step] [2/6] Importing numpy + torchaudio...", flush=True)
    import numpy as _np
    np = _np
    import torchaudio as _ta
    torchaudio = _ta
    print(f"[ACE-Step] [2/6] numpy={np.__version__}, torchaudio={torchaudio.__version__} OK", flush=True)

    import io as _io
    import tempfile as _tmpf
    _original_ta_load = torchaudio.load
    def _patched_ta_load(filepath, **kwargs):
        if 'backend' not in kwargs:
            kwargs['backend'] = 'soundfile'
        return _original_ta_load(filepath, **kwargs)
    torchaudio.load = _patched_ta_load

    _original_ta_save = torchaudio.save
    def _patched_ta_save(filepath, src, sample_rate, **kwargs):
        fmt = kwargs.get('format', '')
        is_mp3 = (fmt == 'mp3') or (isinstance(filepath, str) and filepath.endswith('.mp3') and not fmt)
        if is_mp3:
            kwargs.pop('compression', None)
            kwargs.pop('bit_rate', None)
            kwargs.pop('format', None)
            wav_path = None
            mp3_path = None
            try:
                with _tmpf.NamedTemporaryFile(suffix='.wav', delete=False) as wav_tmp:
                    wav_path = wav_tmp.name
                _original_ta_save(wav_path, src, sample_rate, format='wav')
                mp3_path = wav_path + '.mp3'
                import subprocess as _sp
                result = _sp.run(['ffmpeg', '-y', '-i', wav_path, '-b:a', '320k', '-q:a', '0', mp3_path],
                                 capture_output=True, timeout=120)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg MP3 encode failed (rc={result.returncode}): {result.stderr.decode()[-300:]}")
                if isinstance(filepath, _io.BytesIO):
                    with open(mp3_path, 'rb') as f:
                        filepath.write(f.read())
                    filepath.seek(0)
                elif isinstance(filepath, str):
                    import shutil
                    shutil.move(mp3_path, filepath)
                    mp3_path = None
                else:
                    with open(mp3_path, 'rb') as f:
                        filepath.write(f.read())
            finally:
                import os as _os
                if wav_path and _os.path.exists(wav_path):
                    _os.unlink(wav_path)
                if mp3_path and _os.path.exists(mp3_path):
                    _os.unlink(mp3_path)
        elif isinstance(filepath, _io.BytesIO):
            tmp_path = None
            try:
                with _tmpf.NamedTemporaryFile(suffix=f'.{fmt or "wav"}', delete=False) as tmp:
                    tmp_path = tmp.name
                _original_ta_save(tmp_path, src, sample_rate, **kwargs)
                with open(tmp_path, 'rb') as f:
                    filepath.write(f.read())
                filepath.seek(0)
            finally:
                import os as _os
                if tmp_path and _os.path.exists(tmp_path):
                    _os.unlink(tmp_path)
        else:
            _original_ta_save(filepath, src, sample_rate, **kwargs)
    torchaudio.save = _patched_ta_save
    print(f"[ACE-Step] [2/6] torchaudio patched OK", flush=True)

    print(f"[ACE-Step] [3/6] Importing ACE-Step...", flush=True)
    try:
        from acestep.handler import AceStepHandler as _AH
        AceStepHandler = _AH
        print(f"[ACE-Step] [3/6] AceStepHandler OK", flush=True)
    except Exception as e:
        print(f"[ACE-Step] [3/6] AceStepHandler FAILED: {e}", flush=True)
        traceback.print_exc()

    print(f"[ACE-Step] [4/6] Importing ACE-Step inference...", flush=True)
    try:
        from acestep.inference import GenerationParams as _GP, GenerationConfig as _GC, generate_music as _gm
        GenerationParams = _GP
        GenerationConfig = _GC
        generate_music = _gm
        print(f"[ACE-Step] [4/6] inference OK", flush=True)
    except Exception as e:
        print(f"[ACE-Step] [4/6] inference FAILED: {e}", flush=True)
        traceback.print_exc()

    print(f"[ACE-Step] [5/6] Importing mastering libs...", flush=True)
    try:
        from pedalboard import Pedalboard as _Pb, HighpassFilter as _Hpf, LowShelfFilter as _Lsf
        from pedalboard import PeakFilter as _Pf, HighShelfFilter as _Hsf, Compressor as _Cmp
        from pedalboard import Limiter as _Lim, Gain as _Gn, LowpassFilter as _Lpf
        import pyloudnorm as _pyln
        Pedalboard = _Pb
        HighpassFilter = _Hpf
        LowShelfFilter = _Lsf
        PeakFilter = _Pf
        HighShelfFilter = _Hsf
        Compressor = _Cmp
        Limiter = _Lim
        Gain = _Gn
        LowpassFilter = _Lpf
        pyln = _pyln
        MASTERING_AVAILABLE = True
        print(f"[ACE-Step] [5/6] Mastering OK", flush=True)
    except ImportError as e:
        print(f"[ACE-Step] [5/6] Mastering not available: {e}", flush=True)

    print(f"[ACE-Step] [6/6] Importing enhancement libs...", flush=True)
    try:
        from resemble_enhance.enhancer.inference import enhance as _re_enh, denoise as _re_den
        _enhance_fn = _re_enh
        _denoise_fn = _re_den
        ENHANCE_AVAILABLE = True
        print(f"[ACE-Step] [6/6] Resemble Enhance OK", flush=True)
    except ImportError:
        try:
            import noisereduce
            ENHANCE_AVAILABLE = True
            print(f"[ACE-Step] [6/6] noisereduce OK", flush=True)
        except ImportError:
            print(f"[ACE-Step] [6/6] Enhancement not available", flush=True)

    _libs_loaded = True
    print(f"[ACE-Step] ===== LAZY IMPORT DONE in {time.time()-t0:.1f}s =====", flush=True)


def enhance_audio(waveform, sample_rate, mode="enhance"):
    if not ENHANCE_AVAILABLE:
        print("[ACE-Step] Enhancement skipped (not available)", flush=True)
        return waveform, sample_rate

    print(f"[ACE-Step] Applying audio enhancement (mode={mode})...", flush=True)

    if isinstance(waveform, torch.Tensor):
        is_tensor = True
        device = waveform.device
        if waveform.ndim == 2:
            is_stereo = waveform.shape[0] == 2
            if is_stereo:
                mono = waveform.mean(dim=0)
            else:
                mono = waveform[0]
        else:
            is_stereo = False
            mono = waveform
    else:
        is_tensor = False
        mono = torch.from_numpy(waveform).float()
        if mono.ndim == 2:
            mono = mono.mean(dim=0)
        is_stereo = False

    mono = mono.cpu().float()

    if _enhance_fn is not None:
        try:
            _fn = _denoise_fn if (mode == "denoise" and _denoise_fn is not None) else _enhance_fn
            _kw = {} if mode == "denoise" else {"nfe": 32}

            if is_stereo:
                left_enh, new_sr = _fn(waveform[0].cpu().float(), sample_rate, device='cuda', **_kw)
                right_enh, _ = _fn(waveform[1].cpu().float(), sample_rate, device='cuda', **_kw)
                min_len = min(left_enh.shape[-1], right_enh.shape[-1])
                enhanced = torch.stack([left_enh[:min_len], right_enh[:min_len]])
            else:
                enhanced, new_sr = _fn(mono, sample_rate, device='cuda', **_kw)
                enhanced = enhanced.unsqueeze(0)

            print(f"[ACE-Step] Resemble Enhance: sr {sample_rate}->{new_sr}, shape {mono.shape}->{enhanced.shape}", flush=True)

            if is_tensor:
                return enhanced.to(device), new_sr
            return enhanced.numpy(), new_sr
        except Exception as e:
            print(f"[ACE-Step] Resemble Enhance error: {e}, falling back", flush=True)

    try:
        import noisereduce as nr
        audio_np = mono.numpy()
        reduced = nr.reduce_noise(y=audio_np, sr=sample_rate, stationary=False, prop_decrease=0.5)
        print(f"[ACE-Step] noisereduce applied", flush=True)
        result = torch.from_numpy(reduced).unsqueeze(0)
        if is_stereo:
            left_np = waveform[0].cpu().numpy()
            right_np = waveform[1].cpu().numpy()
            left_r = nr.reduce_noise(y=left_np, sr=sample_rate, stationary=False, prop_decrease=0.5)
            right_r = nr.reduce_noise(y=right_np, sr=sample_rate, stationary=False, prop_decrease=0.5)
            result = torch.from_numpy(np.stack([left_r, right_r]))
        if is_tensor:
            return result.to(device), sample_rate
        return result.numpy(), sample_rate
    except Exception as e:
        print(f"[ACE-Step] noisereduce error: {e}", flush=True)
        return waveform, sample_rate


def _save_audio_to_bytes(tensor, sr, fmt="mp3"):
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=True) as tmp:
        if fmt == "mp3":
            torchaudio.save(tmp.name, tensor.cpu(), sr, format="mp3", compression=-2)
        else:
            torchaudio.save(tmp.name, tensor.cpu(), sr, format=fmt)
        with open(tmp.name, "rb") as f:
            return f.read()


def master_audio(waveform, sample_rate):
    if not MASTERING_AVAILABLE:
        print("[ACE-Step] Mastering skipped (libraries not installed)", flush=True)
        return waveform

    print("[ACE-Step] Applying mastering pipeline (pod-quality)...", flush=True)

    if isinstance(waveform, torch.Tensor):
        audio_np = waveform.cpu().numpy().astype(np.float32)
    else:
        audio_np = np.array(waveform, dtype=np.float32)

    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]

    is_mono = audio_np.shape[0] == 1
    if is_mono:
        audio_np = np.concatenate([audio_np, audio_np], axis=0)

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
    processed = eq_board(audio_np, sample_rate)

    comp_board = Pedalboard([
        Compressor(threshold_db=-20.0, ratio=3.0, attack_ms=10.0, release_ms=100.0),
    ])
    processed = comp_board(processed, sample_rate)

    mid = (processed[0] + processed[1]) / 2.0
    side = (processed[0] - processed[1]) / 2.0
    side *= 1.3
    processed = np.array([mid + side, mid - side])

    limit_board = Pedalboard([
        Gain(gain_db=2.0),
        Limiter(threshold_db=-1.0, release_ms=50.0),
    ])
    processed = limit_board(processed, sample_rate)

    try:
        meter = pyln.Meter(sample_rate)
        loudness_input = processed.T
        current_lufs = meter.integrated_loudness(loudness_input)
        if not np.isinf(current_lufs) and not np.isnan(current_lufs):
            target_lufs = -14.0
            processed_out = pyln.normalize.loudness(loudness_input, current_lufs, target_lufs)
            peak = np.max(np.abs(processed_out))
            if peak > 0.99:
                processed_out = processed_out * (0.99 / peak)
            processed = processed_out.T
            print(f"[ACE-Step] LUFS normalized: {current_lufs:.1f} -> {target_lufs:.1f}", flush=True)
        else:
            print("[ACE-Step] LUFS measurement failed (silence?), skipping normalization", flush=True)
    except Exception as e:
        print(f"[ACE-Step] LUFS normalization error: {e}, skipping", flush=True)

    if is_mono:
        processed = processed[:1]

    if isinstance(waveform, torch.Tensor):
        return torch.from_numpy(processed)
    return processed

CHECKPOINT_DIR = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/app/checkpoints")
PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/app")
DEFAULT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")
LORA_DIR = os.environ.get("ACESTEP_LORA_DIR", "/app/loras")
NETWORK_VOLUME_LORA_DIR = os.environ.get("NETWORK_VOLUME_LORA_DIR", "/runpod-volume/loras")

dit_handler = None
llm_handler = None
models_loaded = False
current_lora = None

DEFAULT_STEPS = {
    "acestep-v15-turbo": 8,
    "acestep-v15-sft": 32,
    "acestep-v15-base": 50,
    "acestep-v15-turbo-shift3": 8,
}


def _is_lokr_adapter(lora_path):
    lokr_file = os.path.join(lora_path, "lokr_weights.safetensors")
    return os.path.exists(lokr_file)


def _scan_lora_dir(directory, loras):
    if not os.path.exists(directory):
        return
    for name in os.listdir(directory):
        lora_path = os.path.join(directory, name)
        if os.path.isdir(lora_path):
            config_file = os.path.join(lora_path, "adapter_config.json")
            safetensors = os.path.join(lora_path, "adapter_model.safetensors")
            bin_file = os.path.join(lora_path, "adapter_model.bin")
            lokr_file = os.path.join(lora_path, "lokr_weights.safetensors")
            is_peft = os.path.exists(config_file) and (os.path.exists(safetensors) or os.path.exists(bin_file))
            is_lokr = os.path.exists(lokr_file)
            if is_peft or is_lokr:
                source = "volume" if directory == NETWORK_VOLUME_LORA_DIR else "builtin"
                adapter_type = "lokr" if is_lokr else "peft"
                loras[name] = {"path": lora_path, "source": source, "type": adapter_type}
                print(f"[ACE-Step] Found LoRA: {name} -> {lora_path} ({source}, {adapter_type})", flush=True)


def scan_available_loras():
    loras = {}
    _scan_lora_dir(LORA_DIR, loras)
    _scan_lora_dir(NETWORK_VOLUME_LORA_DIR, loras)
    return loras


HF_LORA_REPO_PREFIX = os.environ.get("HF_LORA_REPO_PREFIX", "ruslanmusinrusmus")


def _sanitize_revision(revision):
    import re
    if not revision:
        return None
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', revision)
    if '..' in sanitized or sanitized.startswith('.'):
        return None
    return sanitized


def download_lora_from_hf(lora_name, revision=None):
    rev_str = f" (revision={revision})" if revision else ""
    safe_rev = _sanitize_revision(revision) if revision else None
    try:
        from huggingface_hub import snapshot_download
        repo_id = f"{HF_LORA_REPO_PREFIX}/{lora_name}"
        dir_name = f"{lora_name}_{safe_rev}" if safe_rev else lora_name
        target_dir = os.path.join(NETWORK_VOLUME_LORA_DIR, dir_name)
        os.makedirs(NETWORK_VOLUME_LORA_DIR, exist_ok=True)
        print(f"[ACE-Step] Downloading LoRA from HuggingFace: {repo_id}{rev_str} -> {target_dir}", flush=True)
        dl_kwargs = {
            "repo_id": repo_id,
            "local_dir": target_dir,
            "ignore_patterns": ["*.md", ".gitattributes"],
        }
        if revision:
            dl_kwargs["revision"] = revision
        snapshot_download(**dl_kwargs)
        config_file = os.path.join(target_dir, "adapter_config.json")
        safetensors = os.path.join(target_dir, "adapter_model.safetensors")
        bin_file = os.path.join(target_dir, "adapter_model.bin")
        lokr_file = os.path.join(target_dir, "lokr_weights.safetensors")
        is_peft = os.path.exists(config_file) and (os.path.exists(safetensors) or os.path.exists(bin_file))
        is_lokr = os.path.exists(lokr_file)
        if is_peft or is_lokr:
            adapter_type = "lokr" if is_lokr else "peft"
            print(f"[ACE-Step] LoRA downloaded successfully: {dir_name} ({adapter_type})", flush=True)
            return dir_name
        print(f"[ACE-Step] Downloaded repo missing adapter files: {dir_name}", flush=True)
        return None
    except Exception as e:
        print(f"[ACE-Step] Failed to download LoRA {lora_name}{rev_str} from HF: {e}", flush=True)
        return None


def validate_lora_compatibility(lora_path):
    import json as _json

    if _is_lokr_adapter(lora_path):
        lokr_file = os.path.join(lora_path, "lokr_weights.safetensors")
        try:
            from safetensors import safe_open
            with safe_open(lokr_file, framework="pt") as f:
                keys = list(f.keys())
                has_lokr_keys = any("lokr_w" in k for k in keys)
                if not has_lokr_keys:
                    return False, "LoKR file has no lokr_w keys"
            print(f"[ACE-Step] LoKR adapter validated: {len(keys)} keys", flush=True)
            return True, "OK (LoKR)"
        except Exception as e:
            return False, f"LoKR validation error: {e}"

    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return False, "adapter_config.json not found"

    with open(config_path) as f:
        config = _json.load(f)

    target_modules = config.get("target_modules", [])
    old_modules = {"to_q", "to_k", "to_v", "to_out.0"}
    new_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}

    if old_modules & set(target_modules):
        return False, f"LoRA has old target_modules {target_modules} (ACE-Step v1 format). Need {list(new_modules)} for v1.5."

    if not (new_modules & set(target_modules)):
        return False, f"LoRA target_modules {target_modules} don't match model attention projections {list(new_modules)}"

    st_path = os.path.join(lora_path, "adapter_model.safetensors")
    if os.path.exists(st_path):
        try:
            from safetensors import safe_open
            with safe_open(st_path, framework="pt") as f:
                keys = f.keys()
                for key in keys:
                    tensor = f.get_tensor(key)
                    shape = list(tensor.shape)
                    if any(d == 2560 for d in shape):
                        return False, f"LoRA tensor {key} has dimension 2560 (ACE-Step v1). Model v1.5 uses hidden_size=2048."
                    break
        except Exception as e:
            print(f"[ACE-Step] Warning checking safetensors: {e}", flush=True)

    return True, "OK"


def _apply_lokr_adapter(handler, lora_path, lora_scale=1.0):
    if handler is None or not hasattr(handler, 'model') or handler.model is None:
        return "❌ Model not initialized — cannot load LoKR adapter"

    lokr_file = os.path.join(lora_path, "lokr_weights.safetensors")
    if not os.path.exists(lokr_file):
        return "❌ lokr_weights.safetensors not found"

    try:
        if hasattr(handler, 'add_lora'):
            result = handler.add_lora(lora_path)
            if not (isinstance(result, str) and result.startswith("❌")):
                return result
            print(f"[ACE-Step] Built-in add_lora failed for LoKR, trying manual LyCORIS load: {result}", flush=True)

        if handler.model is None:
            return "❌ Model became None after add_lora — cannot load LoKR manually"

        from lycoris import create_lycoris_from_weights
        import torch

        decoder = getattr(handler.model, 'decoder', None)
        if decoder is None:
            return "❌ handler.model.decoder is None — model not fully initialized"
        device = next(decoder.parameters()).device
        dtype = next(decoder.parameters()).dtype

        if not hasattr(handler, '_base_decoder'):
            handler._base_decoder = {k: v.clone() for k, v in decoder.state_dict().items()}
            print(f"[ACE-Step] Saved base decoder state ({len(handler._base_decoder)} keys)", flush=True)

        lycoris_net, _ = create_lycoris_from_weights(
            lora_scale, lokr_file, decoder
        )
        lycoris_net.merge_to()

        decoder.to(device).to(dtype)
        decoder.eval()

        handler.lora_loaded = True
        handler.use_lora = True
        handler.lora_scale = lora_scale

        return f"✅ LoKR adapter loaded via LyCORIS (scale={lora_scale})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ LoKR load failed: {str(e)}"


def apply_lora(lora_name, lora_scale=1.0, lora_revision=None):
    global dit_handler, current_lora

    safe_rev = _sanitize_revision(lora_revision) if lora_revision else None
    effective_name = f"{lora_name}_{safe_rev}" if safe_rev else lora_name

    if not lora_name or lora_name == "none":
        if current_lora:
            try:
                dit_handler.unload_lora()
                print(f"[ACE-Step] Unloaded LoRA: {current_lora}", flush=True)
                current_lora = None
            except Exception as e:
                print(f"[ACE-Step] Error unloading LoRA: {e}", flush=True)
        return True

    if current_lora == effective_name:
        print(f"[ACE-Step] LoRA already loaded: {effective_name}", flush=True)
        return True

    available = scan_available_loras()
    if effective_name not in available:
        print(f"[ACE-Step] LoRA not found locally: {effective_name}. Trying HuggingFace download...", flush=True)
        downloaded_name = download_lora_from_hf(lora_name, revision=lora_revision)
        if downloaded_name:
            available = scan_available_loras()
        if effective_name not in available:
            print(f"[ACE-Step] LoRA not found: {effective_name}. Available: {list(available.keys())}", flush=True)
            return False

    lora_info = available[effective_name]
    lora_path = lora_info["path"]

    compatible, reason = validate_lora_compatibility(lora_path)
    if not compatible:
        print(f"[ACE-Step] LoRA '{effective_name}' incompatible: {reason}", flush=True)
        return f"LoRA incompatible with ACE-Step v1.5: {reason}"

    try:
        if current_lora:
            try:
                result = dit_handler.unload_lora()
                print(f"[ACE-Step] Unloaded previous LoRA: {current_lora} -> {result}", flush=True)
            except Exception as ue:
                print(f"[ACE-Step] Warning during unload: {ue}", flush=True)

        adapter_type = lora_info.get("type", "peft")
        print(f"[ACE-Step] Loading LoRA: {effective_name} (scale={lora_scale}, type={adapter_type}) from {lora_path} ({lora_info['source']})", flush=True)

        if adapter_type == "lokr":
            result = _apply_lokr_adapter(dit_handler, lora_path, lora_scale)
            print(f"[ACE-Step] LoKR apply result: {result}", flush=True)
            if isinstance(result, str) and result.startswith("❌"):
                return f"LoKR load error: {result}"
        else:
            result = dit_handler.add_lora(lora_path, adapter_name=effective_name)
            print(f"[ACE-Step] add_lora result: {result}", flush=True)
            if isinstance(result, str) and result.startswith("❌"):
                return f"add_lora error: {result}"

        if lora_scale != 1.0 and hasattr(dit_handler, 'set_lora_scale'):
            dit_handler.set_lora_scale(effective_name, lora_scale)
            print(f"[ACE-Step] Set LoRA scale: {lora_scale}", flush=True)

        current_lora = effective_name
        print(f"[ACE-Step] LoRA loaded successfully: {effective_name}", flush=True)
        return True
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)[:1000]}"
        print(f"[ACE-Step] Error loading LoRA {effective_name}: {err_msg}", flush=True)
        traceback.print_exc()
        current_lora = None
        return err_msg


def ensure_models_loaded():
    global dit_handler, llm_handler, models_loaded
    if models_loaded:
        return True

    _lazy_import_all()

    print(f"[ACE-Step] Loading models (first request)...", flush=True)
    start = time.time()

    try:
        print(f"[ACE-Step] Creating AceStepHandler...", flush=True)
        dit_handler = AceStepHandler()
        print(f"[ACE-Step] AceStepHandler created OK", flush=True)

        print(f"[ACE-Step] Calling initialize_service(project_root={PROJECT_ROOT}, config_path={DEFAULT_MODEL})...", flush=True)
        status, success = dit_handler.initialize_service(
            project_root=PROJECT_ROOT,
            config_path=DEFAULT_MODEL,
            device="cuda",
        )
        print(f"[ACE-Step] initialize_service: success={success}, status={status[:300]}", flush=True)
        if not success:
            print(f"[ACE-Step] initialize_service FAILED: {status}", flush=True)
            dit_handler = None
            return False
        for attr_name in ["model", "vae", "text_tokenizer", "text_encoder"]:
            val = getattr(dit_handler, attr_name, "MISSING")
            is_none = val is None if val != "MISSING" else "MISSING"
            print(f"[ACE-Step] dit_handler.{attr_name}: type={type(val).__name__}, is_none={is_none}", flush=True)
    except Exception as e:
        print(f"[ACE-Step] DiT init ERROR: {e}", flush=True)
        traceback.print_exc()
        dit_handler = None
        return False

    try:
        lm_path = os.path.join(CHECKPOINT_DIR, LM_MODEL)
        if os.path.exists(lm_path):
            print(f"[ACE-Step] Loading LLM from {lm_path}...", flush=True)
            from acestep.llm_inference import LLMHandler
            llm_handler = LLMHandler()
            llm_handler.initialize(
                checkpoint_dir=CHECKPOINT_DIR,
                lm_model_path=LM_MODEL,
                backend="pt",
                device="cuda",
            )
            print(f"[ACE-Step] LLM loaded OK", flush=True)
        else:
            print(f"[ACE-Step] LM model not found at {lm_path}, skipping", flush=True)
    except Exception as e:
        print(f"[ACE-Step] LLM init ERROR (non-fatal): {e}", flush=True)
        traceback.print_exc()
        llm_handler = None

    available_loras = scan_available_loras()
    print(f"[ACE-Step] Available LoRAs: {list(available_loras.keys()) if available_loras else 'none'}", flush=True)

    elapsed = time.time() - start
    print(f"[ACE-Step] Models loaded in {elapsed:.1f}s", flush=True)
    models_loaded = True
    return True


demucs_model = None

def ensure_demucs_loaded():
    global demucs_model
    if demucs_model is not None:
        return True
    try:
        import demucs.pretrained
        print("[ACE-Step] Loading Demucs model (htdemucs)...", flush=True)
        demucs_model = demucs.pretrained.get_model("htdemucs")
        demucs_model.to("cuda")
        demucs_model.eval()
        print("[ACE-Step] Demucs loaded OK", flush=True)
        return True
    except Exception as e:
        print(f"[ACE-Step] Demucs load error: {e}", flush=True)
        traceback.print_exc()
        return False


def separate_vocals(audio_path):
    if not ensure_demucs_loaded():
        raise RuntimeError("Demucs not available")

    import torchaudio
    import demucs.apply

    waveform, sr = torchaudio.load(audio_path)
    if sr != demucs_model.samplerate:
        waveform = torchaudio.functional.resample(waveform, sr, demucs_model.samplerate)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    sources = demucs.apply.apply_model(demucs_model, waveform.unsqueeze(0).to("cuda"), progress=False)
    sources = sources * ref.std() + ref.mean()

    source_names = demucs_model.sources
    vocals_idx = source_names.index("vocals")
    vocals = sources[0, vocals_idx].cpu()

    return vocals, demucs_model.samplerate


def mix_audio(vocals_tensor, vocals_sr, instrumental_path, output_path, vocal_volume=1.0, instrumental_volume=0.85, do_mastering=True):
    import torchaudio

    inst_waveform, inst_sr = torchaudio.load(instrumental_path)
    if inst_sr != vocals_sr:
        inst_waveform = torchaudio.functional.resample(inst_waveform, inst_sr, vocals_sr)

    if inst_waveform.shape[0] == 1:
        inst_waveform = inst_waveform.repeat(2, 1)
    if vocals_tensor.shape[0] == 1:
        vocals_tensor = vocals_tensor.repeat(2, 1)

    min_len = min(vocals_tensor.shape[1], inst_waveform.shape[1])
    vocals_tensor = vocals_tensor[:, :min_len]
    inst_waveform = inst_waveform[:, :min_len]

    mixed = vocals_tensor * vocal_volume + inst_waveform * instrumental_volume
    peak = mixed.abs().max()
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    if do_mastering:
        mixed = master_audio(mixed, vocals_sr)

    fmt = os.path.splitext(output_path)[1].lstrip(".") or "mp3"
    if fmt == "mp3":
        torchaudio.save(output_path, mixed, vocals_sr, format="mp3", compression=-2)
    else:
        torchaudio.save(output_path, mixed, vocals_sr, format=fmt)
    return output_path


def generate_single(job_input, job_id, override_params=None):
    params_dict = {
        "caption": job_input.get("prompt", ""),
        "lyrics": job_input.get("lyrics", ""),
        "duration": float(job_input.get("audio_duration", job_input.get("duration", -1))),
        "task_type": job_input.get("task_type", "text2music"),
        "seed": int(job_input.get("seed", -1)),
        "inference_steps": int(job_input.get("inference_steps", job_input.get("infer_step",
            DEFAULT_STEPS.get(job_input.get("model", DEFAULT_MODEL), 8)))),
        "guidance_scale": float(job_input.get("guidance_scale", 7.0)),
        "thinking": job_input.get("thinking", True) if llm_handler is not None else False,
        "bpm": int(job_input.get("bpm")) if job_input.get("bpm") is not None else None,
        "keyscale": job_input.get("key_scale", job_input.get("keyscale", "")),
        "timesignature": job_input.get("time_signature", job_input.get("timesignature", "")),
        "vocal_language": job_input.get("vocal_language", "unknown"),
        "instrumental": job_input.get("instrumental", False),
    }
    if override_params:
        params_dict.update(override_params)

    params = GenerationParams(**params_dict)
    audio_format = job_input.get("audio_format", "mp3")
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=(params_dict["seed"] < 0),
        seeds=[params_dict["seed"]] if params_dict["seed"] >= 0 else None,
        audio_format=audio_format if audio_format in ("mp3", "wav", "flac") else "mp3",
    )

    save_dir = tempfile.mkdtemp(prefix="ace_step_")
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=save_dir,
    )

    if not result.success:
        return None, result.error or "Generation failed"

    for audio_info in result.audios:
        audio_path = audio_info.get("path", "")
        if audio_path and os.path.exists(audio_path):
            return audio_path, None
        elif "tensor" in audio_info and audio_info["tensor"] is not None:
            import torchaudio
            tensor = audio_info["tensor"]
            sr = audio_info.get("sample_rate", 44100)
            out_path = os.path.join(save_dir, f"output.{audio_format}")
            if audio_format == "mp3":
                torchaudio.save(out_path, tensor.cpu(), sr, format="mp3", compression=-2)
            else:
                torchaudio.save(out_path, tensor.cpu(), sr, format=audio_format)
            return out_path, None

    return None, "No audio generated"


def _reencode_file(filepath, do_mastering, audio_format="mp3", do_enhance=False, enhance_mode="enhance"):
    wav, sr = torchaudio.load(filepath)
    if do_enhance:
        wav, sr = enhance_audio(wav, sr, mode=enhance_mode)
    if do_mastering:
        wav = master_audio(wav, sr)
    audio_bytes = _save_audio_to_bytes(wav, sr, audio_format)
    return base64.b64encode(audio_bytes).decode("utf-8"), sr


def handle_hybrid(job, job_input, model_name, lora_name, lora_scale, audio_format, do_mastering=True, do_enhance=False, enhance_mode="enhance"):
    start = time.time()
    vocal_volume = float(job_input.get("vocal_volume", 1.0))
    instrumental_volume = float(job_input.get("instrumental_volume", 0.85))

    lora_info = f", lora={lora_name}(x{lora_scale})" if lora_name and lora_name != "none" else ""
    print(f"[ACE-Step] HYBRID mode: generating full track + instrumental{lora_info}", flush=True)

    print("[ACE-Step] Step 1/4: Generating full track with vocals...", flush=True)
    full_path, err = generate_single(job_input, job["id"])
    if err:
        return {"error": f"Hybrid step 1 (full track) failed: {err}"}
    print(f"[ACE-Step] Full track generated: {full_path}", flush=True)

    print("[ACE-Step] Step 2/4: Generating clean instrumental...", flush=True)
    inst_path, err = generate_single(job_input, job["id"], override_params={"instrumental": True, "lyrics": ""})
    if err:
        return {"error": f"Hybrid step 2 (instrumental) failed: {err}"}
    print(f"[ACE-Step] Instrumental generated: {inst_path}", flush=True)

    print("[ACE-Step] Step 3/4: Separating vocals with Demucs...", flush=True)
    try:
        vocals_tensor, vocals_sr = separate_vocals(full_path)
        print(f"[ACE-Step] Vocals separated: shape={vocals_tensor.shape}, sr={vocals_sr}", flush=True)
    except Exception as e:
        print(f"[ACE-Step] Vocal separation failed: {e}. Returning full track instead.", flush=True)
        audio_b64, actual_sr = _reencode_file(full_path, do_mastering, audio_format, do_enhance=do_enhance, enhance_mode=enhance_mode)
        gen_time = time.time() - start
        return {
            "audio_base64": audio_b64,
            "content_type": "audio/mpeg",
            "filename": f"ace_step_hybrid_{job['id'][:12]}.mp3",
            "generation_time": round(gen_time, 1),
            "duration": float(job_input.get("audio_duration", job_input.get("duration", -1))),
            "sample_rate": actual_sr,
            "model": model_name,
            "lora": lora_name if lora_name and lora_name != "none" else None,
            "mode": "hybrid",
            "hybrid_status": "fallback_no_demucs",
            "mastered": do_mastering,
            "enhanced": do_enhance,
        }

    print("[ACE-Step] Step 4/4: Mixing vocals + clean instrumental...", flush=True)
    mix_dir = tempfile.mkdtemp(prefix="ace_hybrid_")
    mix_path = os.path.join(mix_dir, f"hybrid_mix.mp3")
    try:
        mix_audio(vocals_tensor, vocals_sr, inst_path, mix_path, vocal_volume, instrumental_volume, do_mastering=do_mastering)
    except Exception as e:
        print(f"[ACE-Step] Mix failed: {e}. Returning full track.", flush=True)
        audio_b64, actual_sr = _reencode_file(full_path, do_mastering, audio_format, do_enhance=do_enhance, enhance_mode=enhance_mode)
        gen_time = time.time() - start
        return {
            "audio_base64": audio_b64,
            "content_type": "audio/mpeg",
            "filename": f"ace_step_hybrid_{job['id'][:12]}.mp3",
            "generation_time": round(gen_time, 1),
            "duration": float(job_input.get("audio_duration", job_input.get("duration", -1))),
            "sample_rate": actual_sr,
            "model": model_name,
            "lora": lora_name if lora_name and lora_name != "none" else None,
            "mode": "hybrid",
            "hybrid_status": "fallback_mix_failed",
            "mastered": do_mastering,
            "enhanced": do_enhance,
        }

    gen_time = time.time() - start
    audio_b64, actual_sr = _reencode_file(mix_path, do_mastering=False, audio_format=audio_format, do_enhance=do_enhance, enhance_mode=enhance_mode)

    print(f"[ACE-Step] HYBRID complete in {gen_time:.1f}s", flush=True)
    return {
        "audio_base64": audio_b64,
        "content_type": "audio/mpeg",
        "filename": f"ace_step_hybrid_{job['id'][:12]}.mp3",
        "generation_time": round(gen_time, 1),
        "duration": float(job_input.get("audio_duration", job_input.get("duration", -1))),
        "sample_rate": actual_sr,
        "model": model_name,
        "lora": lora_name if lora_name and lora_name != "none" else None,
        "mode": "hybrid",
        "hybrid_status": "success",
        "mastered": do_mastering,
        "enhanced": do_enhance,
    }


def handler(job):
    global dit_handler, llm_handler

    try:
        job_input = job["input"]

        if job_input.get("action") == "health":
            return {
                "status": "ok",
                "handler_version": HANDLER_VERSION,
                "libs_loaded": _libs_loaded,
                "models_loaded": models_loaded,
            }

        if job_input.get("action") == "exec_python":
            code = job_input.get("code", "")
            if not code:
                return {"error": "No code provided"}
            import io as _io
            import contextlib
            stdout_buf = _io.StringIO()
            stderr_buf = _io.StringIO()
            local_vars = {
                "dit_handler": dit_handler, "llm_handler": llm_handler,
                "torch": torch, "os": os, "sys": sys,
                "LORA_DIR": LORA_DIR, "NETWORK_VOLUME_LORA_DIR": NETWORK_VOLUME_LORA_DIR,
                "scan_available_loras": scan_available_loras,
            }
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(code, local_vars)
                return {
                    "stdout": stdout_buf.getvalue()[-4000:],
                    "stderr": stderr_buf.getvalue()[-2000:],
                    "success": True,
                }
            except Exception as _e:
                return {
                    "stdout": stdout_buf.getvalue()[-2000:],
                    "stderr": stderr_buf.getvalue()[-2000:],
                    "error": f"{type(_e).__name__}: {str(_e)[:2000]}",
                    "success": False,
                }

        if not ensure_models_loaded():
            return {"error": "Failed to load models. Check worker logs."}

        if job_input.get("action") == "list_loras":
            available = scan_available_loras()
            return {
                "loras": [
                    {"name": name, "source": info["source"], "path": info["path"]}
                    for name, info in available.items()
                ],
                "current_lora": current_lora,
                "lora_dirs": {
                    "builtin": LORA_DIR,
                    "network_volume": NETWORK_VOLUME_LORA_DIR,
                },
            }

        mode = job_input.get("mode", "normal")
        model_name = job_input.get("model", DEFAULT_MODEL)
        audio_format = job_input.get("audio_format", "mp3")
        do_mastering = job_input.get("mastering", True)

        lora_name = job_input.get("lora_name", None)
        lora_scale = float(job_input.get("lora_scale", 1.0))
        lora_revision = job_input.get("lora_revision", None)

        if job_input.get("action") == "diagnose_lora":
            if dit_handler is None or dit_handler.model is None:
                return {"error": "Model not initialized — cannot diagnose LoRA"}
            model_obj = dit_handler.model.decoder
            diag = {
                "handler_version": HANDLER_VERSION,
                "dit_type": str(type(model_obj)),
                "dit_class": model_obj.__class__.__name__,
            }
            diag["lora_methods"] = [m for m in dir(dit_handler) if "lora" in m.lower()]
            for mname in ["load_lora", "add_lora", "unload_lora"]:
                diag[f"has_{mname}"] = hasattr(dit_handler, mname)
            sd = model_obj.state_dict()
            attn_shapes = {}
            for k, v in sd.items():
                if any(proj in k for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    if "weight" in k and "lora" not in k:
                        attn_shapes[k] = list(v.shape)
            sample_keys = sorted(attn_shapes.keys())[:16]
            diag["model_attn_shapes"] = {k: attn_shapes[k] for k in sample_keys}
            diag["model_total_params"] = sum(p.numel() for p in model_obj.parameters())

            available = scan_available_loras()
            diag["available_loras"] = {}
            for n, info in available.items():
                lp = info["path"]
                files = os.listdir(lp) if os.path.isdir(lp) else []
                cfg_path = os.path.join(lp, "adapter_config.json")
                cfg_data = {}
                if os.path.exists(cfg_path):
                    import json as _j
                    with open(cfg_path) as _f:
                        cfg_data = _j.load(_f)
                lora_shapes = {}
                st_path = os.path.join(lp, "adapter_model.safetensors")
                if os.path.exists(st_path):
                    from safetensors import safe_open
                    with safe_open(st_path, framework="pt") as f:
                        for key in sorted(f.keys())[:16]:
                            lora_shapes[key] = list(f.get_tensor(key).shape)
                diag["available_loras"][n] = {
                    "path": lp, "source": info["source"], "files": files,
                    "config": cfg_data, "sample_shapes": lora_shapes,
                }

            test_lora = job_input.get("test_lora", list(available.keys())[0] if available else None)
            if test_lora and test_lora in available:
                test_path = available[test_lora]["path"]
                diag["test_results"] = {}
                try:
                    result = dit_handler.add_lora(test_path, adapter_name="test_diag")
                    diag["test_results"]["add_lora"] = f"Result: {str(result)[:500]}"
                    try:
                        dit_handler.unload_lora()
                    except:
                        pass
                except Exception as _e:
                    diag["test_results"]["add_lora"] = f"FAILED: {str(_e)[:800]}"

                try:
                    from safetensors import safe_open
                    st_path = os.path.join(test_path, "adapter_model.safetensors")
                    mismatches = []
                    with safe_open(st_path, framework="pt") as f:
                        for key in f.keys():
                            clean = key.replace("base_model.model.", "")
                            parts = clean.rsplit(".", 2)
                            module_path = parts[0] if len(parts) > 1 else clean
                            parent_key = module_path + ".weight"
                            lora_shape = list(f.get_tensor(key).shape)
                            if parent_key in sd:
                                model_shape = list(sd[parent_key].shape)
                                if any(ls not in model_shape for ls in lora_shape if ls > 64):
                                    mismatches.append({
                                        "lora_key": key, "lora_shape": lora_shape,
                                        "model_key": parent_key, "model_shape": model_shape,
                                    })
                    diag["dimension_mismatches"] = mismatches[:10]
                    diag["total_mismatches"] = len(mismatches)
                except Exception as _e:
                    diag["dimension_check_error"] = str(_e)[:500]

            return diag

        if lora_name and lora_name != "none":
            lora_result = apply_lora(lora_name, lora_scale, lora_revision=lora_revision)
            if lora_result is not True:
                err_detail = lora_result if isinstance(lora_result, str) else "unknown"
                effective = f"{lora_name}_{lora_revision}" if lora_revision else lora_name
                return {"error": f"Failed to load LoRA: {effective}. Detail: {err_detail}. Available: {list(scan_available_loras().keys())}"}
        elif current_lora and (not lora_name or lora_name == "none"):
            apply_lora(None)

        if mode == "hybrid":
            do_enhance = job_input.get("enhance", False)
            enhance_mode = job_input.get("enhance_mode", "enhance")
            return handle_hybrid(job, job_input, model_name, lora_name, lora_scale, audio_format, do_mastering, do_enhance, enhance_mode)

        prompt = job_input.get("prompt", "")
        lyrics = job_input.get("lyrics", "")
        duration = float(job_input.get("audio_duration", job_input.get("duration", -1)))
        task_type = job_input.get("task_type", "text2music")
        seed = int(job_input.get("seed", -1))
        default_steps = DEFAULT_STEPS.get(model_name, 8)
        inference_steps = int(job_input.get("inference_steps", job_input.get("infer_step", default_steps)))
        guidance_scale = float(job_input.get("guidance_scale", 7.0))
        thinking = job_input.get("thinking", True)
        batch_size = int(job_input.get("batch_size", 1))
        bpm = job_input.get("bpm", None)
        if bpm is not None:
            bpm = int(bpm)
        key_scale = job_input.get("key_scale", job_input.get("keyscale", ""))
        time_signature = job_input.get("time_signature", job_input.get("timesignature", ""))
        vocal_language = job_input.get("vocal_language", "unknown")
        instrumental = job_input.get("instrumental", False)

        lora_info = f", lora={lora_name}(x{lora_scale})" if lora_name and lora_name != "none" else ""
        print(f"[ACE-Step] Job {job['id'][:12]}: model={model_name}, prompt='{prompt[:80]}', "
              f"duration={duration}s, steps={inference_steps}{lora_info}", flush=True)

        params = GenerationParams(
            caption=prompt,
            lyrics=lyrics,
            duration=duration,
            task_type=task_type,
            seed=seed,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            thinking=thinking if llm_handler is not None else False,
            bpm=bpm,
            keyscale=key_scale,
            timesignature=time_signature,
            vocal_language=vocal_language,
            instrumental=instrumental,
        )

        config = GenerationConfig(
            batch_size=batch_size,
            use_random_seed=(seed < 0),
            seeds=[seed] if seed >= 0 else None,
            audio_format=audio_format if audio_format in ("mp3", "wav", "flac") else "mp3",
        )

        save_dir = tempfile.mkdtemp(prefix="ace_step_")

        start = time.time()
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=save_dir,
        )
        gen_time = time.time() - start

        if not result.success:
            return {"error": result.error or "Generation failed", "status_message": result.status_message}

        print(f"[ACE-Step] Done in {gen_time:.1f}s, {len(result.audios)} audio(s)", flush=True)

        do_enhance = job_input.get("enhance", False)
        enhance_mode = job_input.get("enhance_mode", "enhance")

        for i, audio_info in enumerate(result.audios):
            tensor = audio_info.get("tensor", None)
            sr = audio_info.get("sample_rate", 48000)
            if tensor is None:
                audio_path = audio_info.get("path", "")
                if audio_path and os.path.exists(audio_path):
                    tensor, sr = torchaudio.load(audio_path)
            if tensor is not None:
                if do_enhance:
                    tensor, sr = enhance_audio(tensor, sr, mode=enhance_mode)
                if do_mastering:
                    tensor = master_audio(tensor, sr)
                audio_bytes = _save_audio_to_bytes(tensor, sr, audio_format)
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                content_type_map = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
                return {
                    "audio_base64": audio_b64,
                    "content_type": content_type_map.get(audio_format, "audio/mpeg"),
                    "filename": f"ace_step_{job['id'][:12]}_{i}.{audio_format}",
                    "generation_time": round(gen_time, 1),
                    "duration": duration,
                    "sample_rate": sr,
                    "model": model_name,
                    "lora": lora_name if lora_name and lora_name != "none" else None,
                    "mastered": do_mastering,
                    "enhanced": do_enhance,
                }

        return {"error": "No audio files generated"}

    except Exception as e:
        print(f"[ACE-Step] Job error: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()[-2000:]}


print(f"[ACE-Step] ===== CALLING runpod.serverless.start() =====", flush=True)
print(f"[ACE-Step] All heavy imports deferred to first request (lazy loading)", flush=True)
runpod.serverless.start({"handler": handler})
