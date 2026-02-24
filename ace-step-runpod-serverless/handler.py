#!/usr/bin/env python3
"""
ACE-Step v1.5 — RunPod Serverless Handler
DIAGNOSTIC VERSION: Does NOT load models at startup.
Just starts the worker and reports environment info.
"""

import os
import sys
import traceback

print("=" * 60, flush=True)
print("[DIAG] Handler starting — DIAGNOSTIC MODE", flush=True)
print(f"[DIAG] Python: {sys.version}", flush=True)
print(f"[DIAG] CWD: {os.getcwd()}", flush=True)
print(f"[DIAG] ENV ACESTEP_CHECKPOINT_DIR: {os.environ.get('ACESTEP_CHECKPOINT_DIR', 'NOT SET')}", flush=True)
print(f"[DIAG] ENV HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}", flush=True)
print("=" * 60, flush=True)

def list_dir_safe(path, depth=0, max_depth=2):
    prefix = "  " * depth
    if not os.path.exists(path):
        print(f"{prefix}{path} — DOES NOT EXIST", flush=True)
        return
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{prefix}{os.path.basename(path)} ({size_mb:.1f} MB)", flush=True)
        return
    try:
        entries = sorted(os.listdir(path))
        print(f"{prefix}{path}/ ({len(entries)} items)", flush=True)
        if depth < max_depth:
            for e in entries[:30]:
                full = os.path.join(path, e)
                if os.path.isdir(full):
                    list_dir_safe(full, depth + 1, max_depth)
                else:
                    size_mb = os.path.getsize(full) / (1024 * 1024)
                    print(f"{prefix}  {e} ({size_mb:.1f} MB)", flush=True)
            if len(entries) > 30:
                print(f"{prefix}  ... and {len(entries) - 30} more", flush=True)
    except Exception as e:
        print(f"{prefix}{path} — ERROR: {e}", flush=True)

print("[DIAG] === /app contents ===", flush=True)
list_dir_safe("/app", max_depth=1)

print("[DIAG] === /app/checkpoints contents ===", flush=True)
list_dir_safe("/app/checkpoints", max_depth=2)

print("[DIAG] === /app/ace-step contents ===", flush=True)
list_dir_safe("/app/ace-step", max_depth=1)

print("[DIAG] === Checking imports ===", flush=True)

try:
    import runpod
    print(f"[DIAG] runpod OK: {runpod.__version__}", flush=True)
except Exception as e:
    print(f"[DIAG] runpod FAIL: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    print(f"[DIAG] torch OK: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[DIAG] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        mem = torch.cuda.get_device_properties(0).total_mem
        print(f"[DIAG] VRAM: {mem / 1e9:.1f} GB", flush=True)
except Exception as e:
    print(f"[DIAG] torch FAIL: {e}", flush=True)

try:
    import acestep
    print(f"[DIAG] acestep OK from: {acestep.__file__}", flush=True)
except Exception as e:
    print(f"[DIAG] acestep FAIL: {e}", flush=True)
    traceback.print_exc()

try:
    from acestep.handler import AceStepHandler
    print(f"[DIAG] AceStepHandler import OK", flush=True)
except Exception as e:
    print(f"[DIAG] AceStepHandler import FAIL: {e}", flush=True)
    traceback.print_exc()

try:
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    print(f"[DIAG] inference imports OK", flush=True)
except Exception as e:
    print(f"[DIAG] inference imports FAIL: {e}", flush=True)
    traceback.print_exc()

try:
    from acestep.pipeline_ace_step import ACEStepPipeline
    print(f"[DIAG] ACEStepPipeline import OK", flush=True)
except Exception as e:
    print(f"[DIAG] ACEStepPipeline import FAIL: {e}", flush=True)
    traceback.print_exc()

print("=" * 60, flush=True)
print("[DIAG] All checks done. Starting worker (no model loaded).", flush=True)
print("=" * 60, flush=True)


def handler(job):
    return {
        "status": "diagnostic_mode",
        "message": "Handler is alive. Models NOT loaded. Check system logs for diagnostics.",
        "python": sys.version,
        "cuda": str(getattr(torch, 'cuda', None) and torch.cuda.is_available()),
    }


runpod.serverless.start({"handler": handler})
