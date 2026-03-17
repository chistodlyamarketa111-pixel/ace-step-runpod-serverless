#!/usr/bin/env python3
"""Initialize RunPod worker: load model, apply LoRA, patch bugs."""
import json, os, urllib.request, time

ENDPOINT_ID = os.environ.get("ACESTEP_ENDPOINT_ID", "u07qbeocmy4479")
API_KEY = os.environ["RUNPOD_API_KEY"]
HF_TOKEN = os.environ.get("HF_API_TOKEN", "")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"


def exec_python(code, timeout=120):
    payload = {"input": {"action": "exec_python", "code": code}}
    req = urllib.request.Request(
        f"{BASE_URL}/run",
        data=json.dumps(payload).encode(),
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        d = json.loads(resp.read())
    job_id = d["id"]

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(5)
        try:
            req2 = urllib.request.Request(f"{BASE_URL}/status/{job_id}")
            req2.add_header("Authorization", f"Bearer {API_KEY}")
            with urllib.request.urlopen(req2, timeout=15) as resp2:
                st = json.loads(resp2.read())
        except Exception:
            continue
        if st.get("status") in ("COMPLETED", "FAILED"):
            o = st.get("output", {})
            return o.get("stdout", ""), o.get("stderr", ""), st.get("status")
    return "", "", "TIMEOUT"


def main():
    print("Step 1: Initialize model via initialize_service...")
    stdout, stderr, status = exec_python('''
import os, time
t0 = time.time()
checkpoint = os.environ.get("ACESTEP_CHECKPOINT_DIR", "/app/checkpoints")
status, success = dit_handler.initialize_service(
    project_root=checkpoint,
    config_path="acestep-v15-turbo-shift3",
    device="cuda",
)
print(f"initialize_service: success={success} ({time.time()-t0:.1f}s)")
print(f"model: {type(dit_handler.model).__name__ if dit_handler.model else None}")
''', timeout=120)
    print(f"  {status}: {stdout.strip()}")
    if status == "FAILED":
        print(f"  stderr: {stderr[-300:]}")
        return False

    print("\nStep 2: Apply LoRA adapter...")
    stdout, stderr, status = exec_python('''
import time
t0 = time.time()
from lycoris import create_lycoris_from_weights
from huggingface_hub import hf_hub_download
import torch as th

lora_path = hf_hub_download("ruslanmusinrusmus/russianrap-v2", "lokr_weights.safetensors", revision="epoch_10")
decoder = dit_handler.model.decoder
device = next(decoder.parameters()).device
dtype = next(decoder.parameters()).dtype

if not hasattr(dit_handler, "_base_decoder"):
    dit_handler._base_decoder = {k: v.clone() for k, v in decoder.state_dict().items()}

lycoris_net, _ = create_lycoris_from_weights(0.7, lora_path, decoder)
lycoris_net.merge_to()
decoder.to(device).to(dtype).eval()
print(f"LoRA applied! ({time.time()-t0:.1f}s)")
''', timeout=60)
    print(f"  {status}: {stdout.strip()}")

    print("\nStep 3: Patch conditioning_batch.py for None target_wavs...")
    stdout, stderr, status = exec_python('''
import time, importlib
t0 = time.time()

path = "/app/ace-step/acestep/core/generation/handler/conditioning_batch.py"
with open(path, "r") as f:
    src = f.read()

old_block = """        target_wavs, target_latents, latent_masks, max_latent_length, silence_latent_tiled = (
            self._prepare_target_latents_and_wavs(batch_size, target_wavs, audio_code_hints)
        )
        wav_lengths = torch.tensor([target_wavs.shape[-1]] * batch_size, dtype=torch.long)"""

new_block = """        if target_wavs is not None:
            target_wavs, target_latents, latent_masks, max_latent_length, silence_latent_tiled = (
                self._prepare_target_latents_and_wavs(batch_size, target_wavs, audio_code_hints)
            )
            wav_lengths = torch.tensor([target_wavs.shape[-1]] * batch_size, dtype=torch.long)
        else:
            self._ensure_silence_latent_on_device()
            silence_latent_tiled = self.silence_latent
            import re as _re
            _dur = 30
            if parsed_metas:
                _m = parsed_metas[0]
                _match = _re.search(r'duration:\\\\s*(\\\\d+)', _m)
                if _match:
                    _dur = int(_match.group(1))
            _sr = 44100
            _hop = 2048
            _n_lat = int(_dur * _sr / _hop) + 1
            _c = silence_latent_tiled.shape[-1] if silence_latent_tiled.dim() >= 2 else 64
            target_latents = torch.zeros(batch_size, _n_lat, _c, device=self.device, dtype=silence_latent_tiled.dtype)
            latent_masks = torch.ones(batch_size, _n_lat, device=self.device)
            max_latent_length = _n_lat
            silence_latent_tiled = target_latents.clone()
            target_wavs = torch.zeros(batch_size, 1, int(_dur * _sr), device=self.device)
            wav_lengths = torch.tensor([target_wavs.shape[-1]] * batch_size, dtype=torch.long)"""

if old_block in src:
    src = src.replace(old_block, new_block)
    with open(path, "w") as f:
        f.write(src)
    print("conditioning_batch.py patched!")
elif "target_wavs is not None" in src:
    print("Already patched!")
else:
    print("ERROR: Pattern not found!")

from acestep.core.generation.handler import conditioning_batch
importlib.reload(conditioning_batch)
cls = type(dit_handler)
cls._prepare_batch = conditioning_batch.ConditioningBatchMixin._prepare_batch

if "_prepare_target_latents_and_wavs" in dit_handler.__dict__:
    del dit_handler.__dict__["_prepare_target_latents_and_wavs"]
    print("Removed instance-level override")

print(f"Done ({time.time()-t0:.1f}s)")
''', timeout=60)
    print(f"  {status}: {stdout.strip()}")

    print("\nStep 4: Test generation...")
    stdout, stderr, status = exec_python('''
import time
t0 = time.time()
result = dit_handler.generate_music(
    captions="Russian rap test, trap beat, BPM 130",
    lyrics="[Chorus]\\nТестовый трек для проверки",
    audio_duration=10,
    inference_steps=10,
    seed=42,
    use_random_seed=False,
)
gen_time = time.time() - t0
success = result.get("success", False)
n_audios = len(result.get("audios", []))
print(f"Test: success={success}, audios={n_audios}, time={gen_time:.1f}s")
if not success:
    print(f"Error: {result.get('error', 'unknown')}")
''', timeout=120)
    print(f"  {status}: {stdout.strip()}")

    if "success=True" in stdout:
        print("\n✓ Worker initialized and ready for batch generation!")
        return True
    else:
        print("\n✗ Worker initialization failed")
        return False


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
