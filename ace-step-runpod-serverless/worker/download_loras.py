import urllib.request
import os
import json

MODULE_RENAME = {
    "to_q": "q_proj",
    "to_k": "k_proj",
    "to_v": "v_proj",
    "to_out.0": "o_proj",
}

files = [
    ("https://huggingface.co/ruslanmusinrusmus/russian-pop-v1/resolve/main/adapter_config.json",
     "/app/loras/russian-pop-v1/adapter_config.json"),
    ("https://huggingface.co/ruslanmusinrusmus/russian-pop-v1/resolve/main/adapter_model.safetensors",
     "/app/loras/russian-pop-v1/adapter_model.safetensors"),
]

for url, path in files:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)
    size = os.path.getsize(path)
    print(f"  -> {path} ({size} bytes)")
    if size < 100:
        raise Exception(f"Download failed for {url}: file too small ({size} bytes)")

config_path = "/app/loras/russian-pop-v1/adapter_config.json"
with open(config_path, "r") as f:
    config = json.load(f)

old_targets = config.get("target_modules", [])
new_targets = [MODULE_RENAME.get(t, t) for t in old_targets]
if new_targets != old_targets:
    print(f"Patching adapter_config.json target_modules: {old_targets} -> {new_targets}")
    config["target_modules"] = new_targets
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
else:
    print(f"target_modules already correct: {old_targets}")

weights_path = "/app/loras/russian-pop-v1/adapter_model.safetensors"
try:
    from safetensors.torch import load_file, save_file

    tensors = load_file(weights_path)
    renamed = {}
    changes = 0
    for key, tensor in tensors.items():
        new_key = key
        for old_name, new_name in MODULE_RENAME.items():
            if f".{old_name}." in new_key:
                new_key = new_key.replace(f".{old_name}.", f".{new_name}.")
                changes += 1
        renamed[new_key] = tensor

    if changes > 0:
        print(f"Renamed {changes} weight keys in safetensors")
        for old_k, new_k in zip(list(tensors.keys())[:5], list(renamed.keys())[:5]):
            if old_k != new_k:
                print(f"  {old_k} -> {new_k}")
        save_file(renamed, weights_path)
        print(f"Saved patched weights to {weights_path}")
    else:
        print("Weight keys already correct, no renaming needed")
except ImportError:
    print("WARNING: safetensors not available, skipping weight key rename")
    print("LoRA may fail to load if weight keys don't match model modules")

print("All LoRA files downloaded and patched successfully!")
