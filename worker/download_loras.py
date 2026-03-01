import urllib.request
import os
import json

MODULE_RENAME = {
    "to_q": "q_proj",
    "to_k": "k_proj",
    "to_v": "v_proj",
    "to_out.0": "o_proj",
}

loras = {
    "russian-pop-v1": {
        "config_url": "https://huggingface.co/ruslanmusinrusmus/russian-pop-v1/resolve/main/adapter_config.json",
        "weights_url": "https://huggingface.co/ruslanmusinrusmus/russian-pop-v1/resolve/main/adapter_model.safetensors",
    },
    "russian-pop-v2": {
        "config_url": "https://huggingface.co/ruslanmusinrusmus/russian-pop-v2/resolve/main/adapter_config.json",
        "weights_url": "https://huggingface.co/ruslanmusinrusmus/russian-pop-v2/resolve/main/pytorch_lora_weights.safetensors",
    },
}

for lora_name, urls in loras.items():
    lora_dir = f"/app/loras/{lora_name}"
    config_path = f"{lora_dir}/adapter_config.json"
    weights_path = f"{lora_dir}/adapter_model.safetensors"

    os.makedirs(lora_dir, exist_ok=True)

    for url, path in [(urls["config_url"], config_path), (urls["weights_url"], weights_path)]:
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path)
        print(f"  -> {path} ({size} bytes)")
        if size < 100:
            raise Exception(f"Download failed for {url}: file too small ({size} bytes)")

    with open(config_path, "r") as f:
        config = json.load(f)

    old_targets = config.get("target_modules", [])
    new_targets = [MODULE_RENAME.get(t, t) for t in old_targets]
    if new_targets != old_targets:
        print(f"Patching {lora_name} target_modules: {old_targets} -> {new_targets}")
        config["target_modules"] = new_targets
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        print(f"{lora_name} target_modules already correct: {old_targets}")

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
            print(f"Renamed {changes} weight keys in {lora_name}")
            save_file(renamed, weights_path)
            print(f"Saved patched weights")
        else:
            print(f"{lora_name} weight keys already correct")
    except ImportError:
        print("WARNING: safetensors not available, skipping weight key rename")

    print(f"{lora_name} ready!")

print("\nAll LoRA files downloaded and patched successfully!")
