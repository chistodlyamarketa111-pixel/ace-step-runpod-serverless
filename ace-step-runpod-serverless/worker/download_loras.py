import urllib.request
import os
import ssl

ctx = ssl.create_default_context()

files = [
    ("https://huggingface.co/ACE-Step/chinese-new-year/resolve/main/adapter_config.json",
     "/app/loras/chinese-new-year/adapter_config.json"),
    ("https://huggingface.co/ACE-Step/chinese-new-year/resolve/main/adapter_model.safetensors",
     "/app/loras/chinese-new-year/adapter_model.safetensors"),
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
        with open(path, "r") as f:
            print(f"  WARNING: File too small! Content: {f.read()[:200]}")
        raise Exception(f"Download failed for {url}: file too small ({size} bytes)")

print("All LoRA files downloaded successfully!")
