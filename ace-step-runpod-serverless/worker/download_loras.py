import urllib.request
import os

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

print("All LoRA files downloaded successfully!")
