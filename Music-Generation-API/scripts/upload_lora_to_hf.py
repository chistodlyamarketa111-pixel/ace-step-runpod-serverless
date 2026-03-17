import os, sys, json
from huggingface_hub import HfApi, create_repo, login

LORA_DIR = os.environ.get("LORA_DIR", "/workspace/loras/anna-asti-v3")
REPO_ID = os.environ.get("REPO_ID", "ruslanmusinrusmus/anna-asti-v3")
BASE_MODEL_ID = "ACE-Step/ACE-Step-v1-3.5B"

if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]
else:
    HF_TOKEN = os.environ.get("HF_API_TOKEN", "")

if not HF_TOKEN:
    print("Usage: python3 upload_lora_to_hf.py hf_your_token_here")
    sys.exit(1)

HF_TOKEN = HF_TOKEN.strip().encode("ascii", "ignore").decode("ascii")
print(f"Token starts with: {HF_TOKEN[:8]}...")

config_path = os.path.join(LORA_DIR, "adapter_config.json")
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        if config and isinstance(config, dict):
            old_base = config.get("base_model_name_or_path", "")
            if old_base != BASE_MODEL_ID:
                print(f"Fixing base_model_name_or_path:")
                print(f"  OLD: {old_base}")
                print(f"  NEW: {BASE_MODEL_ID}")
                config["base_model_name_or_path"] = BASE_MODEL_ID
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print("  Fixed!")
            else:
                print("base_model already correct")
    except Exception as e:
        print(f"Warning: could not fix adapter_config.json: {e}")

login(token=HF_TOKEN, add_to_git_credential=False)

api = HfApi(token=HF_TOKEN)

print(f"Creating repo: {REPO_ID}")
create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True, private=False)

try:
    api.delete_file("README.md", repo_id=REPO_ID, token=HF_TOKEN)
    print("Deleted old README.md from repo")
except:
    pass

files = [f for f in os.listdir(LORA_DIR) if os.path.isfile(os.path.join(LORA_DIR, f))]
print(f"Files to upload: {files}")

for fname in files:
    fpath = os.path.join(LORA_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"Uploading {fname} ({size_mb:.1f}MB)...")
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fname,
        repo_id=REPO_ID,
        token=HF_TOKEN,
    )
    print(f"  OK: {fname}")

print(f"\nDone! https://huggingface.co/{REPO_ID}")
