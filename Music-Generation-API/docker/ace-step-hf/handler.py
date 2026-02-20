"""
ACE-Step v1.5 — Custom Handler for Hugging Face Inference Endpoints.

EndpointHandler class is auto-discovered by HF.
Model repo is mounted at /repository (passed as `path` to __init__).
Additional model variants are downloaded at init from HuggingFace Hub.
"""

import base64
import gc
import glob
import os
import time
import traceback

import torch


VALID_MODELS = {
    "acestep-v15-turbo",
    "acestep-v15-sft",
    "acestep-v15-base",
    "acestep-v15-turbo-shift3",
}

DEFAULT_STEPS = {
    "acestep-v15-turbo": 8,
    "acestep-v15-sft": 32,
    "acestep-v15-base": 50,
    "acestep-v15-turbo-shift3": 8,
}

MODEL_REPOS = {
    "acestep-v15-sft": "ACE-Step/acestep-v15-sft",
    "acestep-v15-base": "ACE-Step/acestep-v15-base",
    "acestep-v15-turbo-shift3": "ACE-Step/acestep-v15-turbo-shift3",
}


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.checkpoint_dir = path if path else "/repository"
        self.cpu_offload = os.environ.get("ACESTEP_CPU_OFFLOAD", "true").lower() == "true"
        self.default_model = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")
        self.pipeline = None
        self.current_model = None

        print(f"[ACE-Step] Checkpoint dir: {self.checkpoint_dir}")
        print(f"[ACE-Step] CPU offload: {self.cpu_offload}")
        print(f"[ACE-Step] Default model: {self.default_model}")

        self._download_extra_models()

        available = [m for m in VALID_MODELS
                     if os.path.exists(os.path.join(self.checkpoint_dir, m))]
        print(f"[ACE-Step] Available models: {available}")

        print(f"[ACE-Step] Pre-loading model: {self.default_model}...")
        try:
            self._get_pipeline(self.default_model)
            print("[ACE-Step] Pipeline ready")
        except Exception as e:
            print(f"[ACE-Step] Warning: failed to preload: {e}")
            traceback.print_exc()

    def _download_extra_models(self):
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("[ACE-Step] huggingface_hub not available, skipping extra model download")
            return

        for model_name, repo in MODEL_REPOS.items():
            model_dir = os.path.join(self.checkpoint_dir, model_name)
            if os.path.exists(model_dir) and any(
                f.endswith((".pt", ".safetensors", ".bin"))
                for f in os.listdir(model_dir)
            ):
                print(f"[ACE-Step] Model {model_name} already at {model_dir}")
                continue
            print(f"[ACE-Step] Downloading {model_name} from {repo}...")
            try:
                snapshot_download(repo, local_dir=model_dir)
                print(f"[ACE-Step] Downloaded {model_name}")
            except Exception as e:
                print(f"[ACE-Step] Failed to download {model_name}: {e}")

    def _get_pipeline(self, model_name: str):
        if self.pipeline is not None and self.current_model == model_name:
            return self.pipeline

        from acestep.pipeline_ace_step import ACEStepPipeline

        model_path = os.path.join(self.checkpoint_dir, model_name)
        if not os.path.exists(model_path):
            raise ValueError(f"Model '{model_name}' not found at {model_path}")

        if self.pipeline is None:
            print(f"[ACE-Step] Creating pipeline with checkpoint_dir={self.checkpoint_dir}")
            start = time.time()
            self.pipeline = ACEStepPipeline(
                checkpoint_dir=self.checkpoint_dir,
                cpu_offload=self.cpu_offload,
            )
            print(f"[ACE-Step] Pipeline created in {time.time() - start:.1f}s")

        if self.current_model != model_name:
            print(f"[ACE-Step] Loading checkpoint: {model_name}")
            start = time.time()
            self.pipeline.load_checkpoint(checkpoint_dir=model_path)
            print(f"[ACE-Step] Loaded in {time.time() - start:.1f}s")
            self.current_model = model_name

        return self.pipeline

    def __call__(self, data: dict) -> dict:
        inputs = data.get("inputs", {})
        params = data.get("parameters", {})

        if isinstance(inputs, str):
            inputs = {"prompt": inputs}

        prompt = inputs.get("prompt", params.get("prompt", ""))
        lyrics = inputs.get("lyrics", params.get("lyrics", ""))
        model_name = inputs.get("model", params.get("model", self.default_model))
        duration = float(inputs.get("duration", params.get("duration", 30)))
        audio_format = inputs.get("audio_format", params.get("audio_format", "wav"))
        seed = int(inputs.get("seed", params.get("seed", -1)))
        infer_step = int(inputs.get("inference_steps", params.get("inference_steps",
                         DEFAULT_STEPS.get(model_name, 8))))
        guidance_scale = float(inputs.get("guidance_scale", params.get("guidance_scale", 15.0)))
        task = inputs.get("task_type", params.get("task_type", "text2music"))
        batch_size = int(inputs.get("batch_size", params.get("batch_size", 1)))

        if audio_format not in ("wav", "mp3", "flac"):
            audio_format = "wav"

        if model_name not in VALID_MODELS:
            return {"error": f"Invalid model '{model_name}'. Available: {list(VALID_MODELS)}"}

        try:
            pipe = self._get_pipeline(model_name)
        except Exception as e:
            return {"error": f"Failed to load model: {e}"}

        if not pipe.loaded:
            return {"error": "Pipeline not loaded"}

        manual_seeds = [seed] * batch_size if seed >= 0 else None

        save_dir = f"/tmp/ace_output/{int(time.time() * 1000)}"
        os.makedirs(save_dir, exist_ok=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[ACE-Step] Generate: model={model_name}, prompt='{prompt[:80]}', "
              f"duration={duration}s, steps={infer_step}")

        start = time.time()
        try:
            result = pipe(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=duration,
                infer_step=infer_step,
                guidance_scale=guidance_scale,
                task=task,
                format=audio_format,
                manual_seeds=manual_seeds,
                batch_size=batch_size,
                save_path=save_dir,
            )
        except TypeError:
            result = pipe(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=duration,
                infer_step=infer_step,
                guidance_scale=guidance_scale,
                task=task,
                manual_seeds=manual_seeds,
                batch_size=batch_size,
                save_path=save_dir,
            )
        gen_time = time.time() - start
        print(f"[ACE-Step] Done in {gen_time:.1f}s")

        audio_b64 = None
        audio_files = glob.glob(os.path.join(save_dir, f"*.{audio_format}"))
        if not audio_files:
            audio_files = (glob.glob(os.path.join(save_dir, "*.wav")) +
                           glob.glob(os.path.join(save_dir, "*.mp3")) +
                           glob.glob(os.path.join(save_dir, "*.flac")))
        if not audio_files:
            audio_files = [f for f in glob.glob(os.path.join(save_dir, "**/*"), recursive=True)
                           if os.path.isfile(f) and os.path.getsize(f) > 1000]

        if audio_files:
            found = audio_files[0]
            with open(found, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = found.rsplit(".", 1)[-1] if "." in found else audio_format
            if ext in ("wav", "mp3", "flac"):
                audio_format = ext

        if audio_b64 is None:
            return {"error": "No audio produced"}

        return {
            "audio_base64": audio_b64,
            "audio_format": audio_format,
            "model": model_name,
            "generation_time": round(gen_time, 1),
            "duration": duration,
            "inference_steps": infer_step,
            "seed": seed,
        }
