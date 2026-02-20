# ACE-Step v1.5 — RunPod Serverless Worker

AI music generation serverless worker for [RunPod](https://runpod.io) using the [ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5) model.

## Features

- **4 DiT models**: turbo (8 steps), sft (32 steps, best quality), base (50 steps), turbo-shift3 (8 steps)
- **Text-to-music generation** with lyrics support
- **Chain-of-Thought** (CoT) via built-in 1.7B LLM
- **Batch generation** support
- **Multiple audio formats**: mp3, wav, flac
- **CPU offload** for long audio generation (150s+)

## Deployment to RunPod Serverless

### 1. Push this repo to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ace-step-runpod-serverless.git
git push -u origin main
```

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:

| Setting | Value |
|---------|-------|
| **Endpoint name** | `ace-step-v15` |
| **GPU** | 48 GB (recommended) |
| **Max workers** | 2 |
| **Active workers** | 0 |
| **GPU count** | 1 |
| **Idle timeout** | 1 sec |
| **Execution timeout** | 600 sec |
| **FlashBoot** | Enabled |

4. Under **Repository configuration**:
   - **Branch**: `main`
   - **Dockerfile Path**: `Dockerfile`
   - **Build Context**: `worker`

5. Click **"Save Endpoint"**

### 3. Get your Endpoint ID

After creation, copy the Endpoint ID from the endpoint page. It looks like: `abc123def456`

### 4. Test the endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "upbeat electronic dance music",
      "lyrics": "[verse]\nFeel the rhythm\n[chorus]\nDance all night",
      "duration": 60,
      "model": "acestep-v15-turbo",
      "inference_steps": 8,
      "audio_format": "mp3"
    }
  }'
```

Check status:
```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

## API Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "" | Music description |
| `lyrics` | string | "" | Song lyrics with [verse], [chorus] tags |
| `duration` | number | -1 | Audio duration in seconds (-1 = auto) |
| `model` | string | "acestep-v15-turbo" | DiT model name |
| `inference_steps` | number | auto | Number of inference steps |
| `guidance_scale` | number | 7.0 | Classifier-free guidance scale |
| `task_type` | string | "text2music" | Task type: text2music, cover, repaint |
| `audio_format` | string | "mp3" | Output format: mp3, wav, flac |
| `seed` | number | -1 | Random seed (-1 = random) |
| `thinking` | boolean | true | Enable Chain-of-Thought reasoning |
| `batch_size` | number | 1 | Number of audio samples to generate |
| `bpm` | number | null | Beats per minute |
| `key_scale` | string | "" | Musical key (e.g., "C major") |
| `time_signature` | string | "" | Time signature (e.g., "4/4") |
| `vocal_language` | string | "unknown" | Vocal language |
| `instrumental` | boolean | false | Generate instrumental only |
| `lm_temperature` | number | 0.85 | LLM temperature |
| `lm_cfg_scale` | number | 2.0 | LLM CFG scale |

## API Output

```json
{
  "audio_base64": "...",
  "content_type": "audio/mpeg",
  "filename": "ace_step_abc123_0.mp3",
  "generation_time": 45.2,
  "duration": 60,
  "sample_rate": 44100,
  "model": "acestep-v15-turbo"
}
```

## Environment Variables (Optional)

| Variable | Description |
|----------|-------------|
| `ACESTEP_DIT_MODEL` | Default DiT model (default: acestep-v15-turbo) |
| `ACESTEP_CPU_OFFLOAD` | Enable CPU offload for long audio (default: false) |

## Build Time

The Docker image takes ~15-25 minutes to build on RunPod due to model downloads (~15GB total).

## License

ACE-Step v1.5 model is licensed under Apache 2.0 / MIT.
