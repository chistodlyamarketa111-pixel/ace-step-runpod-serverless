# Adding a New Engine to Music Generation API

Step-by-step guide for adding a new music generation engine via RunPod Serverless.

## Overview

Each engine consists of 3 parts:
1. **Python serverless worker** — runs on RunPod GPU, handles inference
2. **Docker image** — packages the worker + model + dependencies
3. **TypeScript integration** — client, engine class, registry entry, frontend

## Step 1: Build the Base Image (one-time)

```bash
cd scripts/docker
DOCKER_REGISTRY=your-dockerhub-username ./build.sh base
DOCKER_REGISTRY=your-dockerhub-username ./build.sh push-base
```

The base image includes: PyTorch, ffmpeg, Demucs, Matchering, runpod SDK, shared_utils.

## Step 2: Create the Serverless Worker

Copy the template:
```bash
cp worker_template.py my_engine_worker.py
```

Edit `my_engine_worker.py`:
- Load your model at the top (outside the handler function)
- Implement `generate_audio(job_id, params)` — return `(output_file_path, logs)`
- The template already handles pipeline mode (Demucs + mastering) automatically

## Step 3: Create Engine Dockerfile

Create `Dockerfile.my_engine`:
```dockerfile
FROM your-dockerhub-username/music-gen-base:latest

# Clone and install your model
RUN git clone https://github.com/AUTHOR/MyModel.git /workspace/MyModel
WORKDIR /workspace/MyModel
RUN pip install --no-cache-dir -r requirements.txt

# Copy your worker
COPY my_engine_worker.py /workspace/handler.py

ENV MODEL_DIR=/workspace/MyModel

CMD ["python", "/workspace/handler.py"]
```

## Step 4: Build and Push

```bash
DOCKER_REGISTRY=your-dockerhub-username ./build.sh push my_engine
```

Or manually:
```bash
docker build -t your-username/music-gen-my-engine:latest -f Dockerfile.my_engine .
docker push your-username/music-gen-my-engine:latest
```

## Step 5: Create RunPod Serverless Endpoint

1. Go to runpod.io → Serverless → Deploy New Endpoint
2. Click "Import from Docker Registry"
3. Enter: `your-username/music-gen-my-engine:latest`
4. Select GPU (check your model's VRAM requirements)
5. Set Min Workers = 0, Max Workers = 1-3
6. Create endpoint and copy the **Endpoint ID**

## Step 6: Add Secret

Add `MY_ENGINE_ENDPOINT_ID` to your project secrets (in Replit Secrets tab).

## Step 7: TypeScript Integration

### 7a. Create client (`server/my-engine.ts`)

Copy `server/diffrhythm.ts` and modify:
- Change `DIFFRHYTHM_ENDPOINT_ID` → `MY_ENGINE_ENDPOINT_ID`
- Update task ID prefix (`dr_` → your prefix)
- Adjust parameters as needed

### 7b. Create engine class (`server/engines/my-engine.ts`)

Copy `server/engines/diffrhythm.ts` and modify:
- Update engine `id`, `name`, `description`
- Set correct `maxDuration` and `supportedParams`
- Import your client instead of diffrhythm

### 7c. Register in `server/engines/registry.ts`

Add import and registration:
```typescript
import { MyEngine } from "./my-engine";
// In initializeEngines():
registry.register(new MyEngine());
```

## Step 8: Update Frontend

In `client/src/pages/home.tsx`:
- Add engine to selector grid
- Add EngineBadge entry
- Add hero card if desired
- Add to comparison form if applicable

## Testing

1. Restart the application workflow
2. Check that the engine appears in `/api/engines`
3. Check health: `/api/health`
4. Submit a test generation via the Playground tab

## File Structure

```
scripts/docker/
├── Dockerfile.base          # Base image (PyTorch + shared deps)
├── Dockerfile.diffrhythm    # DiffRhythm engine
├── Dockerfile.example       # Template for new engines
├── shared_utils.py          # Shared code (Demucs, mastering, etc.)
├── worker_template.py       # Template serverless worker
├── build.sh                 # Build helper script
└── ADDING_NEW_ENGINE.md     # This file

server/
├── diffrhythm.ts            # TypeScript client (example)
├── engines/
│   ├── base.ts              # MusicEngine interface
│   ├── registry.ts          # Engine registry
│   ├── diffrhythm.ts        # Engine class (example)
│   └── diffrhythm-pp.ts     # Pipeline engine class (example)
```
