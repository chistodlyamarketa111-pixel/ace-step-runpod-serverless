# Music Generation API

## Overview

This is a web-based music generation API service that accepts music generation parameters and processes them through multiple AI music generation engines deployed on RunPod GPU Pods. The application provides a React frontend playground for testing, API documentation via Swagger, job history tracking, and A/B comparison capabilities between generated music and Suno (a commercial music generation service), with analysis powered by Google Gemini AI.

The main application lives in the `Music-Generation-API/` directory. The root-level `package.json` contains some shared dependencies but the primary project is inside `Music-Generation-API/`.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: React 18 with TypeScript, built with Vite
- **Styling**: Tailwind CSS with CSS variables for theming (light/dark mode support)
- **Component Library**: shadcn/ui (New York style) — extensive set of Radix UI-based components in `client/src/components/ui/`
- **Routing**: Wouter (lightweight client-side router)
- **State Management**: TanStack React Query for server state
- **Forms**: React Hook Form with Zod validation via `@hookform/resolvers`
- **Entry point**: `client/src/main.tsx` → `client/src/App.tsx`
- **Main page**: `client/src/pages/home.tsx` — contains engine selector, playground, API docs, job history, compare, and admin tabs

### Backend
- **Framework**: Express.js with TypeScript, run via `tsx`
- **Entry point**: `server/index.ts`
- **API Routes**: `server/routes.ts` — RESTful endpoints for generation, job management, comparisons, and admin
- **Authentication**: Bearer token auth middleware (`server/middleware/auth.ts`) — single shared token from `API_BEARER_TOKEN` env var
- **API Documentation**: Swagger UI served at `/docs` from `server/openapi.yaml`
- **Build**: Custom build script (`script/build.ts`) using esbuild for server + Vite for client, outputs to `dist/`

### Engine Architecture (Plugin Pattern)
The system uses a **registry-based engine pattern** for extensibility:
- **Base interface**: `server/engines/base.ts` — defines `MusicEngine` interface with methods: `getInfo()`, `isConfigured()`, `checkHealth()`, `submitTask()`, `queryTaskStatus()`, `fetchAudio()`
- **Registry**: `server/engines/registry.ts` — central `EngineRegistry` that manages all engine instances
- **Current engines**:
  - **ACE-Step v1.5** (`server/engines/ace-step.ts` → delegates to `server/runpod.ts`): Instrumental/vocal generation with DiT controls, up to 600s duration
  - **HeartMuLa** (`server/engines/heartmula.ts` → delegates to `server/heartmula.ts`): Full song generation with vocals & lyrics, up to 300s
  - **YuE** (`server/engines/yue.ts` → delegates to `server/yue.ts`): Lyrics-to-song generation using open-source YuE model on RunPod RTX 4090
  - **YuE + Post-Processing** (`server/engines/yue-pp.ts`): Chains YuE generation with post-processing (RVC voice conversion + ffmpeg mastering: loudness normalization, EQ, compression, limiting). Uses state machine to track generation→postprocessing phases
  - **DiffRhythm** (`server/engines/diffrhythm.ts` → delegates to `server/diffrhythm.ts`): Blazingly fast full-song generation using latent diffusion (4:45 in ~30s). Apache 2.0 license, commercial-ready
  - **DiffRhythm + Pipeline** (`server/engines/diffrhythm-pp.ts`): Modular pipeline: DiffRhythm → Demucs stem separation → stem remixing → Matchering mastering. Higher quality through specialized processing at each stage
- To add a new engine: create a class implementing `MusicEngine`, register it in `initializeEngines()`
- **YuE Post-processing pipeline**: `/postprocess` endpoint on YuE pod accepts completed job ID, optionally applies RVC voice conversion, then runs mastering chain via ffmpeg. Install script: `scripts/install_rvc_runpod.sh`
- **DiffRhythm Pipeline**: API server (`scripts/diffrhythm_api_server.py`) runs on RunPod pod with DiffRhythm + Demucs + Matchering installed. Endpoints: `/generate`, `/pipeline`, `/demucs`, `/master`, `/status/:id`, `/download/:path`

### Database
- **Database**: PostgreSQL
- **ORM**: Drizzle ORM with `drizzle-zod` for schema-to-validation integration
- **Schema**: `shared/schema.ts` — defines `jobs`, `users`, and `comparisons` tables
- **Connection**: `server/db.ts` using `pg.Pool` with `DATABASE_URL` env var
- **Migrations**: Drizzle Kit with config in `drizzle.config.ts`, migrations output to `./migrations`
- **Storage layer**: `server/storage.ts` — `DatabaseStorage` class implementing `IStorage` interface

### Key Database Tables
- **jobs**: Tracks music generation jobs (id, engine, status, prompt, lyrics, duration, style, progress, output URL, etc.)
- **comparisons**: A/B or 3-way comparison records between our engines and Suno (supports optional post-processing track via `ourPpJobId`, `ourPpStatus`, `ourPpAudioUrl`, `enablePP` fields)
- **users**: Basic user table
- **conversations/messages**: Chat integration tables (from Replit AI integrations)

### API Design
All endpoints are prefixed with `/api/`. Key routes:
- `GET /api/health` — Health check for API and all engine pods
- `GET /api/engines` — List available engines and their capabilities
- `POST /api/generate` — Submit a music generation job (requires Bearer auth)
- `GET /api/jobs` / `GET /api/jobs/:id` — List/get job status (polls engine APIs)
- `GET /api/jobs/:id/audio` — Download generated audio (proxied from engine pod)
- `POST /api/compare` — Start A/B or 3-way comparison (our engine vs Suno, optionally with post-processing, analyzed by Gemini)
- `GET /api/comparisons` / `GET /api/comparisons/:id` — Comparison results (supports `enablePP` flag for 3-way)
- `POST /api/generate-song-idea` — Gemini generates random song ideas

## External Dependencies

### Docker Infrastructure
- **Base image** (`scripts/docker/Dockerfile.base`): Shared dependencies (PyTorch, ffmpeg, Demucs, Matchering, runpod SDK, shared_utils.py)
- **Shared utilities** (`scripts/docker/shared_utils.py`): Reusable functions for Demucs, mastering, stem remixing, base64 conversion
- **Worker template** (`scripts/docker/worker_template.py`): Template for creating new engine serverless workers
- **Build script** (`scripts/docker/build.sh`): Helper for building and pushing Docker images
- **Adding new engines**: See `scripts/docker/ADDING_NEW_ENGINE.md` for step-by-step guide

### RunPod GPU Pods
- Music generation models run on persistent RunPod pods accessed via proxy URLs (`https://{POD_ID}-8000.proxy.runpod.net`)
- **ACE-Step Pod** (`RUNPOD_POD_ID` env var): Docker image `valyriantech/ace-step-1.5:latest`
- **HeartMuLa Pod** (`HEARTMULA_POD_ID` env var): Docker image `ambsd/heartmula-studio:latest`
- **YuE Pod** (`YUE_POD_ID` env var): Custom Python API server (`scripts/yue_api_server.py`) running YuE model with post-processing support
- **DiffRhythm Serverless** (`DIFFRHYTHM_ENDPOINT_ID` env var): RunPod Serverless endpoint running `scripts/diffrhythm_serverless_worker.py`. GPU spins up only on request (min workers = 0), auto-scales. Needs 6-8 GB VRAM (FP16). Audio returned as base64 in job output.
- Deployment scripts in `scripts/deploy-runpod.ts` and `scripts/deploy-heartmula.ts` use RunPod REST API (`RUNPOD_API_KEY`)

### Suno API
- Commercial music generation service accessed via `https://api.sunoapi.org`
- Used as benchmark in A/B comparisons
- Requires `SUNO_API_KEY` env var
- Client implementation: `server/suno.ts`

### Google Gemini AI
- Used for A/B audio comparison analysis and song idea generation
- Configured via `AI_INTEGRATIONS_GEMINI_API_KEY` and `AI_INTEGRATIONS_GEMINI_BASE_URL` env vars
- Client implementation: `server/gemini.ts`
- Also available via Replit AI Integrations (`server/replit_integrations/`)

### Replit AI Integrations
- Pre-configured modules in `server/replit_integrations/` for chat, batch processing, and image generation
- Uses Gemini models through Replit's proxy service
- Chat storage uses the same PostgreSQL database

### Environment Variables Required
- `DATABASE_URL` — PostgreSQL connection string
- `RUNPOD_POD_ID` — ACE-Step pod identifier
- `HEARTMULA_POD_ID` — HeartMuLa pod identifier
- `YUE_POD_ID` — YuE pod identifier
- `DIFFRHYTHM_ENDPOINT_ID` — RunPod Serverless endpoint ID for DiffRhythm (GPU starts only on request)
- `RUNPOD_API_KEY` — RunPod API key for pod management and serverless endpoints
- `API_BEARER_TOKEN` — Bearer token for API authentication
- `SUNO_API_KEY` — Suno API key for comparisons
- `AI_INTEGRATIONS_GEMINI_API_KEY` — Gemini API key
- `AI_INTEGRATIONS_GEMINI_BASE_URL` — Gemini API base URL

## Recent Changes
- **2026-02-17**: Created Docker infrastructure for fast engine onboarding — base image, shared utilities (Demucs, mastering, remix), worker template, build script, step-by-step guide (`scripts/docker/ADDING_NEW_ENGINE.md`). New engines can be added by copying the template, implementing `generate_audio()`, building a Docker image, and registering in the TypeScript codebase.
- **2026-02-16**: Migrated DiffRhythm to RunPod Serverless — GPU starts only on request, no idle costs. Created serverless worker (`scripts/diffrhythm_serverless_worker.py`), rewrote TypeScript client to use RunPod Serverless API (`/run`, `/status/{id}`). Audio returned as base64 in job output. Env var changed: `DIFFRHYTHM_POD_ID` → `DIFFRHYTHM_ENDPOINT_ID`. Old HTTP server kept as reference (`scripts/diffrhythm_api_server.py`).
- **2026-02-16**: Added DiffRhythm and DiffRhythm+Pipeline engines — modular architecture: DiffRhythm (latent diffusion, Apache 2.0) → Demucs stem separation → Matchering mastering. Registered in engine registry (6 engines total). Updated frontend: engine selector (3-col grid), EngineBadge, hero section, API docs cards, comparison form.
- **2026-02-13**: Added YuE and YuE+PP engines to frontend (Playground engine selector, hero section, EngineBadge component). Updated comparison UI for 3-way layout (Suno vs YuE raw vs YuE+PP). Added enablePP checkbox to comparison form. Updated header and descriptions to reflect multi-engine architecture.