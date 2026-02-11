# Ace-Step Music Generation API

## Overview
Web-based API service that accepts music generation parameters and processes them using either ACE-Step v1.5 or HeartMuLa models, both deployed on RunPod GPU Pods. Returns generated music files via a React frontend with API docs, playground, and job history.

## Architecture
- **Frontend**: React + Vite with Tailwind CSS, shadcn/ui components
- **Backend**: Express.js REST API with multi-engine routing
- **Database**: PostgreSQL (Drizzle ORM)
- **External**: RunPod persistent Pods
  - ACE-Step: valyriantech/ace-step-1.5:latest (instrumental/vocal generation)
  - HeartMuLa: ambsd/heartmula-studio:latest (full song generation with vocals)

## Key Files
- `shared/schema.ts` - Data models (jobs, comparisons tables with engine field)
- `server/routes.ts` - API endpoints with multi-engine routing + comparison endpoints
- `server/runpod.ts` - ACE-Step Pod API integration (release_task, query_result, v1/audio)
- `server/heartmula.ts` - HeartMuLa Pod API integration (generate/music, jobs, audio)
- `server/suno.ts` - Suno API v1 client via sunoapi.org (generate, poll task status, download)
- `server/gemini.ts` - Gemini AI audio comparison module (A/B analysis) + song idea generation
- `server/storage.ts` - Database storage layer
- `server/db.ts` - Database connection
- `client/src/pages/home.tsx` - Main UI with engine selector, Playground, API Docs, Job History, Compare, Admin
- `scripts/deploy-runpod.ts` - RunPod Pod deployment script

## API Endpoints
- `GET /api/health` - Check API and both engine pod health
- `POST /api/generate` - Submit music generation job (engine: ace-step | heartmula)
- `GET /api/jobs` - List all jobs (actively polls engine APIs for status updates)
- `GET /api/jobs/:id` - Get job status (polls correct engine for updates)
- `GET /api/jobs/:id/audio` - Download generated audio file (proxied from correct pod)
- `GET /api/pod/diagnostics` - Pod diagnostics for both engines
- `GET /api/heartmula/gpu-settings` - View HeartMuLa GPU settings
- `POST /api/heartmula/reload-gpu` - Reload HeartMuLa model on GPU (fails if jobs processing)
- `POST /api/heartmula/enforce-gpu` - Manually enforce GPU-only settings on HeartMuLa pod
- `POST /api/compare` - Start A/B comparison (our engine vs Suno, analyzed by Gemini)
- `GET /api/comparisons` - List all comparisons
- `GET /api/comparisons/:id` - Get comparison status and Gemini analysis
- `GET /api/comparisons/:id/audio/:source` - Stream audio (source: ours | suno)
- `GET /api/suno/credits` - Check Suno API credits
- `POST /api/generate-song-idea` - Gemini generates a random song idea (prompt, lyrics, tags, title)

## ACE-Step Pod API (Internal)
- Pod URL: `https://{RUNPOD_POD_ID}-8000.proxy.runpod.net`
- `POST /release_task` - Create music generation task (caption, lyrics, duration)
- `POST /query_result` - Poll task status (task_id_list parameter)
- `GET /v1/audio` - Download audio file (path parameter)

## HeartMuLa Pod API (Internal)
- Pod URL: `https://{HEARTMULA_POD_ID}-8000.proxy.runpod.net`
- `POST /generate/music` - Create song generation task (prompt, lyrics, tags, duration_ms, seed)
- `GET /jobs/{job_id}` - Poll job status
- `GET /download_track/{job_id}` - Download generated audio file
- `GET /health` - Health check
- `GET /settings/gpu` - View GPU settings
- `POST /settings/gpu/reload` - Reload model on GPU (body: {})
- `PUT /settings/gpu` - Update GPU settings (quantization_4bit, sequential_offload, torch_compile)

## Environment Variables
- `DATABASE_URL` - PostgreSQL connection string
- `RUNPOD_API_KEY` - RunPod API key (secret, used by deploy script and diagnostics)
- `RUNPOD_POD_ID` - ACE-Step persistent Pod ID (env var, currently: 29y7b9eza8mof3)
- `HEARTMULA_POD_ID` - HeartMuLa persistent Pod ID (secret, currently: vb787qolnhhdjz with GPU env vars)
- `SESSION_SECRET` - Session secret
- `RUNPOD_ENDPOINT_ID` - Legacy RunPod endpoint (unused)

## GPU Requirements
- ACE-Step requires minimum 32GB VRAM recommended (works on 24GB with limitations)
- HeartMuLa requires 24GB+ VRAM (works on RTX 3090/4090/A100)
- Deploy script targets: L40S, RTX A6000, A100 80GB
- Container disk: 100GB (15GB image + model weights + outputs)

## Recent Changes
- 2026-02-10: Fixed GPU execution: Created new HeartMuLa pod (vb787qolnhhdjz) with correct env vars (HEARTMULA_SEQUENTIAL_OFFLOAD=false, HEARTMULA_4BIT=false, HEARTMULA_USE_MMGP=true). Root cause: env vars control model device placement at startup, API settings alone don't change it.
- 2026-02-10: Rewrote Suno API client to use v1 API (POST /api/v1/generate, GET /api/v1/generate/record-info, GET /api/v1/generate/credit)
- 2026-02-10: Suno model names updated: V5 (default), V4_5ALL, V4_5PLUS, V4_5, V4 (old chirp-* names mapped for compatibility)
- 2026-02-10: Fixed critical GPU enforcement bug: HeartMuLa resets settings to "auto" (CPU) after reload; new sequence: set→reload→wait→re-set→final reload
- 2026-02-10: Added startup-forced GPU reload (startupReloadDone flag) — forces reload even when settings appear correct
- 2026-02-10: Fixed Suno API v1 callBackUrl error — removed empty string (field is optional for polling mode)
- 2026-02-10: GPU enforcement now triggers model reload after settings change + full health verification
- 2026-02-10: Comparison error handling: each engine surfaces errors immediately to DB, doesn't wait for both
- 2026-02-10: Gemini song idea generator: POST /api/generate-song-idea generates creative prompts/lyrics/tags for testing
- 2026-02-10: "Generate Idea" button in Compare tab — Gemini fills all form fields with a creative song idea
- 2026-02-10: A/B comparison system: Suno API integration (sunoapi.org), Gemini audio analysis, Compare tab with dual players and score breakdown
- 2026-02-10: New endpoints: POST /api/compare, GET /api/comparisons, GET /api/comparisons/:id, GET /api/comparisons/:id/audio/:source
- 2026-02-10: server/gemini.ts - Gemini AI audio comparison (7 categories, 1-10 scores, recommendations)
- 2026-02-09: Reduced HeartMuLa max duration from 360s to 300s (model context limit: 8192 tokens, 360s overflows)
- 2026-02-09: Added Admin tab with pod statistics, GPU settings, health status, and GPU control buttons
- 2026-02-09: Added scheduled GPU reload system: /api/heartmula/schedule-reload, auto-reload when jobs complete
- 2026-02-09: GPU-only enforcement system: auto-check on startup, pre-generation hard-fail if GPU unverified, /api/heartmula/enforce-gpu endpoint
- 2026-02-09: Deploy script now supports both engines: `npx tsx scripts/deploy-runpod.ts heartmula` with auto GPU enforcement
- 2026-02-09: Added HeartMuLa quality params: title, negative_tags, temperature, cfg_scale, topk
- 2026-02-09: Added HeartMuLa as second generation engine with full API integration
- 2026-02-09: Added engine field to jobs schema, multi-engine routing in routes.ts
- 2026-02-09: Frontend engine selector (ACE-Step blue / HeartMuLa pink) with engine-specific UI
- 2026-02-09: HeartMuLa params: tags (style), lyrics, duration (up to 300s), seed
- 2026-02-09: Diagnostics endpoint now covers both engines
- 2026-02-09: Added pod diagnostics endpoint (/api/pod/diagnostics)
- 2026-02-09: Fixed audio download - handle ACE-Step's `/v1/audio?path=` URL format
- 2026-02-09: Deployed working ACE-Step pod on A100 SXM (29y7b9eza8mof3)
- 2026-02-09: Refactored from RunPod serverless to persistent Pod with direct API calls
- 2026-02-09: Added deploy script (scripts/deploy-runpod.ts) for RunPod Pod provisioning
- 2025-02-09: Initial build - schema, frontend UI, backend API

## GPU Policy (CRITICAL)
- **ALL pods MUST run in GPU-only mode. CPU fallback is NEVER acceptable.**
- HeartMuLa required settings: `sequential_offload="false"`, `quantization_4bit="false"`, `torch_compile=false`
- Deploy script (`scripts/deploy-runpod.ts`) automatically enforces GPU-only settings after pod creation
- Server auto-enforces GPU settings on startup, before every HeartMuLa generation, and after model reloads
- If GPU settings cannot be verified before generation, the request is rejected (HTTP 503)
- When deploying new pods of ANY engine, always ensure GPU-only execution is configured

## Notes
- RunPod API does NOT provide programmatic access to pod logs (no REST/GraphQL endpoint)
- Pod diagnostics uses: /health, /v1/stats, /v1/models from ACE-Step API + RunPod GraphQL for GPU metrics
- ACE-Step generates 2 audio tracks per batch by default (batch_size=2), we use the first
- HeartMuLa pod needs to be deployed separately and HEARTMULA_POD_ID set as a secret

## Environment variables

API_BEARER_TOKEN=change-me

