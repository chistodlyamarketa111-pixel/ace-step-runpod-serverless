# Music Generation API

## Overview

This is a web-based music generation API service that accepts music generation parameters and processes them through ACE-Step v1.5 AI model deployed on RunPod Serverless GPU infrastructure. The application provides a React frontend playground for testing, API documentation via Swagger, job history tracking, and A/B comparison capabilities with Suno (a commercial music generation service), with analysis powered by Google Gemini AI.

The main application lives in the `Music-Generation-API/` directory. The root-level `package.json` contains some shared dependencies but the primary project is inside `Music-Generation-API/`.

## User Preferences

Preferred communication style: Simple, everyday language.
Preferred language: Russian.

## System Architecture

### Frontend
- **Framework**: React 18 with TypeScript, built with Vite
- **Styling**: Tailwind CSS with CSS variables for theming (light/dark mode support)
- **Component Library**: shadcn/ui (New York style) — extensive set of Radix UI-based components in `client/src/components/ui/`
- **Routing**: Wouter (lightweight client-side router)
- **State Management**: TanStack React Query for server state
- **Forms**: React Hook Form with Zod validation via `@hookform/resolvers`
- **Entry point**: `client/src/main.tsx` → `client/src/App.tsx`
- **Main page**: `client/src/pages/home.tsx` — contains playground, API docs, job history, compare, and admin tabs

### Backend
- **Framework**: Express.js with TypeScript, run via `tsx`
- **Entry point**: `server/index.ts`
- **API Routes**: `server/routes.ts` — RESTful endpoints for generation, job management, comparisons
- **Authentication**: Bearer token auth middleware (`server/middleware/auth.ts`) — single shared token from `API_BEARER_TOKEN` env var
- **API Documentation**: Swagger UI served at `/docs` from `server/openapi.yaml`
- **Build**: Custom build script (`script/build.ts`) using esbuild for server + Vite for client, outputs to `dist/`

### Engine Architecture
- **Single engine**: ACE-Step v1.5 (`server/engines/ace-step.ts` → delegates to `server/runpod.ts`)
- **RunPod Serverless**: Uses RunPod Serverless API (`/v2/{endpoint}/run`, `/v2/{endpoint}/status/{id}`)
- **Audio delivery**: Base64-encoded audio returned in job output
- **Serverless worker**: `scripts/ace_step_serverless_worker.py` — RunPod handler using `ace-step` Python package
- **Registry**: `server/engines/registry.ts` — engine registry (currently single engine, extensible)

### Database
- **Database**: PostgreSQL
- **ORM**: Drizzle ORM with `drizzle-zod` for schema-to-validation integration
- **Schema**: `shared/schema.ts` — defines `jobs`, `users`, and `comparisons` tables
- **Connection**: `server/db.ts` using `pg.Pool` with `DATABASE_URL` env var
- **Migrations**: Drizzle Kit with config in `drizzle.config.ts`, migrations output to `./migrations`
- **Storage layer**: `server/storage.ts` — `DatabaseStorage` class implementing `IStorage` interface

### Key Database Tables
- **jobs**: Tracks music generation jobs (id, engine, status, prompt, lyrics, duration, style, progress, output URL, etc.)
- **comparisons**: A/B comparison records between ACE-Step and Suno
- **users**: Basic user table

### API Design
All endpoints are prefixed with `/api/`. Key routes:
- `GET /api/health` — Health check for API and ACE-Step endpoint
- `GET /api/engines` — List available engines
- `POST /api/generate` — Submit a music generation job (requires Bearer auth)
- `GET /api/jobs` / `GET /api/jobs/:id` — List/get job status (polls RunPod Serverless API)
- `GET /api/jobs/:id/audio` — Download generated audio (decoded from base64)
- `POST /api/compare` — Start A/B comparison (ACE-Step vs Suno, analyzed by Gemini)
- `GET /api/comparisons` / `GET /api/comparisons/:id` — Comparison results
- `POST /api/generate-song-idea` — Gemini generates random song ideas

## External Dependencies

### RunPod Serverless
- ACE-Step model runs on RunPod Serverless endpoint
- GPU spins up only on request (no idle costs)
- Docker image: `docker/ace-step/` — complete Dockerfile + handler for RunPod Serverless
- Worker handler: `docker/ace-step/handler.py` — uses ACE-Step v1.5 inference API (GenerationParams, generate_music)
- Build: `cd docker/ace-step && ./build.sh ace-step-serverless latest`
- Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` + ACE-Step v1.5 from GitHub
- Models pre-downloaded at build time from HuggingFace (`ACE-Step/Ace-Step1.5`)
- Env var: `ACESTEP_ENDPOINT_ID` — RunPod Serverless endpoint ID
- Env var: `RUNPOD_API_KEY` — RunPod API key

### Suno API
- Commercial music generation service accessed via `https://api.sunoapi.org`
- Used as benchmark in A/B comparisons
- Requires `SUNO_API_KEY` env var
- Client implementation: `server/suno.ts`

### Google Gemini AI
- Used for A/B audio comparison analysis and song idea generation
- Configured via `AI_INTEGRATIONS_GEMINI_API_KEY` and `AI_INTEGRATIONS_GEMINI_BASE_URL` env vars
- Client implementation: `server/gemini.ts`

### Environment Variables Required
- `DATABASE_URL` — PostgreSQL connection string
- `ACESTEP_ENDPOINT_ID` — RunPod Serverless endpoint ID for ACE-Step
- `RUNPOD_API_KEY` — RunPod API key for serverless endpoints
- `API_BEARER_TOKEN` — Bearer token for API authentication
- `SUNO_API_KEY` — Suno API key for comparisons (optional)
- `AI_INTEGRATIONS_GEMINI_API_KEY` — Gemini API key
- `AI_INTEGRATIONS_GEMINI_BASE_URL` — Gemini API base URL

## Recent Changes
- **2026-02-19**: Removed turbo-shift3 model (incompatible with ACE-Step v1.5 API). Now 3 models: turbo (8 steps), sft (32 steps), base (50 steps). Added http_server.py to Docker image for pod testing. Added CPU offload support for long audio generation (150s+). Tested all 3 models successfully on RTX 4090 with 150s blues tracks.
- **2026-02-18**: Added dynamic model switching. Handler supports 3 DiT models (turbo/sft/base) with lazy loading and caching. Frontend model selector auto-sets recommended inference steps. Model parameter flows through API → routes → runpod → handler.
- **2026-02-18**: Created full Docker image for ACE-Step v1.5 RunPod Serverless deployment (`docker/ace-step/`). Handler uses proper v1.5 inference API with GenerationParams + generate_music. Models pre-downloaded at build time. Cleaned up old engine references (HeartMuLa, YuE, DiffRhythm) from frontend and backend.
- **2026-02-18**: Simplified to single ACE-Step engine on RunPod Serverless. Removed HeartMuLa, YuE, YuE+PP, DiffRhythm, DiffRhythm+PP engines and all related files. Rewrote `runpod.ts` to use RunPod Serverless API with base64 audio. Simplified frontend to single-engine UI. Removed unused scripts and Docker infrastructure. Env var changed: `RUNPOD_POD_ID` → `ACESTEP_ENDPOINT_ID`.
