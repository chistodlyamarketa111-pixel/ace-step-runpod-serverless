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
- To add a new engine: create a class implementing `MusicEngine`, register it in `initializeEngines()`

### Database
- **Database**: PostgreSQL
- **ORM**: Drizzle ORM with `drizzle-zod` for schema-to-validation integration
- **Schema**: `shared/schema.ts` — defines `jobs`, `users`, and `comparisons` tables
- **Connection**: `server/db.ts` using `pg.Pool` with `DATABASE_URL` env var
- **Migrations**: Drizzle Kit with config in `drizzle.config.ts`, migrations output to `./migrations`
- **Storage layer**: `server/storage.ts` — `DatabaseStorage` class implementing `IStorage` interface

### Key Database Tables
- **jobs**: Tracks music generation jobs (id, engine, status, prompt, lyrics, duration, style, progress, output URL, etc.)
- **comparisons**: A/B comparison records between our engines and Suno
- **users**: Basic user table
- **conversations/messages**: Chat integration tables (from Replit AI integrations)

### API Design
All endpoints are prefixed with `/api/`. Key routes:
- `GET /api/health` — Health check for API and all engine pods
- `GET /api/engines` — List available engines and their capabilities
- `POST /api/generate` — Submit a music generation job (requires Bearer auth)
- `GET /api/jobs` / `GET /api/jobs/:id` — List/get job status (polls engine APIs)
- `GET /api/jobs/:id/audio` — Download generated audio (proxied from engine pod)
- `POST /api/compare` — Start A/B comparison (our engine vs Suno, analyzed by Gemini)
- `GET /api/comparisons` / `GET /api/comparisons/:id` — Comparison results
- `POST /api/generate-song-idea` — Gemini generates random song ideas

## External Dependencies

### RunPod GPU Pods
- Music generation models run on persistent RunPod pods accessed via proxy URLs (`https://{POD_ID}-8000.proxy.runpod.net`)
- **ACE-Step Pod** (`RUNPOD_POD_ID` env var): Docker image `valyriantech/ace-step-1.5:latest`
- **HeartMuLa Pod** (`HEARTMULA_POD_ID` env var): Docker image `ambsd/heartmula-studio:latest`
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
- `RUNPOD_API_KEY` — RunPod API key for pod management
- `API_BEARER_TOKEN` — Bearer token for API authentication
- `SUNO_API_KEY` — Suno API key for comparisons
- `AI_INTEGRATIONS_GEMINI_API_KEY` — Gemini API key
- `AI_INTEGRATIONS_GEMINI_BASE_URL` — Gemini API base URL