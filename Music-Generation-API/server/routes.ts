import type { Express } from "express";
import { type Server } from "http";
import { storage } from "./storage";
import { createJobSchema, createComparisonSchema } from "@shared/schema";
import { registry } from "./engines";
import * as suno from "./suno";
import * as gemini from "./gemini";
import { log } from "./index";
import { ZodError } from "zod";
import { fromZodError } from "zod-validation-error";
import { requireBearerAuth } from "./middleware/auth";
import path from "path";
import fs from "fs";

export async function registerRoutes(
  _httpServer: Server,
  app: Express,
): Promise<Server> {
  // ===== PUBLIC =====

  app.get("/api/scripts/yue-server", (_req, res) => {
    const scriptPath = path.resolve(process.cwd(), "scripts/yue_api_server.py");
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({ error: "Script not found" });
    }
    res.setHeader("Content-Type", "text/plain");
    res.sendFile(scriptPath);
  });

  app.get("/api/scripts/diffrhythm-server", (_req, res) => {
    const scriptPath = path.resolve(process.cwd(), "scripts/diffrhythm_api_server.py");
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({ error: "Script not found" });
    }
    res.setHeader("Content-Type", "text/plain");
    res.sendFile(scriptPath);
  });

  app.get("/api/engines", async (_req, res) => {
    res.json({
      engines: registry.getAllInfo(),
    });
  });

  app.get("/api/health", async (_req, res) => {
    const aceStepEngine = registry.get("ace-step");
    const heartmulaEngine = registry.get("heartmula");
    const yueEngine = registry.get("yue");

    const [aceStepHealthy, heartmulaHealthy, yueHealthy] = await Promise.all([
      aceStepEngine?.checkHealth() ?? Promise.resolve(false),
      heartmulaEngine?.checkHealth() ?? Promise.resolve(false),
      yueEngine?.checkHealth() ?? Promise.resolve(false),
    ]);

    res.json({
      api: true,
      aceStep: aceStepHealthy,
      aceStepConfigured: aceStepEngine?.isConfigured() ?? false,
      heartmula: heartmulaHealthy,
      heartmulaConfigured: heartmulaEngine?.isConfigured() ?? false,
      yue: yueHealthy,
      yueConfigured: yueEngine?.isConfigured() ?? false,
    });
  });

  // ===== GENERATION =====

  app.post("/api/generate", requireBearerAuth, async (req, res) => {
    try {
      const parsed = createJobSchema.parse(req.body);
      const engineId = parsed.engine || "ace-step";

      const musicEngine = registry.get(engineId);
      if (!musicEngine) {
        return res.status(400).json({ error: `Unknown engine: ${engineId}` });
      }

      if (!musicEngine.isConfigured()) {
        return res.status(503).json({ error: `Engine ${engineId} not configured` });
      }

      const job = await storage.createJob({
        engine: engineId,
        prompt: parsed.prompt,
        lyrics: parsed.lyrics,
        duration: parsed.duration,
        style: parsed.style,
        instrument: parsed.instrument,
        tempo: parsed.bpm,
        inputParams: parsed,
      });

      // Специальная проверка для HeartMuLa (временный хак для GPU)
      if (engineId === "heartmula") {
        const heartmula = await import("./heartmula");
        const gpuCheck = await heartmula.verifyGpuAndReload();
        if (!gpuCheck.success) {
          return res.status(503).json({ error: gpuCheck.message });
        }
      }

      const result = await musicEngine.submitTask({
        prompt: parsed.prompt,
        lyrics: parsed.lyrics,
        duration: parsed.duration,
        style: parsed.style,
        instrument: parsed.instrument,
        tags: parsed.tags,
        negative_tags: parsed.negative_tags,
        title: parsed.title,
        bpm: parsed.bpm,
        seed: parsed.seed,
        temperature: parsed.temperature,
        cfg_scale: parsed.cfg_scale,
        topk: parsed.topk,
      });

      await storage.updateJob(job.id, {
        runpodJobId: result.taskId,
        status: "IN_PROGRESS",
      });

      res.status(201).json(await storage.getJob(job.id));
    } catch (err: any) {
      if (err instanceof ZodError) {
        return res.status(400).json({ error: fromZodError(err).message });
      }
      log(`Generate error: ${err.message}`, "routes");
      res.status(500).json({ error: "Failed to create generation job" });
    }
  });

  // ===== JOBS =====

  app.get("/api/jobs", requireBearerAuth, async (_req, res) => {
    res.json(await storage.getAllJobs());
  });

  app.get("/api/jobs/:id", requireBearerAuth, async (req, res) => {
    const job = await storage.getJob(req.params.id);
    if (!job) return res.status(404).json({ error: "Job not found" });

    // Poll engine if job is still in progress
    if (job.status === "IN_PROGRESS" && job.runpodJobId) {
      try {
        const musicEngine = registry.get(job.engine);
        if (musicEngine) {
          const taskStatus = await musicEngine.queryTaskStatus(job.runpodJobId);
          if (taskStatus.status === "COMPLETED" && taskStatus.audio_path) {
            await storage.updateJob(job.id, {
              status: "COMPLETED",
              outputUrl: taskStatus.audio_path,
              progress: 100,
              completedAt: new Date(),
            });
          } else if (taskStatus.status === "FAILED") {
            await storage.updateJob(job.id, {
              status: "FAILED",
              errorMessage: taskStatus.error || "Task failed",
              completedAt: new Date(),
            });
          }
          const updated = await storage.getJob(job.id);
          return res.json(updated);
        }
      } catch (err: any) {
        log(`Poll error for job ${job.id}: ${err.message}`, "routes");
      }
    }

    res.json(job);
  });

  app.get("/api/jobs/:id/audio", async (req, res) => {
    const job = await storage.getJob(req.params.id);
    if (!job || job.status !== "COMPLETED" || !job.outputUrl) {
      return res.status(404).json({ error: "Audio not available" });
    }

    const musicEngine = registry.get(job.engine);
    if (!musicEngine) {
      return res.status(500).json({ error: `Engine ${job.engine} not found` });
    }

    const result = await musicEngine.fetchAudio(job.outputUrl);

    res.setHeader("Content-Type", result.contentType);
    res.send(result.buffer);
  });

  // ===== DIAGNOSTICS =====

  app.get("/api/pod/diagnostics", requireBearerAuth, async (_req, res) => {
    const aceStepEngine = registry.get("ace-step");
    const heartmulaEngine = registry.get("heartmula");
    const yueEngine = registry.get("yue");

    const diagnostics: Record<string, any> = {};

    if (aceStepEngine) {
      diagnostics.aceStep = await aceStepEngine.getDiagnostics();
    }
    if (heartmulaEngine) {
      diagnostics.heartmula = await heartmulaEngine.getDiagnostics();
    }
    if (yueEngine) {
      diagnostics.yue = await yueEngine.getDiagnostics();
    }

    res.json(diagnostics);
  });

  // ===== HEARTMULA CONTROL =====
  // Note: These endpoints use HeartMuLa-specific GPU functions not in the universal interface

  app.post(
    "/api/heartmula/reload-gpu",
    requireBearerAuth,
    async (_req, res) => {
      const heartmula = await import("./heartmula");
      res.json(await heartmula.reloadGpuModel());
    },
  );

  app.post(
    "/api/heartmula/enforce-gpu",
    requireBearerAuth,
    async (_req, res) => {
      const heartmula = await import("./heartmula");
      res.json(await heartmula.enforceGpuSettings());
    },
  );

  app.post(
    "/api/heartmula/schedule-reload",
    requireBearerAuth,
    async (_req, res) => {
      const heartmula = await import("./heartmula");
      heartmula.schedulGpuReload();
      res.json({ success: true });
    },
  );

  app.get(
    "/api/heartmula/gpu-settings",
    requireBearerAuth,
    async (_req, res) => {
      const heartmula = await import("./heartmula");
      res.json(await heartmula.getGpuSettings());
    },
  );

  // ===== YUE ADMIN =====

  app.get("/api/yue/deploy-command", requireBearerAuth, async (req, res) => {
    const host = req.get("host") || "localhost:5000";
    const protocol = req.protocol === "https" ? "https" : (host.includes("replit") ? "https" : "http");
    const scriptUrl = `${protocol}://${host}/api/yue/server-script`;
    const command = `pkill -f "python.*gradio_app\\|python.*interface" 2>/dev/null; curl -s "${scriptUrl}" -o /workspace/yue_api_server.py && nohup python /workspace/yue_api_server.py > /workspace/yue_api.log 2>&1 &`;
    res.json({
      message: "Run this command in RunPod Web Terminal to deploy the YuE API server",
      command,
      scriptUrl,
      note: "This stops Gradio and starts our lightweight API server on port 8000",
    });
  });

  app.get("/api/yue/diagnostics", requireBearerAuth, async (_req, res) => {
    const yue = await import("./yue");
    res.json(await yue.getPodDiagnostics());
  });

  app.get("/api/yue/server-script", async (_req, res) => {
    const fs = await import("fs");
    const path = await import("path");
    const possiblePaths = [
      path.join(process.cwd(), "scripts", "yue_api_server.py"),
      path.join(process.cwd(), "Music-Generation-API", "scripts", "yue_api_server.py"),
    ];
    for (const scriptPath of possiblePaths) {
      if (fs.existsSync(scriptPath)) {
        const script = fs.readFileSync(scriptPath, "utf-8");
        res.setHeader("Content-Type", "text/plain");
        res.send(script);
        return;
      }
    }
    res.status(404).json({ error: "Script not found" });
  });

  // ===== COMPARISON =====

  app.post("/api/compare", requireBearerAuth, async (req, res) => {
    const parsed = createComparisonSchema.parse(req.body);
    const comparison = await storage.createComparison(parsed);
    res.status(201).json(comparison);
  });

  app.get("/api/comparisons", requireBearerAuth, async (_req, res) => {
    res.json(await storage.getAllComparisons());
  });

  app.get("/api/comparisons/:id", requireBearerAuth, async (req, res) => {
    const cmp = await storage.getComparison(req.params.id);
    if (!cmp) return res.status(404).json({ error: "Not found" });
    res.json(cmp);
  });

  app.get(
    "/api/comparisons/:id/audio/:source",
    requireBearerAuth,
    async (req, res) => {
      const cmp = await storage.getComparison(req.params.id);
      if (!cmp) return res.status(404).json({ error: "Not found" });

      const source = req.params.source;

      if (source === "ours" || source === "ours_pp") {
        const audioUrl = source === "ours_pp" ? cmp.ourPpAudioUrl : cmp.ourAudioUrl;
        if (!audioUrl) {
          return res.status(404).json({ error: `No ${source} audio available` });
        }

        let engineId = cmp.engine;
        if (source === "ours_pp") {
          if (cmp.engine === "diffrhythm") {
            engineId = "diffrhythm-pp";
          } else {
            engineId = "yue-pp";
          }
        }
        const musicEngine = registry.get(engineId) || registry.get(cmp.engine);
        if (!musicEngine) {
          return res.status(500).json({ error: `Engine ${engineId} not found` });
        }

        const result = await musicEngine.fetchAudio(audioUrl);
        res.setHeader("Content-Type", result.contentType);
        res.send(result.buffer);
      } else if (source === "suno") {
        if (!cmp.sunoAudioUrl) {
          return res.status(404).json({ error: "No Suno audio available" });
        }
        res.setHeader("Content-Type", "audio/mpeg");
        res.send(await suno.downloadAudio(cmp.sunoAudioUrl));
      } else {
        return res.status(400).json({ error: `Unknown source: ${source}` });
      }
    },
  );

  // ===== SUNO / GEMINI =====

  app.get("/api/suno/credits", requireBearerAuth, async (_req, res) => {
    res.json(await suno.getCredits());
  });

  app.post("/api/generate-song-idea", requireBearerAuth, async (req, res) => {
    res.json(await gemini.generateSongIdea(req.body?.engine || "heartmula"));
  });

  return _httpServer;
}
