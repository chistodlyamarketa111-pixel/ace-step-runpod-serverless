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
import * as fs from "fs";
import * as path from "path";

export async function registerRoutes(
  _httpServer: Server,
  app: Express,
): Promise<Server> {

  app.get("/api/engines", async (_req, res) => {
    res.json({
      engines: registry.getAllInfo(),
    });
  });

  app.get("/api/health", async (_req, res) => {
    const aceStepEngine = registry.get("ace-step");

    const aceStepHealthy = await (aceStepEngine?.checkHealth() ?? Promise.resolve(false));

    res.json({
      api: true,
      aceStep: aceStepHealthy,
      aceStepConfigured: aceStepEngine?.isConfigured() ?? false,
    });
  });

  app.post("/api/generate", requireBearerAuth, async (req, res) => {
    try {
      const parsed = createJobSchema.parse(req.body);
      const engineId = "ace-step";

      const musicEngine = registry.get(engineId);
      if (!musicEngine) {
        return res.status(400).json({ error: `Engine not available` });
      }

      if (!musicEngine.isConfigured()) {
        return res.status(503).json({ error: `Engine not configured` });
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
        model: parsed.model,
        inference_steps: parsed.inference_steps,
        guidance_scale: parsed.guidance_scale,
        thinking: parsed.thinking,
        audio_format: parsed.audio_format,
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

  app.get("/api/jobs", requireBearerAuth, async (_req, res) => {
    res.json(await storage.getAllJobs());
  });

  app.get("/api/jobs/:id", requireBearerAuth, async (req, res) => {
    const job = await storage.getJob(req.params.id);
    if (!job) return res.status(404).json({ error: "Job not found" });

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

  app.get("/api/pod/diagnostics", requireBearerAuth, async (_req, res) => {
    const aceStepEngine = registry.get("ace-step");
    const diagnostics: Record<string, any> = {};
    if (aceStepEngine) {
      diagnostics.aceStep = await aceStepEngine.getDiagnostics();
    }
    res.json(diagnostics);
  });

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

      if (source === "ours") {
        if (!cmp.ourAudioUrl) {
          return res.status(404).json({ error: "No audio available" });
        }
        const musicEngine = registry.get(cmp.engine);
        if (!musicEngine) {
          return res.status(500).json({ error: `Engine not found` });
        }
        const result = await musicEngine.fetchAudio(cmp.ourAudioUrl);
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

  app.get("/api/suno/credits", requireBearerAuth, async (_req, res) => {
    res.json(await suno.getCredits());
  });

  app.post("/api/generate-song-idea", requireBearerAuth, async (req, res) => {
    res.json(await gemini.generateSongIdea("ace-step"));
  });

  app.get("/api/download-server-script", (_req, res) => {
    const filePath = path.resolve(__dirname, "../docker/ace-step/http_server.py");
    if (fs.existsSync(filePath)) {
      res.setHeader("Content-Type", "text/plain");
      res.send(fs.readFileSync(filePath, "utf-8"));
    } else {
      res.status(404).send("File not found");
    }
  });

  return _httpServer;
}
