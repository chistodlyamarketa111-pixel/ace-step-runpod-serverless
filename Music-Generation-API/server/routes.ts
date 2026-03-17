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
import { getMode } from "./runpod";
import * as fs from "fs";
import * as path from "path";
import { processBatch, getBatchStatus } from "./compare-processor";

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
      deployMode: getMode(),
    });
  });

  app.get("/api/scripts/:name", (req, res) => {
    const allowed = ["upload_lora_to_hf.py", "pod_setup.sh", "train_v3.py", "download_dataset.py"];
    const name = req.params.name;
    if (!allowed.includes(name)) {
      return res.status(404).json({ error: "Script not found" });
    }
    const scriptPath = path.join(process.cwd(), "scripts", name);
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({ error: "Script file not found" });
    }
    res.setHeader("Content-Type", "text/plain");
    res.setHeader("Content-Disposition", `attachment; filename="${name}"`);
    res.sendFile(path.resolve(scriptPath));
  });

  app.get("/api/loras", async (_req, res) => {
    res.json({
      loras: [
        { id: "none", name: "No LoRA (base model)", description: "Use the base ACE-Step model without any adapter" },
        { id: "russianpop", name: "Russian Pop", description: "Russian pop music style (trained on 150+ Russian pop tracks)" },
        { id: "russianrap", name: "Russian Rap", description: "Russian rap/hip-hop style (trained on 150+ Russian rap tracks)" },
      ],
      custom_lora_info: "LoRA adapters are auto-downloaded from HuggingFace on first use",
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
        lora_name: parsed.lora_name,
        lora_scale: parsed.lora_scale,
        lora_revision: parsed.lora_revision,
        mode: parsed.mode,
        vocal_volume: parsed.vocal_volume,
        instrumental_volume: parsed.instrumental_volume,
        mastering: parsed.mastering,
        enhance: parsed.enhance,
        enhance_mode: parsed.enhance_mode,
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

  app.get("/api/jobs", async (_req, res) => {
    const allJobs = await storage.getAllJobs();
    const sanitized = allJobs.map(({ audioData, ...rest }: any) => rest);
    res.json(sanitized);
  });

  app.get("/api/jobs/:id", async (req, res) => {
    const jobRaw = await storage.getJob(req.params.id);
    if (!jobRaw) return res.status(404).json({ error: "Job not found" });
    const { audioData: _ad, ...job } = jobRaw as any;

    if (job.status === "IN_PROGRESS" && job.runpodJobId) {
      try {
        const musicEngine = registry.get(job.engine);
        if (musicEngine) {
          const taskStatus = await musicEngine.queryTaskStatus(job.runpodJobId);
          if (taskStatus.status === "COMPLETED" && taskStatus.audio_path) {
            const updateData: Record<string, any> = {
              status: "COMPLETED",
              outputUrl: taskStatus.audio_path,
              progress: 100,
              completedAt: new Date(),
            };
            try {
              const audioResult = await musicEngine.fetchAudio(taskStatus.audio_path);
              updateData.audioData = audioResult.buffer.toString("base64");
              updateData.audioFormat = audioResult.contentType.includes("wav") ? "wav" : audioResult.contentType.includes("flac") ? "flac" : "mp3";
            } catch (audioErr: any) {
              log(`Could not persist audio for job ${job.id}: ${audioErr.message}`, "routes");
            }
            await storage.updateJob(job.id, updateData);
          } else if (taskStatus.status === "FAILED") {
            await storage.updateJob(job.id, {
              status: "FAILED",
              errorMessage: taskStatus.error || "Task failed",
              completedAt: new Date(),
            });
          }
          const updatedRaw = await storage.getJob(job.id);
          const { audioData: _ad2, ...updated } = (updatedRaw || {}) as any;
          return res.json(updated);
        }
      } catch (err: any) {
        log(`Poll error for job ${job.id}: ${err.message}`, "routes");
      }
    }

    res.json(job);
  });

  app.get("/api/jobs/:id/audio", async (req, res) => {
    const job = await storage.getJob(req.params.id) as any;
    if (!job || job.status !== "COMPLETED") {
      return res.status(404).json({ error: "Audio not available" });
    }

    if (job.audioData) {
      const buffer = Buffer.from(job.audioData, "base64");
      const fmt = job.audioFormat || "wav";
      const contentType = fmt === "wav" ? "audio/wav" : fmt === "flac" ? "audio/flac" : "audio/mpeg";
      const ext = fmt === "wav" ? "wav" : fmt === "flac" ? "flac" : "mp3";
      const params = (job as any).inputParams || {};
      const model = (params.model || "sft").replace("acestep-v15-", "");
      const loraName = params.lora_name && params.lora_name !== "none" ? params.lora_name : "no-lora";
      const loraScale = params.lora_scale != null ? `x${params.lora_scale}` : "";
      const engine = job.engine || "ace-step";
      const filename = `${engine}-${model}-${loraName}${loraScale ? `-${loraScale}` : ""}.${ext}`;
      res.setHeader("Content-Type", contentType);
      res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
      return res.send(buffer);
    }

    if (!job.outputUrl) {
      return res.status(404).json({ error: "Audio not available" });
    }

    const musicEngine = registry.get(job.engine);
    if (!musicEngine) {
      return res.status(500).json({ error: `Engine ${job.engine} not found` });
    }

    try {
      const result = await musicEngine.fetchAudio(job.outputUrl);
      await storage.updateJob(job.id, {
        audioData: result.buffer.toString("base64"),
        audioFormat: result.contentType.includes("wav") ? "wav" : result.contentType.includes("flac") ? "flac" : "mp3",
      } as any);
      res.setHeader("Content-Type", result.contentType);
      res.send(result.buffer);
    } catch (err: any) {
      return res.status(404).json({ error: `Audio fetch failed: ${err.message}` });
    }
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

  app.post("/api/run-comparison-batch", requireBearerAuth, async (req, res) => {
    try {
      const status = getBatchStatus();
      if (status.running) {
        return res.status(409).json({ error: "Batch is already running", status });
      }
      const { caseIds } = req.body || {};
      if (caseIds && (!Array.isArray(caseIds) || caseIds.some((id: any) => typeof id !== "string"))) {
        return res.status(400).json({ error: "caseIds must be an array of strings" });
      }
      processBatch(caseIds).catch((err) => {
        log(`Batch error: ${err.message}`, "compare");
      });
      res.json({ started: true, message: "Batch comparison started in background" });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/api/comparison-batch-status", requireBearerAuth, async (_req, res) => {
    res.json(getBatchStatus());
  });

  app.get("/api/compare/cases", (_req, res) => {
    const casesPath = path.resolve(process.cwd(), "data/comparison_cases.json");
    if (!fs.existsSync(casesPath)) {
      return res.status(404).json({ error: "Comparison cases not found" });
    }
    try {
      const data = JSON.parse(fs.readFileSync(casesPath, "utf-8"));
      const audioDir = path.resolve(process.cwd(), "public/audio");
      const enriched = data.map((c: any) => ({
        ...c,
        ready: fs.existsSync(path.join(audioDir, `${c.id}_a.mp3`)) && fs.existsSync(path.join(audioDir, `${c.id}_b.mp3`)),
      }));
      res.json(enriched);
    } catch (err: any) {
      res.status(500).json({ error: "Failed to read comparison cases" });
    }
  });

  app.get("/api/compare/reveal", (_req, res) => {
    const mappingPath = path.resolve(process.cwd(), "data/track_mapping.json");
    if (!fs.existsSync(mappingPath)) {
      return res.status(404).json({ error: "Track mapping not found" });
    }
    try {
      const mapping = JSON.parse(fs.readFileSync(mappingPath, "utf-8"));
      res.json(mapping);
    } catch (err: any) {
      res.status(500).json({ error: "Failed to read track mapping" });
    }
  });

  app.get("/api/download-server-script", (_req, res) => {
    const possiblePaths = [
      path.resolve(process.cwd(), "docker/ace-step/http_server.py"),
      path.resolve(process.cwd(), "../docker/ace-step/http_server.py"),
      path.resolve(process.cwd(), "Music-Generation-API/docker/ace-step/http_server.py"),
    ];
    for (const filePath of possiblePaths) {
      if (fs.existsSync(filePath)) {
        res.setHeader("Content-Type", "text/plain; charset=utf-8");
        res.setHeader("Content-Disposition", "attachment; filename=http_server.py");
        res.send(fs.readFileSync(filePath, "utf-8"));
        return;
      }
    }
    res.status(404).send("File not found. Tried: " + possiblePaths.join(", "));
  });

  app.post("/api/internal/upload-audio", (req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString());
        const { filename, audio_base64 } = body;
        if (!filename || !audio_base64) {
          res.status(400).json({ error: "Missing filename or audio_base64" });
          return;
        }
        const safeName = path.basename(filename).replace(/[^a-zA-Z0-9_.-]/g, "");
        const outDir = path.resolve(process.cwd(), "public/audio/ace-step");
        if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
        const outPath = path.join(outDir, safeName);
        fs.writeFileSync(outPath, Buffer.from(audio_base64, "base64"));
        const sz = fs.statSync(outPath).size;
        res.json({ ok: true, path: outPath, size: sz });
      } catch (e: any) {
        res.status(500).json({ error: e.message });
      }
    });
  });

  return _httpServer;
}
