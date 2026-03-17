import * as fs from "fs";
import * as path from "path";
import { registry } from "./engines";
import * as suno from "./suno";
import { log } from "./index";

interface ComparisonCase {
  id: string;
  category: string;
  title: string;
  style_prompt: string;
  lyrics: string;
  track_a: string;
  track_b: string;
}

interface TrackMapping {
  [caseId: string]: {
    track_a: "ace-step" | "suno";
    track_b: "ace-step" | "suno";
  };
}

interface BatchStatus {
  running: boolean;
  total: number;
  completed: number;
  failed: number;
  current: string | null;
  results: { id: string; status: string; error?: string }[];
}

let batchStatus: BatchStatus = {
  running: false,
  total: 0,
  completed: 0,
  failed: 0,
  current: null,
  results: [],
};

export function getBatchStatus(): BatchStatus {
  return { ...batchStatus };
}

function getAudioDir(): string {
  const dir = path.resolve(process.cwd(), "public/audio");
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return dir;
}

function getDataDir(): string {
  return path.resolve(process.cwd(), "data");
}

function loadCases(): ComparisonCase[] {
  const casesPath = path.join(getDataDir(), "comparison_cases.json");
  return JSON.parse(fs.readFileSync(casesPath, "utf-8"));
}

function loadMapping(): TrackMapping {
  const mappingPath = path.join(getDataDir(), "track_mapping.json");
  if (fs.existsSync(mappingPath)) {
    return JSON.parse(fs.readFileSync(mappingPath, "utf-8"));
  }
  return {};
}

function saveMapping(mapping: TrackMapping): void {
  const mappingPath = path.join(getDataDir(), "track_mapping.json");
  fs.writeFileSync(mappingPath, JSON.stringify(mapping, null, 2));
}

async function pollAceStep(taskId: string, maxWaitMs = 600000): Promise<Buffer> {
  const engine = registry.get("ace-step");
  if (!engine) throw new Error("ACE-Step engine not available");

  const startTime = Date.now();
  while (Date.now() - startTime < maxWaitMs) {
    const status = await engine.queryTaskStatus(taskId);
    log(`ACE-Step poll: ${taskId} -> ${status.status}`, "compare");

    if (status.status === "COMPLETED" && status.audio_path) {
      const result = await engine.fetchAudio(status.audio_path);
      return result.buffer;
    }

    if (status.status === "FAILED") {
      throw new Error(`ACE-Step failed: ${status.error || "unknown"}`);
    }

    await new Promise((r) => setTimeout(r, 10000));
  }

  throw new Error("ACE-Step generation timed out after 10 minutes");
}

async function processCase(c: ComparisonCase): Promise<void> {
  const audioDir = getAudioDir();
  const aPath = path.join(audioDir, `${c.id}_a.mp3`);
  const bPath = path.join(audioDir, `${c.id}_b.mp3`);

  if (fs.existsSync(aPath) && fs.existsSync(bPath)) {
    log(`Skipping ${c.id} — both tracks already exist`, "compare");
    return;
  }

  const mapping = loadMapping();
  let assignment = mapping[c.id];
  if (!assignment) {
    const sunoIsA = Math.random() < 0.5;
    assignment = {
      track_a: sunoIsA ? "suno" : "ace-step",
      track_b: sunoIsA ? "ace-step" : "suno",
    };
    mapping[c.id] = assignment;
    saveMapping(mapping);
  }

  log(`Processing ${c.id}: "${c.title}" (A=${assignment.track_a}, B=${assignment.track_b})`, "compare");

  const engine = registry.get("ace-step");
  if (!engine || !engine.isConfigured()) {
    throw new Error("ACE-Step engine not configured");
  }

  if (!suno.isConfigured()) {
    throw new Error("Suno API not configured");
  }

  const aceStepPromise = (async () => {
    log(`${c.id}: Submitting to ACE-Step...`, "compare");
    const result = await engine.submitTask({
      prompt: c.style_prompt,
      lyrics: c.lyrics,
      duration: 60,
      audio_format: "mp3",
      mastering: true,
      lora_name: "russianrap",
      lora_scale: 0.7,
    });
    log(`${c.id}: ACE-Step taskId=${result.taskId}`, "compare");
    return pollAceStep(result.taskId);
  })();

  const sunoPromise = (async () => {
    log(`${c.id}: Submitting to Suno...`, "compare");
    const result = await suno.generateMusic({
      prompt: c.style_prompt,
      lyrics: c.lyrics,
      tags: c.style_prompt,
      title: c.title,
      model: "V5",
    });
    log(`${c.id}: Suno taskId=${result.taskId}`, "compare");
    const completed = await suno.pollUntilComplete(result.taskId);
    return suno.downloadAudio(completed.audio_url);
  })();

  const [aceStepAudio, sunoAudio] = await Promise.all([aceStepPromise, sunoPromise]);

  const trackABuffer = assignment.track_a === "ace-step" ? aceStepAudio : sunoAudio;
  const trackBBuffer = assignment.track_b === "ace-step" ? aceStepAudio : sunoAudio;

  fs.writeFileSync(aPath, trackABuffer);
  fs.writeFileSync(bPath, trackBBuffer);

  log(`${c.id}: Saved ${aPath} (${trackABuffer.length}b) and ${bPath} (${trackBBuffer.length}b)`, "compare");
}

export async function processBatch(caseIds?: string[]): Promise<void> {
  if (batchStatus.running) {
    throw new Error("Batch is already running");
  }

  const allCases = loadCases();
  const cases = caseIds
    ? allCases.filter((c) => caseIds.includes(c.id))
    : allCases;

  batchStatus = {
    running: true,
    total: cases.length,
    completed: 0,
    failed: 0,
    current: null,
    results: [],
  };

  log(`Starting batch: ${cases.length} cases`, "compare");

  for (const c of cases) {
    batchStatus.current = c.id;

    try {
      await processCase(c);
      batchStatus.completed++;
      batchStatus.results.push({ id: c.id, status: "completed" });
      log(`${c.id}: Done (${batchStatus.completed}/${batchStatus.total})`, "compare");
    } catch (err: any) {
      batchStatus.failed++;
      batchStatus.results.push({ id: c.id, status: "failed", error: err.message });
      log(`${c.id}: FAILED — ${err.message}`, "compare");
    }

    if (batchStatus.completed + batchStatus.failed < cases.length) {
      await new Promise((r) => setTimeout(r, 5000));
    }
  }

  batchStatus.running = false;
  batchStatus.current = null;
  log(`Batch complete: ${batchStatus.completed} ok, ${batchStatus.failed} failed`, "compare");
}
