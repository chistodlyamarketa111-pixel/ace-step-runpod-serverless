import { log } from "./index";

const ACESTEP_ENDPOINT_ID = process.env.ACESTEP_ENDPOINT_ID;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;

if (!ACESTEP_ENDPOINT_ID) {
  console.warn("ACESTEP_ENDPOINT_ID not set - ACE-Step integration will not work");
}
if (!RUNPOD_API_KEY) {
  console.warn("RUNPOD_API_KEY not set - ACE-Step integration will not work");
}

function getRunUrl(): string {
  return `https://api.runpod.ai/v2/${ACESTEP_ENDPOINT_ID}/run`;
}

function getStatusUrl(jobId: string): string {
  return `https://api.runpod.ai/v2/${ACESTEP_ENDPOINT_ID}/status/${jobId}`;
}

function getHeaders(): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${RUNPOD_API_KEY}`,
  };
}

export function isConfigured(): boolean {
  return Boolean(ACESTEP_ENDPOINT_ID && RUNPOD_API_KEY);
}

export async function submitTask(params: {
  prompt: string;
  lyrics?: string;
  duration?: number;
  style?: string;
  instrument?: string;
  bpm?: number;
  key_scale?: string;
  time_signature?: string;
  vocal_language?: string;
  inference_steps?: number;
  guidance_scale?: number;
  thinking?: boolean;
  shift?: number;
  infer_method?: string;
  use_adg?: boolean;
  batch_size?: number;
  seed?: number;
  audio_format?: string;
  model?: string;
}): Promise<{ task_id: string }> {
  if (!isConfigured()) {
    throw new Error("ACE-Step Serverless is not configured. Set ACESTEP_ENDPOINT_ID and RUNPOD_API_KEY.");
  }

  const caption = buildCaption(params);

  const input: Record<string, any> = {
    prompt: caption,
    audio_duration: params.duration || 30,
    task_type: "text2music",
    audio_format: params.audio_format || "mp3",
  };

  if (params.lyrics?.trim()) input.lyrics = params.lyrics;
  if (params.bpm !== undefined) input.bpm = params.bpm;
  if (params.key_scale) input.key_scale = params.key_scale;
  if (params.time_signature) input.time_signature = params.time_signature;
  if (params.vocal_language) input.vocal_language = params.vocal_language;
  if (params.inference_steps !== undefined) input.inference_steps = params.inference_steps;
  if (params.guidance_scale !== undefined) input.guidance_scale = params.guidance_scale;
  if (params.thinking !== undefined) input.thinking = params.thinking;
  if (params.shift !== undefined) input.shift = params.shift;
  if (params.infer_method) input.infer_method = params.infer_method;
  if (params.use_adg !== undefined) input.use_adg = params.use_adg;
  if (params.batch_size !== undefined && params.batch_size > 1) input.batch_size = params.batch_size;
  if (params.seed !== undefined && params.seed !== -1) {
    input.seed = params.seed;
  }
  if (params.model) input.model = params.model;

  log(`Submitting task to ACE-Step Serverless: ${JSON.stringify(input)}`, "runpod");

  const response = await fetch(getRunUrl(), {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ input }),
    signal: AbortSignal.timeout(30000),
  });

  if (!response.ok) {
    const text = await response.text();
    log(`ACE-Step submit error: ${response.status} ${text}`, "runpod");
    throw new Error(`ACE-Step Serverless error: ${response.status} - ${text}`);
  }

  const raw = await response.json();
  log(`ACE-Step submit response: ${JSON.stringify(raw)}`, "runpod");

  if (!raw.id) {
    throw new Error("ACE-Step Serverless did not return an id");
  }

  return { task_id: raw.id };
}

export async function queryTaskStatus(taskId: string): Promise<{
  status: string;
  audio_path?: string;
  error?: string;
}> {
  if (!isConfigured()) {
    throw new Error("ACE-Step Serverless is not configured.");
  }

  const response = await fetch(getStatusUrl(taskId), {
    headers: getHeaders(),
    signal: AbortSignal.timeout(15000),
  });

  if (!response.ok) {
    const text = await response.text();
    log(`ACE-Step status error: ${response.status} ${text}`, "runpod");
    throw new Error(`ACE-Step status error: ${response.status} - ${text}`);
  }

  const raw = await response.json();
  log(`ACE-Step status for ${taskId}: status=${raw.status}`, "runpod");

  const status = raw.status;

  if (status === "COMPLETED") {
    const output = raw.output;
    if (output?.error) {
      return {
        status: "FAILED",
        error: output.error,
      };
    }
    return {
      status: "COMPLETED",
      audio_path: taskId,
    };
  }

  if (status === "FAILED") {
    return {
      status: "FAILED",
      error: raw.error || "Task failed",
    };
  }

  if (status === "CANCELLED") {
    return {
      status: "FAILED",
      error: "Task was cancelled",
    };
  }

  return { status: "IN_PROGRESS" };
}

export async function fetchAudio(audioIdentifier: string): Promise<{ buffer: Buffer; contentType: string }> {
  if (!isConfigured()) {
    throw new Error("ACE-Step Serverless is not configured.");
  }

  const taskId = audioIdentifier;

  const response = await fetch(getStatusUrl(taskId), {
    headers: getHeaders(),
    signal: AbortSignal.timeout(30000),
  });

  if (!response.ok) {
    throw new Error(`Failed to get job status: ${response.status}`);
  }

  const raw = await response.json();

  if (raw.status !== "COMPLETED" || !raw.output) {
    throw new Error(`Job ${taskId} is not completed`);
  }

  const output = raw.output;

  if (output.audio_base64) {
    const buffer = Buffer.from(output.audio_base64, "base64");
    const contentType = output.content_type || "audio/mpeg";
    return { buffer, contentType };
  }

  throw new Error("No audio data in job output");
}

export async function checkHealth(): Promise<boolean> {
  if (!isConfigured()) return false;

  try {
    const response = await fetch(`https://api.runpod.ai/v2/${ACESTEP_ENDPOINT_ID}/health`, {
      headers: getHeaders(),
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) return false;
    const data = await response.json();
    const workers = data.workers;
    if (workers) {
      const ready = workers.ready || 0;
      const running = workers.running || 0;
      return ready > 0 || running > 0;
    }
    return true;
  } catch {
    return false;
  }
}

export async function getPodDiagnostics(): Promise<Record<string, any>> {
  const diagnostics: Record<string, any> = {
    configured: isConfigured(),
    endpointId: ACESTEP_ENDPOINT_ID || null,
    mode: "serverless",
    timestamp: new Date().toISOString(),
  };

  if (!isConfigured()) return diagnostics;

  try {
    const response = await fetch(`https://api.runpod.ai/v2/${ACESTEP_ENDPOINT_ID}/health`, {
      headers: getHeaders(),
      signal: AbortSignal.timeout(10000),
    });
    if (response.ok) {
      diagnostics.health = await response.json();
    } else {
      diagnostics.health = { status: "unreachable", code: response.status };
    }
  } catch (e: any) {
    diagnostics.health = { error: e.message };
  }

  return diagnostics;
}

function buildCaption(params: {
  prompt: string;
  style?: string;
  instrument?: string;
}): string {
  const parts: string[] = [];
  if (params.style) parts.push(params.style);
  parts.push(params.prompt);
  if (params.instrument) parts.push(params.instrument);
  return parts.join(", ");
}
