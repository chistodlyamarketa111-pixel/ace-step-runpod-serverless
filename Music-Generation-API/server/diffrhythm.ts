import { log } from "./index";

const DIFFRHYTHM_ENDPOINT_ID = process.env.DIFFRHYTHM_ENDPOINT_ID;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;

if (!DIFFRHYTHM_ENDPOINT_ID) {
  console.warn("DIFFRHYTHM_ENDPOINT_ID not set - DiffRhythm integration will not work");
}

function getBaseUrl(): string {
  return `https://api.runpod.ai/v2/${DIFFRHYTHM_ENDPOINT_ID}`;
}

function getHeaders(): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${RUNPOD_API_KEY}`,
  };
}

export function isConfigured(): boolean {
  return Boolean(DIFFRHYTHM_ENDPOINT_ID) && Boolean(RUNPOD_API_KEY);
}

export async function submitTask(params: {
  prompt: string;
  lyrics?: string;
  duration?: number;
  style?: string;
  seed?: number;
}): Promise<{ task_id: string }> {
  if (!isConfigured()) {
    throw new Error("DiffRhythm Serverless is not configured. Set DIFFRHYTHM_ENDPOINT_ID and RUNPOD_API_KEY.");
  }

  const prompt = params.style
    ? `${params.style}. ${params.prompt}`
    : params.prompt;

  const payload = {
    input: {
      mode: "generate",
      prompt,
      lyrics: params.lyrics || "",
      duration: params.duration || 95,
      seed: params.seed ?? -1,
      fp16: true,
      chunked: true,
    },
  };

  log(`DiffRhythm serverless submit: prompt=${prompt.slice(0, 80)}, duration=${payload.input.duration}`, "diffrhythm");

  const maxRetries = 3;
  let response: Response | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      response = await fetch(`${getBaseUrl()}/run`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30000),
      });

      if (response.ok) break;

      if (response.status === 429 || response.status === 502 || response.status === 503) {
        log(`DiffRhythm submit attempt ${attempt}/${maxRetries} got ${response.status}, retrying in ${attempt * 5}s...`, "diffrhythm");
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, attempt * 5000));
          continue;
        }
      }

      const text = await response.text();
      log(`DiffRhythm submit error: ${response.status} ${text.slice(0, 200)}`, "diffrhythm");
      throw new Error(`DiffRhythm serverless error (${response.status}): ${text.slice(0, 200)}`);
    } catch (e: any) {
      if (e.message.includes("DiffRhythm serverless error")) throw e;
      if (attempt < maxRetries) {
        log(`DiffRhythm submit attempt ${attempt} failed: ${e.message}, retrying...`, "diffrhythm");
        await new Promise(r => setTimeout(r, attempt * 5000));
        continue;
      }
      throw new Error(`DiffRhythm serverless not reachable after ${maxRetries} attempts: ${e.message}`);
    }
  }

  if (!response || !response.ok) {
    throw new Error("DiffRhythm serverless unavailable after retries");
  }

  const result = await response.json() as any;
  log(`DiffRhythm submit response: id=${result.id}, status=${result.status}`, "diffrhythm");

  const jobId = result.id;
  if (!jobId) {
    throw new Error("RunPod did not return a job id");
  }

  return { task_id: `dr_${jobId}` };
}

export async function submitPipeline(params: {
  prompt: string;
  lyrics?: string;
  duration?: number;
  style?: string;
  seed?: number;
  vol_vocals?: number;
  vol_drums?: number;
  vol_bass?: number;
  vol_other?: number;
  demucs_model?: string;
}): Promise<{ task_id: string }> {
  if (!isConfigured()) {
    throw new Error("DiffRhythm Serverless is not configured. Set DIFFRHYTHM_ENDPOINT_ID and RUNPOD_API_KEY.");
  }

  const prompt = params.style
    ? `${params.style}. ${params.prompt}`
    : params.prompt;

  const payload = {
    input: {
      mode: "pipeline",
      prompt,
      lyrics: params.lyrics || "",
      duration: params.duration || 95,
      seed: params.seed ?? -1,
      fp16: true,
      chunked: true,
      vol_vocals: params.vol_vocals ?? 1.0,
      vol_drums: params.vol_drums ?? 1.0,
      vol_bass: params.vol_bass ?? 1.0,
      vol_other: params.vol_other ?? 1.0,
      demucs_model: params.demucs_model || "htdemucs",
    },
  };

  log(`DiffRhythm pipeline serverless submit: prompt=${prompt.slice(0, 80)}`, "diffrhythm-pp");

  const maxRetries = 3;
  let response: Response | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      response = await fetch(`${getBaseUrl()}/run`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30000),
      });

      if (response.ok) break;

      if (response.status === 429 || response.status === 502 || response.status === 503) {
        log(`DiffRhythm pipeline attempt ${attempt}/${maxRetries} got ${response.status}, retrying...`, "diffrhythm-pp");
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, attempt * 5000));
          continue;
        }
      }

      const text = await response.text();
      throw new Error(`DiffRhythm pipeline serverless error (${response.status}): ${text.slice(0, 200)}`);
    } catch (e: any) {
      if (e.message.includes("serverless error")) throw e;
      if (attempt < maxRetries) {
        log(`DiffRhythm pipeline attempt ${attempt} failed: ${e.message}, retrying...`, "diffrhythm-pp");
        await new Promise(r => setTimeout(r, attempt * 5000));
        continue;
      }
      throw new Error(`DiffRhythm pipeline serverless not reachable after ${maxRetries} attempts: ${e.message}`);
    }
  }

  if (!response || !response.ok) {
    throw new Error("DiffRhythm pipeline serverless unavailable after retries");
  }

  const result = await response.json() as any;
  log(`DiffRhythm pipeline response: id=${result.id}, status=${result.status}`, "diffrhythm-pp");

  const jobId = result.id;
  if (!jobId) {
    throw new Error("RunPod did not return a job id");
  }

  return { task_id: `drpp_${jobId}` };
}

export async function queryTaskStatus(taskId: string): Promise<{
  status: string;
  audio_base64?: string;
  content_type?: string;
  error?: string;
  logs?: string;
}> {
  if (!isConfigured()) {
    return { status: "FAILED", error: "DiffRhythm not configured" };
  }

  const jobId = taskId.startsWith("dr_")
    ? taskId.slice(3)
    : taskId.startsWith("drpp_")
      ? taskId.slice(5)
      : taskId;

  try {
    const response = await fetch(`${getBaseUrl()}/status/${jobId}`, {
      headers: getHeaders(),
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      if (response.status === 404) {
        return { status: "FAILED", error: "Job not found on RunPod" };
      }
      if (response.status === 429 || response.status === 502 || response.status === 503) {
        log(`RunPod temporarily unavailable (${response.status}), treating as in-progress`, "diffrhythm");
        return { status: "IN_PROGRESS", logs: `RunPod busy (${response.status}), generation likely in progress...` };
      }
      return { status: "FAILED", error: `Status check failed: ${response.status}` };
    }

    const data = await response.json() as any;

    if (data.status === "COMPLETED") {
      const output = data.output;
      if (output?.error) {
        return { status: "FAILED", error: output.error, logs: output.logs };
      }
      return {
        status: "COMPLETED",
        audio_base64: output?.audio_base64,
        content_type: output?.content_type || "audio/wav",
        logs: output?.logs,
      };
    }

    if (data.status === "FAILED") {
      return {
        status: "FAILED",
        error: data.error || "Generation failed on RunPod",
      };
    }

    const statusMap: Record<string, string> = {
      "IN_QUEUE": "IN_PROGRESS",
      "IN_PROGRESS": "IN_PROGRESS",
    };

    return {
      status: statusMap[data.status] || "IN_PROGRESS",
      logs: data.status === "IN_QUEUE" ? "Waiting for GPU worker to start..." : "Generating...",
    };
  } catch (e: any) {
    log(`DiffRhythm status check error: ${e.message}`, "diffrhythm");
    return { status: "IN_PROGRESS", logs: `Status check temporarily unavailable: ${e.message}` };
  }
}

export async function fetchAudio(audioIdentifier: string): Promise<{ buffer: Buffer; contentType: string }> {
  if (!isConfigured()) {
    throw new Error("DiffRhythm Serverless is not configured.");
  }

  if (audioIdentifier.startsWith("data:") || audioIdentifier.length > 1000) {
    const buffer = Buffer.from(audioIdentifier, "base64");
    return { buffer, contentType: "audio/wav" };
  }

  const jobId = audioIdentifier.startsWith("dr_")
    ? audioIdentifier.slice(3)
    : audioIdentifier.startsWith("drpp_")
      ? audioIdentifier.slice(5)
      : audioIdentifier;

  const response = await fetch(`${getBaseUrl()}/status/${jobId}`, {
    headers: getHeaders(),
    signal: AbortSignal.timeout(30000),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch DiffRhythm audio: RunPod status ${response.status}`);
  }

  const data = await response.json() as any;

  if (data.status !== "COMPLETED" || !data.output?.audio_base64) {
    throw new Error(`Job ${jobId} is not completed or has no audio output`);
  }

  const buffer = Buffer.from(data.output.audio_base64, "base64");
  const contentType = data.output.content_type || "audio/wav";

  return { buffer, contentType };
}

export async function checkHealth(): Promise<boolean> {
  if (!isConfigured()) return false;

  try {
    const response = await fetch(`${getBaseUrl()}/health`, {
      headers: getHeaders(),
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) return false;
    return true;
  } catch {
    return false;
  }
}

export async function getPodDiagnostics(): Promise<Record<string, any>> {
  const diagnostics: Record<string, any> = {
    configured: isConfigured(),
    endpointId: DIFFRHYTHM_ENDPOINT_ID || null,
    mode: "serverless",
    baseUrl: isConfigured() ? getBaseUrl() : null,
    timestamp: new Date().toISOString(),
  };

  if (!isConfigured()) return diagnostics;

  try {
    const res = await fetch(`${getBaseUrl()}/health`, {
      headers: getHeaders(),
      signal: AbortSignal.timeout(10000),
    });
    if (res.ok) {
      diagnostics.health = await res.json();
    } else {
      diagnostics.health = { status: "unreachable", code: res.status };
    }
  } catch (e: any) {
    diagnostics.health = { error: e.message };
  }

  return diagnostics;
}
