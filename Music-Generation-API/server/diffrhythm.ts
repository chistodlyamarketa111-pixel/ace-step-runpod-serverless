import { log } from "./index";

const DIFFRHYTHM_POD_ID = process.env.DIFFRHYTHM_POD_ID;

if (!DIFFRHYTHM_POD_ID) {
  console.warn("DIFFRHYTHM_POD_ID not set - DiffRhythm integration will not work");
}

function getBaseUrl(): string {
  return `https://${DIFFRHYTHM_POD_ID}-8000.proxy.runpod.net`;
}

export function isConfigured(): boolean {
  return Boolean(DIFFRHYTHM_POD_ID);
}

export async function submitTask(params: {
  prompt: string;
  lyrics?: string;
  duration?: number;
  style?: string;
  seed?: number;
}): Promise<{ task_id: string }> {
  if (!isConfigured()) {
    throw new Error("DiffRhythm is not configured. Set DIFFRHYTHM_POD_ID.");
  }

  const prompt = params.style
    ? `${params.style}. ${params.prompt}`
    : params.prompt;

  const payload = {
    prompt,
    lyrics: params.lyrics || "",
    duration: params.duration || 95,
    seed: params.seed ?? -1,
    fp16: true,
    chunked: true,
  };

  log(`DiffRhythm submit: prompt=${prompt.slice(0, 80)}, duration=${payload.duration}`, "diffrhythm");

  const maxRetries = 3;
  let response: Response | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      response = await fetch(`${getBaseUrl()}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      throw new Error(`DiffRhythm error (${response.status}): ${text.slice(0, 200)}`);
    } catch (e: any) {
      if (e.message.includes("DiffRhythm error")) throw e;
      if (attempt < maxRetries) {
        log(`DiffRhythm submit attempt ${attempt} failed: ${e.message}, retrying...`, "diffrhythm");
        await new Promise(r => setTimeout(r, attempt * 5000));
        continue;
      }
      throw new Error(`DiffRhythm not reachable after ${maxRetries} attempts: ${e.message}`);
    }
  }

  if (!response || !response.ok) {
    throw new Error("DiffRhythm unavailable after retries");
  }

  const result = await response.json() as any;
  log(`DiffRhythm submit response: job_id=${result.job_id}, status=${result.status}`, "diffrhythm");

  const jobId = result.job_id;
  if (!jobId) {
    throw new Error("DiffRhythm did not return a job_id");
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
    throw new Error("DiffRhythm is not configured. Set DIFFRHYTHM_POD_ID.");
  }

  const prompt = params.style
    ? `${params.style}. ${params.prompt}`
    : params.prompt;

  const payload = {
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
  };

  log(`DiffRhythm pipeline submit: prompt=${prompt.slice(0, 80)}`, "diffrhythm-pp");

  const maxRetries = 3;
  let response: Response | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      response = await fetch(`${getBaseUrl()}/pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      throw new Error(`DiffRhythm pipeline error (${response.status}): ${text.slice(0, 200)}`);
    } catch (e: any) {
      if (e.message.includes("pipeline error")) throw e;
      if (attempt < maxRetries) {
        log(`DiffRhythm pipeline attempt ${attempt} failed: ${e.message}, retrying...`, "diffrhythm-pp");
        await new Promise(r => setTimeout(r, attempt * 5000));
        continue;
      }
      throw new Error(`DiffRhythm pipeline not reachable after ${maxRetries} attempts: ${e.message}`);
    }
  }

  if (!response || !response.ok) {
    throw new Error("DiffRhythm pipeline unavailable after retries");
  }

  const result = await response.json() as any;
  log(`DiffRhythm pipeline response: job_id=${result.job_id}, status=${result.status}`, "diffrhythm-pp");

  const jobId = result.job_id;
  if (!jobId) {
    throw new Error("DiffRhythm did not return a job_id");
  }

  return { task_id: `drpp_${jobId}` };
}

export async function queryTaskStatus(taskId: string): Promise<{
  status: string;
  output_files?: string[];
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
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      if (response.status === 404) {
        return { status: "FAILED", error: "Job not found" };
      }
      return { status: "FAILED", error: `Status check failed: ${response.status}` };
    }

    const data = await response.json() as any;

    if (data.status === "COMPLETED") {
      if (data.error) {
        return { status: "FAILED", error: data.error, logs: data.logs };
      }
      return {
        status: "COMPLETED",
        output_files: data.output_files || [],
        logs: data.logs,
      };
    }

    if (data.status === "FAILED" || data.status === "ERROR") {
      return {
        status: "FAILED",
        error: data.error || "Generation failed",
        logs: data.logs,
      };
    }

    return {
      status: "IN_PROGRESS",
      logs: data.logs || "Generating...",
    };
  } catch (e: any) {
    log(`DiffRhythm status check error: ${e.message}`, "diffrhythm");
    return { status: "IN_PROGRESS", logs: `Status check temporarily unavailable: ${e.message}` };
  }
}

export async function fetchAudio(audioIdentifier: string): Promise<{ buffer: Buffer; contentType: string }> {
  if (!isConfigured()) {
    throw new Error("DiffRhythm is not configured.");
  }

  const jobId = audioIdentifier.startsWith("dr_")
    ? audioIdentifier.slice(3)
    : audioIdentifier.startsWith("drpp_")
      ? audioIdentifier.slice(5)
      : audioIdentifier;

  const statusRes = await fetch(`${getBaseUrl()}/status/${jobId}`, {
    signal: AbortSignal.timeout(15000),
  });

  if (!statusRes.ok) {
    throw new Error(`Failed to get job status: ${statusRes.status}`);
  }

  const data = await statusRes.json() as any;

  if (data.status !== "COMPLETED" || !data.output_files?.length) {
    throw new Error(`Job ${jobId} is not completed or has no output files`);
  }

  const audioFile = data.output_files[0];

  const audioRes = await fetch(`${getBaseUrl()}/download/${encodeURIComponent(audioFile)}`, {
    signal: AbortSignal.timeout(60000),
  });

  if (!audioRes.ok) {
    throw new Error(`Failed to download audio: ${audioRes.status}`);
  }

  const arrayBuffer = await audioRes.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);

  const ext = audioFile.split('.').pop()?.toLowerCase();
  const contentType = ext === "mp3" ? "audio/mpeg" : ext === "wav" ? "audio/wav" : "audio/mpeg";

  return { buffer, contentType };
}

export async function checkHealth(): Promise<boolean> {
  if (!isConfigured()) return false;

  try {
    const response = await fetch(`${getBaseUrl()}/health`, {
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) return false;
    const data = await response.json() as any;
    return data.status === "ok";
  } catch {
    return false;
  }
}

export async function getPodDiagnostics(): Promise<Record<string, any>> {
  const diagnostics: Record<string, any> = {
    configured: isConfigured(),
    podId: DIFFRHYTHM_POD_ID || null,
    mode: "pod",
    baseUrl: isConfigured() ? getBaseUrl() : null,
    timestamp: new Date().toISOString(),
  };

  if (!isConfigured()) return diagnostics;

  try {
    const res = await fetch(`${getBaseUrl()}/health`, {
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
