import { log } from "./index";

const DIFFRHYTHM_POD_ID = process.env.DIFFRHYTHM_POD_ID;
const DIFFRHYTHM_API_PORT = process.env.DIFFRHYTHM_API_PORT || "8000";

if (!DIFFRHYTHM_POD_ID) {
  console.warn("DIFFRHYTHM_POD_ID not set - DiffRhythm integration will not work");
}

function getBaseUrl(): string {
  return `https://${DIFFRHYTHM_POD_ID}-${DIFFRHYTHM_API_PORT}.proxy.runpod.net`;
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
    throw new Error("DiffRhythm Pod is not configured. Set DIFFRHYTHM_POD_ID.");
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

      if (response.status === 502 || response.status === 503 || response.status === 504) {
        log(`DiffRhythm submit attempt ${attempt}/${maxRetries} got ${response.status}, retrying in ${attempt * 10}s...`, "diffrhythm");
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, attempt * 10000));
          continue;
        }
      }

      const text = await response.text();
      log(`DiffRhythm submit error: ${response.status} ${text.slice(0, 200)}`, "diffrhythm");
      throw new Error(`DiffRhythm pod unavailable (${response.status}). Ensure the pod is running.`);
    } catch (e: any) {
      if (e.message.includes("DiffRhythm pod unavailable")) throw e;
      if (attempt < maxRetries) {
        log(`DiffRhythm submit attempt ${attempt} failed: ${e.message}, retrying...`, "diffrhythm");
        await new Promise(r => setTimeout(r, attempt * 10000));
        continue;
      }
      throw new Error(`DiffRhythm pod not reachable after ${maxRetries} attempts: ${e.message}`);
    }
  }

  if (!response || !response.ok) {
    throw new Error("DiffRhythm pod unavailable after retries");
  }

  const result = await response.json() as any;
  log(`DiffRhythm submit response: ${JSON.stringify(result)}`, "diffrhythm");

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
  reference_path?: string;
  vol_vocals?: number;
  vol_drums?: number;
  vol_bass?: number;
  vol_other?: number;
  demucs_model?: string;
}): Promise<{ task_id: string }> {
  if (!isConfigured()) {
    throw new Error("DiffRhythm Pod is not configured. Set DIFFRHYTHM_POD_ID.");
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
    reference_path: params.reference_path,
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

      if (response.status === 502 || response.status === 503 || response.status === 504) {
        log(`DiffRhythm pipeline attempt ${attempt}/${maxRetries} got ${response.status}, retrying...`, "diffrhythm-pp");
        if (attempt < maxRetries) {
          await new Promise(r => setTimeout(r, attempt * 10000));
          continue;
        }
      }

      const text = await response.text();
      throw new Error(`DiffRhythm pipeline pod unavailable (${response.status}): ${text.slice(0, 200)}`);
    } catch (e: any) {
      if (e.message.includes("pod unavailable")) throw e;
      if (attempt < maxRetries) {
        log(`DiffRhythm pipeline attempt ${attempt} failed: ${e.message}, retrying...`, "diffrhythm-pp");
        await new Promise(r => setTimeout(r, attempt * 10000));
        continue;
      }
      throw new Error(`DiffRhythm pipeline pod not reachable after ${maxRetries} attempts: ${e.message}`);
    }
  }

  if (!response || !response.ok) {
    throw new Error("DiffRhythm pipeline pod unavailable after retries");
  }

  const result = await response.json() as any;
  log(`DiffRhythm pipeline response: ${JSON.stringify(result)}`, "diffrhythm-pp");

  const jobId = result.job_id;
  if (!jobId) {
    throw new Error("DiffRhythm pipeline did not return a job_id");
  }

  return { task_id: `drpp_${jobId}` };
}

export async function queryTaskStatus(taskId: string): Promise<{
  status: string;
  audio_path?: string;
  error?: string;
  logs?: string;
  stems?: Record<string, string>;
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
        return { status: "FAILED", error: "Job not found on DiffRhythm pod" };
      }
      if (response.status === 502 || response.status === 503 || response.status === 504) {
        log(`DiffRhythm pod temporarily unavailable (${response.status}), treating as in-progress`, "diffrhythm");
        return { status: "IN_PROGRESS", logs: `Pod busy (${response.status}), generation likely in progress...` };
      }
      return { status: "FAILED", error: `Status check failed: ${response.status}` };
    }

    const data = await response.json() as any;

    if (data.status === "COMPLETED" && data.output_files?.length > 0) {
      return {
        status: "COMPLETED",
        audio_path: data.output_files[0],
        logs: data.logs,
        stems: data.stems,
      };
    }

    if (data.status === "FAILED") {
      return {
        status: "FAILED",
        error: data.error || "Generation failed",
        logs: data.logs,
      };
    }

    return {
      status: "IN_PROGRESS",
      logs: data.logs,
    };
  } catch (e: any) {
    log(`DiffRhythm status check error: ${e.message}`, "diffrhythm");
    return { status: "IN_PROGRESS", logs: `Status check temporarily unavailable: ${e.message}` };
  }
}

export async function fetchAudio(audioPath: string): Promise<{ buffer: Buffer; contentType: string }> {
  if (!isConfigured()) {
    throw new Error("DiffRhythm Pod is not configured.");
  }

  let url: string;
  if (audioPath.startsWith("http")) {
    url = audioPath;
  } else {
    const encodedPath = encodeURIComponent(audioPath);
    url = `${getBaseUrl()}/download/${encodedPath}`;
  }

  log(`Fetching DiffRhythm audio from: ${url}`, "diffrhythm");

  const response = await fetch(url, {
    signal: AbortSignal.timeout(120000),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to fetch DiffRhythm audio: ${response.status} - ${text}`);
  }

  const contentType = response.headers.get("content-type") || "audio/wav";
  const buffer = Buffer.from(await response.arrayBuffer());

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
    apiPort: DIFFRHYTHM_API_PORT,
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

  try {
    const res = await fetch(`${getBaseUrl()}/jobs`, {
      signal: AbortSignal.timeout(10000),
    });
    if (res.ok) {
      const data = await res.json() as any;
      diagnostics.recentJobs = data.jobs?.length || 0;
    }
  } catch {}

  const apiKey = process.env.RUNPOD_API_KEY;
  if (apiKey && DIFFRHYTHM_POD_ID) {
    try {
      const res = await fetch(
        `https://api.runpod.io/graphql?api_key=${apiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: `query { pod(input:{podId:"${DIFFRHYTHM_POD_ID}"}) { id name desiredStatus imageName runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } } machine { gpuDisplayName } } }`,
          }),
          signal: AbortSignal.timeout(10000),
        }
      );
      if (res.ok) {
        const data = await res.json() as any;
        diagnostics.runpod = data.data?.pod || { error: "Pod not found" };
      }
    } catch (e: any) {
      diagnostics.runpod = { error: e.message };
    }
  }

  return diagnostics;
}
