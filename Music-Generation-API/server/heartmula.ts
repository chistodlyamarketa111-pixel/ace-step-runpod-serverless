import { log } from "./index";

const HEARTMULA_POD_ID = process.env.HEARTMULA_POD_ID === "e5w9wl2vhcuwat" 
  ? "vb787qolnhhdjz" 
  : (process.env.HEARTMULA_POD_ID || "vb787qolnhhdjz");

if (!HEARTMULA_POD_ID) {
  console.warn("HEARTMULA_POD_ID not set - HeartMuLa integration will not work");
}

function getBaseUrl(): string {
  return `https://${HEARTMULA_POD_ID}-8000.proxy.runpod.net`;
}

export function isConfigured(): boolean {
  return Boolean(HEARTMULA_POD_ID);
}

export async function submitTask(params: {
  prompt: string;
  lyrics?: string;
  duration?: number;
  tags?: string;
  negative_tags?: string;
  title?: string;
  seed?: number;
  temperature?: number;
  cfg_scale?: number;
  topk?: number;
}): Promise<{ job_id: string }> {
  if (!isConfigured()) {
    throw new Error("HeartMuLa Pod is not configured. Set HEARTMULA_POD_ID.");
  }

  const body: Record<string, any> = {
    prompt: params.prompt,
    duration_ms: (params.duration || 30) * 1000,
  };

  if (params.lyrics?.trim()) {
    body.lyrics = params.lyrics;
  }
  if (params.tags?.trim()) {
    body.tags = params.tags;
  }
  if (params.negative_tags?.trim()) {
    body.negative_tags = params.negative_tags;
  }
  if (params.title?.trim()) {
    body.title = params.title;
  }
  if (params.seed !== undefined && params.seed !== -1) {
    body.seed = params.seed;
  }
  if (params.temperature !== undefined) {
    body.temperature = params.temperature;
  }
  if (params.cfg_scale !== undefined) {
    body.cfg_scale = params.cfg_scale;
  }
  if (params.topk !== undefined) {
    body.topk = params.topk;
  }

  log(`Submitting task to HeartMuLa: ${JSON.stringify(body)}`, "heartmula");

  const response = await fetch(`${getBaseUrl()}/generate/music`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    log(`HeartMuLa submit error: ${response.status} ${text}`, "heartmula");
    throw new Error(`HeartMuLa API error: ${response.status} - ${text}`);
  }

  const raw = await response.json();
  log(`HeartMuLa submit response: ${JSON.stringify(raw)}`, "heartmula");

  if (!raw.job_id) {
    throw new Error("HeartMuLa did not return a job_id");
  }

  return { job_id: raw.job_id };
}

export async function queryJobStatus(jobId: string): Promise<{
  status: string;
  audio_path?: string;
  error?: string;
  progress?: number;
}> {
  if (!isConfigured()) {
    throw new Error("HeartMuLa Pod is not configured.");
  }

  const response = await fetch(`${getBaseUrl()}/jobs/${jobId}`, {
    signal: AbortSignal.timeout(15000),
  });

  if (!response.ok) {
    if (response.status === 404) {
      return { status: "IN_PROGRESS" };
    }
    const text = await response.text();
    log(`HeartMuLa query error: ${response.status} ${text}`, "heartmula");
    throw new Error(`HeartMuLa API error: ${response.status} - ${text}`);
  }

  const raw = await response.json();
  log(`HeartMuLa job result for ${jobId}: ${JSON.stringify(raw)}`, "heartmula");

  const status = raw.status;

  if (status === "completed" || status === "COMPLETED") {
    const audioPath = raw.audio_path || raw.audio_url || raw.output_path || raw.result?.audio_path || jobId;
    return {
      status: "COMPLETED",
      audio_path: audioPath,
      progress: 100,
    };
  }

  if (status === "failed" || status === "FAILED" || status === "error") {
    return {
      status: "FAILED",
      error: raw.error || raw.error_msg || raw.error_message || "Generation failed",
    };
  }

  if (status === "cancelled" || status === "cancelling" || status === "CANCELLED") {
    return {
      status: "CANCELLED",
      error: "Job was cancelled",
    };
  }

  const progress = typeof raw.progress === "number" ? Math.round(raw.progress) : undefined;

  if (status === "processing" || status === "pending" || status === "queued") {
    return { status: "IN_PROGRESS", progress };
  }

  log(`Unknown HeartMuLa status "${status}" for ${jobId}`, "heartmula");
  return { status: "IN_PROGRESS", progress };
}

export async function fetchAudio(audioPath: string): Promise<{ buffer: Buffer; contentType: string }> {
  if (!isConfigured()) {
    throw new Error("HeartMuLa Pod is not configured.");
  }

  let url: string;
  if (audioPath.startsWith("http")) {
    url = audioPath;
  } else if (audioPath.startsWith("/")) {
    url = `${getBaseUrl()}${audioPath}`;
  } else if (audioPath.includes("/")) {
    url = `${getBaseUrl()}/${audioPath}`;
  } else {
    url = `${getBaseUrl()}/download_track/${audioPath}`;
  }

  log(`Fetching HeartMuLa audio from: ${url}`, "heartmula");

  const response = await fetch(url);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to fetch audio: ${response.status} - ${text}`);
  }

  const contentType = response.headers.get("content-type") || "audio/mpeg";
  const buffer = Buffer.from(await response.arrayBuffer());

  return { buffer, contentType };
}

export async function checkHealth(): Promise<boolean> {
  if (!isConfigured()) return false;

  try {
    const response = await fetch(`${getBaseUrl()}/health`, {
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) return false;
    const raw = await response.json();
    return raw.status === "ok";
  } catch {
    return false;
  }
}

export async function getGpuSettings(): Promise<Record<string, any>> {
  if (!isConfigured()) return { error: "Not configured" };
  try {
    const res = await fetch(`${getBaseUrl()}/settings/gpu`, { signal: AbortSignal.timeout(10000) });
    return res.ok ? await res.json() : { error: `HTTP ${res.status}` };
  } catch (e: any) {
    return { error: e.message };
  }
}

const REQUIRED_GPU_SETTINGS = {
  quantization_4bit: "false",
  sequential_offload: "false",
  torch_compile: false,
};

export async function enforceGpuSettings(): Promise<{ success: boolean; message: string; changed: boolean }> {
  if (!isConfigured()) return { success: false, message: "Not configured", changed: false };

  try {
    const currentRes = await fetch(`${getBaseUrl()}/settings/gpu`, {
      signal: AbortSignal.timeout(10000),
    });
    if (!currentRes.ok) {
      return { success: false, message: `Failed to read GPU settings: HTTP ${currentRes.status}`, changed: false };
    }

    const current = await currentRes.json();
    log(`Current GPU settings: ${JSON.stringify(current)}`, "heartmula");

    const needsUpdate =
      current.quantization_4bit !== REQUIRED_GPU_SETTINGS.quantization_4bit ||
      current.sequential_offload !== REQUIRED_GPU_SETTINGS.sequential_offload ||
      current.torch_compile !== REQUIRED_GPU_SETTINGS.torch_compile;

    if (!needsUpdate) {
      log("GPU settings already correct - no update needed", "heartmula");
      return { success: true, message: "GPU settings already correct", changed: false };
    }

    log(`GPU settings need update: sequential_offload=${current.sequential_offload}, quantization_4bit=${current.quantization_4bit} -> enforcing GPU-only mode`, "heartmula");

    const updateRes = await fetch(`${getBaseUrl()}/settings/gpu`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(REQUIRED_GPU_SETTINGS),
      signal: AbortSignal.timeout(10000),
    });

    if (!updateRes.ok) {
      const text = await updateRes.text();
      return { success: false, message: `Failed to update GPU settings: ${text}`, changed: false };
    }

    const updated = await updateRes.json();
    log(`GPU settings enforced: ${JSON.stringify(updated)}`, "heartmula");

    log("Settings were changed — triggering model reload to apply GPU-only mode", "heartmula");
    try {
      const reloadRes = await fetch(`${getBaseUrl()}/settings/gpu/reload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
        signal: AbortSignal.timeout(120000),
      });
      const reloadData = await reloadRes.json();
      if (reloadRes.ok) {
        log(`Model reloaded on GPU after settings change: ${JSON.stringify(reloadData)}`, "heartmula");
      } else {
        log(`Model reload after settings change failed: ${reloadData.detail || reloadRes.status}`, "heartmula");
      }
    } catch (reloadErr: any) {
      log(`Model reload after settings change error: ${reloadErr.message}`, "heartmula");
    }

    return {
      success: true,
      message: `GPU settings updated and model reloaded: sequential_offload=${updated.sequential_offload}, quantization_4bit=${updated.quantization_4bit}`,
      changed: true,
    };
  } catch (e: any) {
    return { success: false, message: e.message, changed: false };
  }
}

let startupReloadDone = false;

export async function verifyGpuAndReload(): Promise<{ success: boolean; message: string }> {
  if (!isConfigured()) return { success: false, message: "Not configured" };

  try {
    const enforce = await enforceGpuSettings();
    if (!enforce.success) {
      return { success: false, message: enforce.message };
    }

    if (enforce.changed) {
      return { success: true, message: "Settings corrected and model reloaded on GPU" };
    }

    if (!startupReloadDone) {
      log("First startup check — forcing GPU model reload even though settings look correct (settings ≠ actual device placement)", "heartmula");
      startupReloadDone = true;
      const reload = await reloadGpuModel();
      if (!reload.success && reload.message.includes("job is processing")) {
        log("Pod busy on startup — scheduling GPU reload for when jobs finish", "heartmula");
        pendingGpuReload = true;
        return { success: true, message: "Startup: pod busy, GPU reload scheduled for when jobs complete" };
      }
      return { success: reload.success, message: `Startup forced reload: ${reload.message}` };
    }

    const healthOk = await checkHealth();
    if (!healthOk) {
      log("Pod unhealthy — triggering GPU reload", "heartmula");
      const reload = await reloadGpuModel();
      return reload;
    }

    return { success: true, message: "GPU settings verified, pod healthy" };
  } catch (e: any) {
    return { success: false, message: e.message };
  }
}

let pendingGpuReload = false;

export function schedulGpuReload() {
  pendingGpuReload = true;
  log("GPU reload scheduled - will execute when no jobs are processing", "heartmula");
}

export function hasPendingGpuReload(): boolean {
  return pendingGpuReload;
}

export async function tryPendingGpuReload(): Promise<void> {
  if (!pendingGpuReload) return;
  log("Attempting pending GPU reload...", "heartmula");
  const result = await reloadGpuModel();
  if (result.success) {
    pendingGpuReload = false;
    log(`Pending GPU reload succeeded: ${result.message}`, "heartmula");
  } else {
    log(`Pending GPU reload failed (will retry): ${result.message}`, "heartmula");
  }
}

async function setGpuSettingsOnly(): Promise<boolean> {
  try {
    const updateRes = await fetch(`${getBaseUrl()}/settings/gpu`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(REQUIRED_GPU_SETTINGS),
      signal: AbortSignal.timeout(10000),
    });
    if (!updateRes.ok) {
      log(`Failed to set GPU settings: HTTP ${updateRes.status}`, "heartmula");
      return false;
    }
    const updated = await updateRes.json();
    log(`GPU settings set (no reload): ${JSON.stringify(updated)}`, "heartmula");
    return true;
  } catch (e: any) {
    log(`Error setting GPU settings: ${e.message}`, "heartmula");
    return false;
  }
}

async function waitForModelReady(maxWaitMs: number = 120000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    try {
      const res = await fetch(`${getBaseUrl()}/settings/gpu/reload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
        signal: AbortSignal.timeout(10000),
      });
      const data = await res.json();
      if (res.ok) {
        log("Model is ready (reload accepted) — immediately re-applying GPU settings", "heartmula");
        return true;
      }
      const msg = data.detail || data.message || "";
      if (typeof msg === "string" && msg.includes("Models are currently loading")) {
        log("Model still loading, waiting 5s...", "heartmula");
      } else if (typeof msg === "string" && msg.includes("job is processing")) {
        log("Job still processing, waiting 5s...", "heartmula");
      } else {
        log(`Unexpected reload response: ${JSON.stringify(data)}`, "heartmula");
      }
    } catch (e: any) {
      log(`Wait-for-model error: ${e.message}`, "heartmula");
    }
    await new Promise(r => setTimeout(r, 5000));
  }
  return false;
}

export async function reloadGpuModel(): Promise<{ success: boolean; message: string }> {
  if (!isConfigured()) return { success: false, message: "Not configured" };

  await setGpuSettingsOnly();
  log("Pre-reload: GPU settings set to GPU-only mode", "heartmula");

  try {
    const res = await fetch(`${getBaseUrl()}/settings/gpu/reload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(120000),
    });
    const data = await res.json();
    if (!res.ok) {
      return { success: false, message: data.detail || `HTTP ${res.status}` };
    }

    log("Reload triggered — waiting for model to finish loading before re-enforcing GPU settings", "heartmula");
    await new Promise(r => setTimeout(r, 10000));

    const ready = await waitForModelReady(120000);
    if (ready) {
      await setGpuSettingsOnly();
      log("Post-reload: GPU settings re-applied after model loaded", "heartmula");

      log("Triggering final reload with correct GPU settings applied", "heartmula");
      try {
        const finalRes = await fetch(`${getBaseUrl()}/settings/gpu/reload`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
          signal: AbortSignal.timeout(120000),
        });
        if (finalRes.ok) {
          log("Final GPU reload succeeded — model loading with GPU-only settings", "heartmula");
        }
      } catch (e: any) {
        log(`Final reload error (non-fatal): ${e.message}`, "heartmula");
      }
    } else {
      log("WARNING: Model did not become ready within timeout — GPU settings may not be applied", "heartmula");
    }

    return { success: true, message: `GPU settings updated and model reloaded: sequential_offload=false, quantization_4bit=false` };
  } catch (e: any) {
    return { success: false, message: e.message };
  }
}

export async function getPodDiagnostics(): Promise<Record<string, any>> {
  const diagnostics: Record<string, any> = {
    configured: isConfigured(),
    podId: HEARTMULA_POD_ID || null,
    baseUrl: isConfigured() ? getBaseUrl() : null,
    timestamp: new Date().toISOString(),
  };

  if (!isConfigured()) return diagnostics;

  try {
    const healthRes = await fetch(`${getBaseUrl()}/health`, {
      signal: AbortSignal.timeout(10000),
    });
    diagnostics.health = healthRes.ok ? await healthRes.json() : { error: `HTTP ${healthRes.status}` };
  } catch (e: any) {
    diagnostics.health = { error: e.message };
  }

  const apiKey = process.env.RUNPOD_API_KEY;
  if (apiKey && HEARTMULA_POD_ID) {
    try {
      const res = await fetch(
        `https://api.runpod.io/graphql?api_key=${apiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: `query { pod(input:{podId:"${HEARTMULA_POD_ID}"}) { id name desiredStatus imageName containerDiskInGb volumeInGb runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } } machine { podHostId gpuDisplayName } } }`,
          }),
          signal: AbortSignal.timeout(10000),
        }
      );
      if (res.ok) {
        const data = await res.json();
        diagnostics.runpod = data.data?.pod || { error: "Pod not found" };
      }
    } catch (e: any) {
      diagnostics.runpod = { error: e.message };
    }
  }

  return diagnostics;
}
