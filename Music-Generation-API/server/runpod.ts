import { log } from "./index";

const RUNPOD_POD_ID = process.env.RUNPOD_POD_ID;

if (!RUNPOD_POD_ID) {
  console.warn("RUNPOD_POD_ID not set - ACE-Step integration will not work");
}

function getBaseUrl(): string {
  return `https://${RUNPOD_POD_ID}-8000.proxy.runpod.net`;
}

export function isConfigured(): boolean {
  return Boolean(RUNPOD_POD_ID);
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
    throw new Error("RunPod Pod is not configured. Set RUNPOD_POD_ID.");
  }

  const caption = buildCaption(params);

  const body: Record<string, any> = {
    prompt: caption,
    audio_duration: params.duration || 30,
    task_type: "text2music",
    audio_format: params.audio_format || "mp3",
  };

  if (params.lyrics && params.lyrics.trim()) {
    body.lyrics = params.lyrics;
  }
  if (params.bpm !== undefined) {
    body.bpm = params.bpm;
  }
  if (params.key_scale) {
    body.key_scale = params.key_scale;
  }
  if (params.time_signature) {
    body.time_signature = params.time_signature;
  }
  if (params.vocal_language) {
    body.vocal_language = params.vocal_language;
  }
  if (params.inference_steps !== undefined) {
    body.inference_steps = params.inference_steps;
  }
  if (params.guidance_scale !== undefined) {
    body.guidance_scale = params.guidance_scale;
  }
  if (params.thinking !== undefined) {
    body.thinking = params.thinking;
  }
  if (params.shift !== undefined) {
    body.shift = params.shift;
  }
  if (params.infer_method) {
    body.infer_method = params.infer_method;
  }
  if (params.use_adg !== undefined) {
    body.use_adg = params.use_adg;
  }
  if (params.batch_size !== undefined && params.batch_size > 1) {
    body.batch_size = params.batch_size;
  }
  if (params.seed !== undefined && params.seed !== -1) {
    body.seed = params.seed;
    body.use_random_seed = false;
  }
  if (params.model) {
    body.model = params.model;
  }

  log(`Submitting task to ACE-Step: ${JSON.stringify(body)}`, "runpod");

  const response = await fetch(`${getBaseUrl()}/release_task`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    log(`ACE-Step submit error: ${response.status} ${text}`, "runpod");
    throw new Error(`ACE-Step API error: ${response.status} - ${text}`);
  }

  const raw = await response.json();
  log(`ACE-Step submit response: ${JSON.stringify(raw)}`, "runpod");

  const data = raw.data || raw;

  if (!data.task_id) {
    throw new Error("ACE-Step did not return a task_id");
  }

  return { task_id: data.task_id };
}

export async function queryTaskStatus(taskId: string): Promise<{
  status: string;
  audio_path?: string;
  error?: string;
}> {
  if (!isConfigured()) {
    throw new Error("RunPod Pod is not configured.");
  }

  const response = await fetch(`${getBaseUrl()}/query_result`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_id_list: [taskId] }),
  });

  if (!response.ok) {
    const text = await response.text();
    log(`ACE-Step query error: ${response.status} ${text}`, "runpod");
    throw new Error(`ACE-Step API error: ${response.status} - ${text}`);
  }

  const raw = await response.json();
  log(`ACE-Step query result for ${taskId}: ${JSON.stringify(raw)}`, "runpod");

  const dataArray = raw.data || raw;

  if (Array.isArray(dataArray) && dataArray.length === 0) {
    return { status: "IN_PROGRESS" };
  }

  const taskResult = Array.isArray(dataArray)
    ? dataArray.find((d: any) => d.task_id === taskId) || dataArray[0]
    : dataArray;

  if (!taskResult) {
    return { status: "IN_PROGRESS" };
  }

  const status = taskResult.status;

  let innerResult: any = null;
  if (typeof taskResult.result === "string") {
    try {
      innerResult = JSON.parse(taskResult.result);
      if (Array.isArray(innerResult)) {
        innerResult = innerResult[0];
      }
    } catch {
      innerResult = { raw: taskResult.result };
    }
  } else if (taskResult.result) {
    innerResult = taskResult.result;
  }

  if (status === 1 || status === "completed" || status === "done" || status === "success") {
    const audioPath = extractAudioPath(innerResult) || extractAudioPath(taskResult);
    if (!audioPath) {
      log(`Task ${taskId} succeeded but no audio path in: ${JSON.stringify(taskResult)}`, "runpod");
    }
    return { status: "COMPLETED", audio_path: audioPath };
  }

  if (status === 2 || status === "failed" || status === "error") {
    const innerStatus = innerResult?.status;
    if (innerStatus === 2) {
      return {
        status: "FAILED",
        error: innerResult?.error || taskResult.error || "Task failed",
      };
    }
    return {
      status: "FAILED",
      error: taskResult.error || innerResult?.error || "Task failed",
    };
  }

  if (status === 0 || status === "queued" || status === "running" || status === "processing") {
    return { status: "IN_PROGRESS" };
  }

  log(`Unknown task status "${status}" for ${taskId}: ${JSON.stringify(taskResult)}`, "runpod");
  return { status: "IN_PROGRESS" };
}

function extractAudioPath(data: any): string | undefined {
  if (!data) return undefined;
  return data.file
    || data.audio_path
    || data.path
    || data.output_path
    || data.wave
    || undefined;
}

export function getAudioUrl(audioPath: string): string {
  return `${getBaseUrl()}/v1/audio?path=${encodeURIComponent(audioPath)}`;
}

export async function fetchAudio(audioPath: string): Promise<{ buffer: Buffer; contentType: string }> {
  if (!isConfigured()) {
    throw new Error("RunPod Pod is not configured.");
  }

  let url: string;
  if (audioPath.startsWith("http")) {
    url = audioPath;
  } else if (audioPath.startsWith("/v1/audio")) {
    url = `${getBaseUrl()}${audioPath}`;
  } else {
    url = getAudioUrl(audioPath);
  }
  log(`Fetching audio from: ${url}`, "runpod");

  const response = await fetch(url);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to fetch audio: ${response.status} - ${text}`);
  }

  const contentType = response.headers.get("content-type") || "audio/wav";
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
    return raw.data?.status === "ok" || raw.code === 200;
  } catch {
    return false;
  }
}

export async function getPodDiagnostics(): Promise<Record<string, any>> {
  const diagnostics: Record<string, any> = {
    configured: isConfigured(),
    podId: RUNPOD_POD_ID || null,
    baseUrl: isConfigured() ? getBaseUrl() : null,
    timestamp: new Date().toISOString(),
  };

  if (!isConfigured()) return diagnostics;

  const fetchJson = async (path: string, method = "GET", body?: any) => {
    try {
      const opts: RequestInit = {
        method,
        signal: AbortSignal.timeout(10000),
      };
      if (body) {
        opts.headers = { "Content-Type": "application/json" };
        opts.body = JSON.stringify(body);
      }
      const res = await fetch(`${getBaseUrl()}${path}`, opts);
      if (!res.ok) return { error: `HTTP ${res.status}`, body: await res.text() };
      return await res.json();
    } catch (e: any) {
      return { error: e.message };
    }
  };

  const [health, stats, models, runpodInfo] = await Promise.all([
    fetchJson("/health"),
    fetchJson("/v1/stats"),
    fetchJson("/v1/models"),
    getPodInfoFromRunPod(),
  ]);

  diagnostics.health = health;
  diagnostics.stats = stats;
  diagnostics.models = models;
  diagnostics.runpod = runpodInfo;

  return diagnostics;
}

async function getPodInfoFromRunPod(): Promise<Record<string, any>> {
  const apiKey = process.env.RUNPOD_API_KEY;
  if (!apiKey || !RUNPOD_POD_ID) return { error: "Missing API key or Pod ID" };

  try {
    const res = await fetch(
      `https://api.runpod.io/graphql?api_key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: `query { pod(input:{podId:"${RUNPOD_POD_ID}"}) { id name desiredStatus imageName containerDiskInGb volumeInGb runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } } machine { podHostId gpuDisplayName } } }`,
        }),
        signal: AbortSignal.timeout(10000),
      }
    );
    if (!res.ok) return { error: `HTTP ${res.status}` };
    const data = await res.json();
    return data.data?.pod || { error: "Pod not found" };
  } catch (e: any) {
    return { error: e.message };
  }
}

function buildCaption(params: {
  prompt: string;
  style?: string;
  instrument?: string;
}): string {
  const parts: string[] = [];

  if (params.style) {
    parts.push(params.style);
  }

  parts.push(params.prompt);

  if (params.instrument) {
    parts.push(params.instrument);
  }

  return parts.join(", ");
}
