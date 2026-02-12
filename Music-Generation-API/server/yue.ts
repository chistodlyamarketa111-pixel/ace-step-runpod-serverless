import { log } from "./index";

const YUE_POD_ID = process.env.YUE_POD_ID;
const YUE_PORT = process.env.YUE_PORT || "7860";

if (!YUE_POD_ID) {
  console.warn("YUE_POD_ID not set - YuE integration will not work");
}

function getBaseUrl(): string {
  return `https://${YUE_POD_ID}-${YUE_PORT}.proxy.runpod.net`;
}

export function isConfigured(): boolean {
  return Boolean(YUE_POD_ID);
}

interface YueJobState {
  eventId: string;
  status: "IN_PROGRESS" | "COMPLETED" | "FAILED";
  audioFiles?: string[];
  error?: string;
  startedAt: number;
}

const jobStates = new Map<string, YueJobState>();

export async function discoverApiEndpoints(): Promise<string[]> {
  if (!isConfigured()) return [];
  try {
    const res = await fetch(`${getBaseUrl()}/info`, {
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) return [];
    const info = await res.json();
    const endpoints: string[] = [];
    if (info.named_endpoints) {
      for (const name of Object.keys(info.named_endpoints)) {
        endpoints.push(name);
      }
    }
    return endpoints;
  } catch {
    return [];
  }
}

async function findGenerateEndpoint(): Promise<string> {
  const endpoints = await discoverApiEndpoints();
  log(`YuE discovered endpoints: ${JSON.stringify(endpoints)}`, "yue");

  const preferred = [
    "/generate_music",
    "/generate",
    "/predict",
    "/run",
  ];

  for (const ep of preferred) {
    if (endpoints.includes(ep)) return ep;
  }

  const generateEp = endpoints.find(
    (e) => e.includes("generat") || e.includes("music") || e.includes("run")
  );
  if (generateEp) return generateEp;

  return endpoints[0] || "/predict";
}

export async function submitTask(params: {
  prompt: string;
  lyrics?: string;
  duration?: number;
  style?: string;
  genre?: string;
  seed?: number;
  num_segments?: number;
}): Promise<{ task_id: string }> {
  if (!isConfigured()) {
    throw new Error("YuE Pod is not configured. Set YUE_POD_ID.");
  }

  const endpointName = await findGenerateEndpoint();
  log(`YuE using endpoint: ${endpointName}`, "yue");

  const lyricsText = params.lyrics?.trim() || buildDefaultLyrics(params.prompt);
  const genreText = params.genre || params.style || buildGenreFromPrompt(params.prompt);
  const numSegments = params.num_segments || 2;
  const seed = params.seed ?? -1;

  const data = [
    lyricsText,
    genreText,
    numSegments,
    seed,
  ];

  log(`YuE submit: endpoint=${endpointName}, data=${JSON.stringify(data)}`, "yue");

  const response = await fetch(`${getBaseUrl()}/call${endpointName}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
    signal: AbortSignal.timeout(30000),
  });

  if (!response.ok) {
    const text = await response.text();
    log(`YuE submit error: ${response.status} ${text}`, "yue");
    throw new Error(`YuE API error: ${response.status} - ${text}`);
  }

  const result = await response.json();
  log(`YuE submit response: ${JSON.stringify(result)}`, "yue");

  const eventId = result.event_id;
  if (!eventId) {
    throw new Error("YuE did not return an event_id");
  }

  const taskId = `yue_${eventId}`;
  jobStates.set(taskId, {
    eventId,
    status: "IN_PROGRESS",
    startedAt: Date.now(),
  });

  pollGradioResult(taskId, endpointName, eventId);

  return { task_id: taskId };
}

async function pollGradioResult(
  taskId: string,
  endpointName: string,
  eventId: string,
): Promise<void> {
  const maxWaitMs = 900000;
  const state = jobStates.get(taskId);
  if (!state) return;

  try {
    const sseUrl = `${getBaseUrl()}/call${endpointName}/${eventId}`;
    log(`YuE SSE polling: ${sseUrl}`, "yue");

    const response = await fetch(sseUrl, {
      signal: AbortSignal.timeout(maxWaitMs),
    });

    if (!response.ok) {
      state.status = "FAILED";
      state.error = `SSE connection failed: ${response.status}`;
      return;
    }

    const text = await response.text();
    log(`YuE SSE response (first 500 chars): ${text.substring(0, 500)}`, "yue");

    const events = parseSSEEvents(text);

    for (const event of events) {
      if (event.event === "complete") {
        try {
          const data = JSON.parse(event.data);
          log(`YuE complete data: ${JSON.stringify(data)}`, "yue");
          const audioFiles = extractAudioFiles(data);
          if (audioFiles.length > 0) {
            state.status = "COMPLETED";
            state.audioFiles = audioFiles;
            log(`YuE task ${taskId} completed with ${audioFiles.length} audio files`, "yue");
          } else {
            state.status = "FAILED";
            state.error = "Generation completed but no audio files returned";
          }
        } catch (e: any) {
          state.status = "FAILED";
          state.error = `Failed to parse result: ${e.message}`;
        }
        return;
      }

      if (event.event === "error") {
        state.status = "FAILED";
        state.error = event.data || "Unknown error";
        log(`YuE task ${taskId} failed: ${state.error}`, "yue");
        return;
      }

      if (event.event === "heartbeat") {
        continue;
      }
    }

    if (state.status === "IN_PROGRESS") {
      if (Date.now() - state.startedAt > maxWaitMs) {
        state.status = "FAILED";
        state.error = "Generation timed out (15 minutes)";
      }
    }
  } catch (e: any) {
    if (e.name === "TimeoutError" || e.name === "AbortError") {
      state.status = "FAILED";
      state.error = "Generation timed out";
    } else {
      state.status = "FAILED";
      state.error = e.message;
    }
    log(`YuE poll error for ${taskId}: ${e.message}`, "yue");
  }
}

function parseSSEEvents(text: string): Array<{ event: string; data: string }> {
  const events: Array<{ event: string; data: string }> = [];
  const lines = text.split("\n");
  let currentEvent = "";
  let currentData = "";

  for (const line of lines) {
    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      currentData = line.slice(6);
    } else if (line === "" && currentEvent) {
      events.push({ event: currentEvent, data: currentData });
      currentEvent = "";
      currentData = "";
    }
  }

  if (currentEvent) {
    events.push({ event: currentEvent, data: currentData });
  }

  return events;
}

function extractAudioFiles(data: any): string[] {
  const files: string[] = [];

  function traverse(obj: any) {
    if (!obj) return;

    if (typeof obj === "string" && (obj.endsWith(".wav") || obj.endsWith(".mp3") || obj.endsWith(".flac"))) {
      files.push(obj);
      return;
    }

    if (typeof obj === "object" && obj.path) {
      files.push(obj.path);
      return;
    }

    if (typeof obj === "object" && obj.url) {
      files.push(obj.url);
      return;
    }

    if (Array.isArray(obj)) {
      for (const item of obj) {
        traverse(item);
      }
      return;
    }

    if (typeof obj === "object") {
      for (const value of Object.values(obj)) {
        traverse(value);
      }
    }
  }

  traverse(data);
  return files;
}

function buildDefaultLyrics(prompt: string): string {
  return `[verse]\n${prompt}\n\n[chorus]\n${prompt}`;
}

function buildGenreFromPrompt(prompt: string): string {
  return prompt;
}

export async function queryTaskStatus(taskId: string): Promise<{
  status: string;
  audio_path?: string;
  error?: string;
}> {
  const state = jobStates.get(taskId);
  if (!state) {
    return { status: "FAILED", error: "Task not found" };
  }

  if (state.status === "COMPLETED" && state.audioFiles && state.audioFiles.length > 0) {
    return {
      status: "COMPLETED",
      audio_path: state.audioFiles[0],
    };
  }

  if (state.status === "FAILED") {
    return {
      status: "FAILED",
      error: state.error || "Generation failed",
    };
  }

  return { status: "IN_PROGRESS" };
}

export async function fetchAudio(audioPath: string): Promise<{ buffer: Buffer; contentType: string }> {
  if (!isConfigured()) {
    throw new Error("YuE Pod is not configured.");
  }

  let url: string;
  if (audioPath.startsWith("http")) {
    url = audioPath;
  } else if (audioPath.startsWith("/file=")) {
    url = `${getBaseUrl()}${audioPath}`;
  } else {
    url = `${getBaseUrl()}/file=${encodeURIComponent(audioPath)}`;
  }

  log(`Fetching YuE audio from: ${url}`, "yue");

  const response = await fetch(url, {
    signal: AbortSignal.timeout(60000),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to fetch YuE audio: ${response.status} - ${text}`);
  }

  const contentType = response.headers.get("content-type") || "audio/wav";
  const buffer = Buffer.from(await response.arrayBuffer());

  return { buffer, contentType };
}

export async function checkHealth(): Promise<boolean> {
  if (!isConfigured()) return false;

  try {
    const response = await fetch(`${getBaseUrl()}/`, {
      signal: AbortSignal.timeout(15000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function getPodDiagnostics(): Promise<Record<string, any>> {
  const diagnostics: Record<string, any> = {
    configured: isConfigured(),
    podId: YUE_POD_ID || null,
    port: YUE_PORT,
    baseUrl: isConfigured() ? getBaseUrl() : null,
    timestamp: new Date().toISOString(),
    activeJobs: jobStates.size,
  };

  if (!isConfigured()) return diagnostics;

  try {
    const endpoints = await discoverApiEndpoints();
    diagnostics.endpoints = endpoints;
  } catch (e: any) {
    diagnostics.endpoints = { error: e.message };
  }

  try {
    const healthOk = await checkHealth();
    diagnostics.health = { status: healthOk ? "ok" : "unreachable" };
  } catch (e: any) {
    diagnostics.health = { error: e.message };
  }

  const apiKey = process.env.RUNPOD_API_KEY;
  if (apiKey && YUE_POD_ID) {
    try {
      const res = await fetch(
        `https://api.runpod.io/graphql?api_key=${apiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: `query { pod(input:{podId:"${YUE_POD_ID}"}) { id name desiredStatus imageName containerDiskInGb volumeInGb runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } } machine { podHostId gpuDisplayName } } }`,
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
