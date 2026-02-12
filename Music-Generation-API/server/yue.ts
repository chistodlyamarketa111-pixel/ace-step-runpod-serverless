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
  logs?: string;
}

const jobStates = new Map<string, YueJobState>();

const API_PREFIX = "/gradio_api";
const GENERATE_ENDPOINT = "/on_generate_click";

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

  const lyricsText = params.lyrics?.trim() || buildDefaultLyrics(params.prompt);
  const genreText = params.genre || params.style || buildGenreFromPrompt(params.prompt);
  const numSegments = params.num_segments || 2;
  const seed = params.seed ?? 42;

  const dataArray = [
    "/workspace/models/YuE-s1-7B-anneal-en-cot",
    "bf16",
    "/workspace/models/YuE-s2-1B-general",
    "bf16",
    "/workspace/YuE-exllamav2-UI/src/yue/mm_tokenizer_v0.2_hf/tokenizer.model",
    genreText,
    lyricsText,
    numSegments,
    4,
    "/workspace/outputs",
    0,
    3000,
    seed,
    false,
    null,
    0,
    30,
    false,
    null,
    null,
    0,
    30,
    false,
    false,
    "",
    16384,
    "FP16",
    8192,
    "FP16",
  ];

  log(`YuE submit: genre=${genreText}, segments=${numSegments}, seed=${seed}`, "yue");
  log(`YuE lyrics (first 200 chars): ${lyricsText.substring(0, 200)}`, "yue");

  const callUrl = `${getBaseUrl()}${API_PREFIX}/call${GENERATE_ENDPOINT}`;
  log(`YuE POST to: ${callUrl}`, "yue");

  const response = await fetch(callUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: dataArray }),
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

  pollGradioResult(taskId, eventId);

  return { task_id: taskId };
}

async function pollGradioResult(
  taskId: string,
  eventId: string,
): Promise<void> {
  const maxWaitMs = 900000;
  const state = jobStates.get(taskId);
  if (!state) return;

  try {
    const sseUrl = `${getBaseUrl()}${API_PREFIX}/call${GENERATE_ENDPOINT}/${eventId}`;
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
    log(`YuE SSE response (first 1000 chars): ${text.substring(0, 1000)}`, "yue");

    const events = parseSSEEvents(text);

    for (const event of events) {
      if (event.event === "complete") {
        try {
          const data = JSON.parse(event.data);
          log(`YuE complete data: ${JSON.stringify(data).substring(0, 500)}`, "yue");

          const logText = Array.isArray(data) ? (typeof data[0] === "string" ? data[0] : "") : (typeof data === "string" ? data : "");
          state.logs = logText;

          if (logText.includes("Inference started")) {
            log(`YuE task ${taskId}: inference started in background, polling for output files...`, "yue");
            await pollForOutputFiles(state, taskId);
            return;
          }

          const outputMatch = logText.match(/Saved.*?(\S+\.mp3)/i);
          if (outputMatch) {
            state.status = "COMPLETED";
            state.audioFiles = [outputMatch[1]];
            log(`YuE task ${taskId} completed: ${outputMatch[1]}`, "yue");
            return;
          }

          const audioFiles = extractAudioFiles(data);
          if (audioFiles.length > 0) {
            state.status = "COMPLETED";
            state.audioFiles = audioFiles;
            return;
          }

          await pollForOutputFiles(state, taskId);
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

      if (event.event === "generating") {
        try {
          const data = JSON.parse(event.data);
          if (typeof data === "string") {
            state.logs = data;
          } else if (Array.isArray(data) && typeof data[0] === "string") {
            state.logs = data[0];
          }
        } catch {}
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

async function pollForOutputFiles(state: YueJobState, taskId: string): Promise<void> {
  const maxWaitMs = 600000;
  const pollIntervalMs = 15000;
  const startTime = Date.now();

  log(`YuE polling for output files (max ${maxWaitMs / 60000} min)...`, "yue");

  while (Date.now() - startTime < maxWaitMs) {
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));

    try {
      const logResult = await callGradioEndpoint("/refresh_state", [state.logs || ""]);
      if (logResult) {
        const logText = Array.isArray(logResult) ? logResult[0] : logResult;
        if (typeof logText === "string") {
          state.logs = logText;
          log(`YuE logs update (last 200 chars): ${logText.substring(logText.length - 200)}`, "yue");

          const savedMatch = logText.match(/Saved.*?(\S+\.mp3)/i);
          if (savedMatch) {
            state.status = "COMPLETED";
            state.audioFiles = [savedMatch[1]];
            log(`YuE task ${taskId} completed: ${savedMatch[1]}`, "yue");
            return;
          }

          if (logText.includes("Error") || logText.includes("error") || logText.includes("CUDA out of memory")) {
            const errorLines = logText.split("\n").filter(l => l.toLowerCase().includes("error")).slice(-3);
            state.status = "FAILED";
            state.error = errorLines.join("; ") || "Generation failed";
            log(`YuE task ${taskId} failed: ${state.error}`, "yue");
            return;
          }

          if (logText.includes("Generation completed") || logText.includes("Done!")) {
            const fileResult = await callGradioEndpoint("/update_file_explorer", []);
            if (fileResult) {
              log(`YuE file explorer after completion: ${JSON.stringify(fileResult).substring(0, 500)}`, "yue");
            }

            const getFileResult = await callGradioEndpoint("/update_file_explorer_2", []);
            if (getFileResult) {
              log(`YuE file explorer 2: ${JSON.stringify(getFileResult).substring(0, 500)}`, "yue");
              const files = extractAudioFiles(getFileResult);
              if (files.length > 0) {
                state.status = "COMPLETED";
                state.audioFiles = [files[files.length - 1]];
                log(`YuE task ${taskId} completed via explorer: ${files[files.length - 1]}`, "yue");
                return;
              }
            }
          }
        }
      }
    } catch (e: any) {
      log(`YuE poll iteration error: ${e.message}`, "yue");
    }
  }

  state.status = "FAILED";
  state.error = "Generation timed out waiting for output files (10 minutes)";
  log(`YuE task ${taskId} timed out`, "yue");
}

async function callGradioEndpoint(endpoint: string, data: any[]): Promise<any> {
  try {
    const callUrl = `${getBaseUrl()}${API_PREFIX}/call${endpoint}`;
    const res = await fetch(callUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data }),
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) return null;
    const result = await res.json();
    const eventId = result.event_id;
    if (!eventId) return null;

    const sseRes = await fetch(`${callUrl}/${eventId}`, {
      signal: AbortSignal.timeout(15000),
    });
    if (!sseRes.ok) return null;
    const text = await sseRes.text();
    const events = parseSSEEvents(text);
    for (const ev of events) {
      if (ev.event === "complete") {
        return JSON.parse(ev.data);
      }
    }
    return null;
  } catch {
    return null;
  }
}

async function findOutputFiles(state: YueJobState, taskId: string): Promise<void> {
  try {
    const listUrl = `${getBaseUrl()}${API_PREFIX}/call/update_file_explorer`;
    const res = await fetch(listUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [] }),
      signal: AbortSignal.timeout(10000),
    });
    if (res.ok) {
      const result = await res.json();
      const eventId = result.event_id;
      if (eventId) {
        const sseRes = await fetch(`${listUrl}/${eventId}`, {
          signal: AbortSignal.timeout(10000),
        });
        if (sseRes.ok) {
          const text = await sseRes.text();
          const events = parseSSEEvents(text);
          for (const ev of events) {
            if (ev.event === "complete") {
              const data = JSON.parse(ev.data);
              log(`YuE file explorer: ${JSON.stringify(data).substring(0, 500)}`, "yue");
              const files = extractAudioFiles(data);
              if (files.length > 0) {
                const sorted = files.sort().reverse();
                state.status = "COMPLETED";
                state.audioFiles = [sorted[0]];
                log(`YuE task ${taskId} completed via file explorer: ${sorted[0]}`, "yue");
                return;
              }
            }
          }
        }
      }
    }
  } catch (e: any) {
    log(`YuE file explorer error: ${e.message}`, "yue");
  }

  state.status = "FAILED";
  state.error = "Generation completed but no audio files found";
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

    if (typeof obj === "string") {
      if (obj.endsWith(".wav") || obj.endsWith(".mp3") || obj.endsWith(".flac")) {
        files.push(obj);
      }
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
  logs?: string;
}> {
  const state = jobStates.get(taskId);
  if (!state) {
    return { status: "FAILED", error: "Task not found" };
  }

  if (state.status === "COMPLETED" && state.audioFiles && state.audioFiles.length > 0) {
    return {
      status: "COMPLETED",
      audio_path: state.audioFiles[0],
      logs: state.logs,
    };
  }

  if (state.status === "FAILED") {
    return {
      status: "FAILED",
      error: state.error || "Generation failed",
      logs: state.logs,
    };
  }

  return { status: "IN_PROGRESS", logs: state.logs };
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
    apiPrefix: API_PREFIX,
    generateEndpoint: GENERATE_ENDPOINT,
    timestamp: new Date().toISOString(),
    activeJobs: jobStates.size,
  };

  if (!isConfigured()) return diagnostics;

  try {
    const res = await fetch(`${getBaseUrl()}${API_PREFIX}/info`, {
      signal: AbortSignal.timeout(10000),
    });
    if (res.ok) {
      const info = await res.json();
      diagnostics.endpoints = Object.keys(info.named_endpoints || {});
    }
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
