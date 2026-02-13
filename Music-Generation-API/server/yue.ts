import { log } from "./index";

const YUE_POD_ID = process.env.YUE_POD_ID;
const YUE_API_PORT = process.env.YUE_API_PORT || "8000";

if (!YUE_POD_ID) {
  console.warn("YUE_POD_ID not set - YuE integration will not work");
}

function getBaseUrl(): string {
  return `https://${YUE_POD_ID}-${YUE_API_PORT}.proxy.runpod.net`;
}

export function isConfigured(): boolean {
  return Boolean(YUE_POD_ID);
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

  const lyricsText = params.lyrics?.trim() || buildDefaultLyrics(params.prompt);
  const genreText = params.genre || params.style || params.prompt;
  const duration = params.duration || 60;
  const numSegments = params.num_segments || Math.max(2, Math.ceil(duration / 30));
  const seed = params.seed ?? 42;

  const customFilename = `yue${Date.now().toString(36)}`;
  log(`YuE submit: genre=${genreText}, segments=${numSegments}, seed=${seed}, filename=${customFilename}`, "yue");

  const response = await fetch(`${getBaseUrl()}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      genre: genreText,
      lyrics: lyricsText,
      num_segments: numSegments,
      duration,
      seed,
      max_new_tokens: 3000,
      custom_filename: customFilename,
    }),
    signal: AbortSignal.timeout(30000),
  });

  if (!response.ok) {
    const text = await response.text();
    log(`YuE submit error: ${response.status} ${text}`, "yue");
    throw new Error(`YuE API error: ${response.status} - ${text}`);
  }

  const result = await response.json() as any;
  log(`YuE submit response: ${JSON.stringify(result)}`, "yue");

  const jobId = result.job_id;
  if (!jobId) {
    throw new Error("YuE did not return a job_id");
  }

  return { task_id: `yue_${jobId}` };
}

export async function queryTaskStatus(taskId: string): Promise<{
  status: string;
  audio_path?: string;
  error?: string;
  logs?: string;
}> {
  if (!isConfigured()) {
    return { status: "FAILED", error: "YuE not configured" };
  }

  const jobId = taskId.startsWith("yue_") ? taskId.slice(4) : taskId;

  try {
    const response = await fetch(`${getBaseUrl()}/status/${jobId}`, {
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      if (response.status === 404) {
        return { status: "FAILED", error: "Job not found on YuE pod" };
      }
      return { status: "FAILED", error: `Status check failed: ${response.status}` };
    }

    const data = await response.json() as any;

    if (data.status === "COMPLETED" && data.output_files?.length > 0) {
      return {
        status: "COMPLETED",
        audio_path: data.output_files[0],
        logs: data.logs,
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
    log(`YuE status check error: ${e.message}`, "yue");
    return { status: "IN_PROGRESS", logs: `Status check temporarily unavailable: ${e.message}` };
  }
}

export async function fetchAudio(audioPath: string): Promise<{ buffer: Buffer; contentType: string }> {
  if (!isConfigured()) {
    throw new Error("YuE Pod is not configured.");
  }

  let url: string;
  if (audioPath.startsWith("http")) {
    url = audioPath;
  } else {
    const encodedPath = encodeURIComponent(audioPath);
    url = `${getBaseUrl()}/download/${encodedPath}`;
  }

  log(`Fetching YuE audio from: ${url}`, "yue");

  const response = await fetch(url, {
    signal: AbortSignal.timeout(120000),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to fetch YuE audio: ${response.status} - ${text}`);
  }

  const contentType = response.headers.get("content-type") || "audio/mpeg";
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
    podId: YUE_POD_ID || null,
    apiPort: YUE_API_PORT,
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
  if (apiKey && YUE_POD_ID) {
    try {
      const res = await fetch(
        `https://api.runpod.io/graphql?api_key=${apiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: `query { pod(input:{podId:"${YUE_POD_ID}"}) { id name desiredStatus imageName runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } } machine { gpuDisplayName } } }`,
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

// ===== POST-PROCESSING (YuE+PP) =====

const ppStateMap = new Map<string, {
  phase: "generating" | "postprocessing";
  genJobId: string;
  ppJobId?: string;
}>();

export async function submitTaskWithPP(params: {
  prompt: string;
  lyrics?: string;
  duration?: number;
  style?: string;
  genre?: string;
  seed?: number;
  num_segments?: number;
}): Promise<{ task_id: string }> {
  const genResult = await submitTask(params);
  const genJobId = genResult.task_id;
  const ppTaskId = `yuepp_${genJobId}`;

  ppStateMap.set(ppTaskId, {
    phase: "generating",
    genJobId,
  });

  log(`YuE+PP submit: generation started as ${genJobId}, tracking as ${ppTaskId}`, "yue-pp");
  return { task_id: ppTaskId };
}

export async function queryTaskStatusPP(taskId: string): Promise<{
  status: string;
  audio_path?: string;
  error?: string;
  logs?: string;
}> {
  if (!isConfigured()) {
    return { status: "FAILED", error: "YuE not configured" };
  }

  const state = ppStateMap.get(taskId);
  if (!state) {
    return { status: "FAILED", error: "PP task not found" };
  }

  if (state.phase === "generating") {
    const genStatus = await queryTaskStatus(state.genJobId);

    if (genStatus.status === "COMPLETED") {
      log(`YuE+PP: generation complete, starting post-processing`, "yue-pp");
      state.phase = "postprocessing";

      const rawJobId = state.genJobId.startsWith("yue_") ? state.genJobId.slice(4) : state.genJobId;

      try {
        const response = await fetch(`${getBaseUrl()}/postprocess`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            job_id: rawJobId,
          }),
          signal: AbortSignal.timeout(30000),
        });

        if (!response.ok) {
          const text = await response.text();
          return { status: "FAILED", error: `Post-processing start failed: ${text}` };
        }

        const ppResult = await response.json() as any;
        state.ppJobId = ppResult.pp_job_id;
        log(`YuE+PP: post-processing started as ${state.ppJobId}`, "yue-pp");

        return { status: "IN_PROGRESS", logs: "Generation complete. Post-processing in progress..." };
      } catch (e: any) {
        return { status: "FAILED", error: `Failed to start post-processing: ${e.message}` };
      }
    }

    if (genStatus.status === "FAILED") {
      return { status: "FAILED", error: genStatus.error, logs: genStatus.logs };
    }

    return { status: "IN_PROGRESS", logs: genStatus.logs };
  }

  if (state.phase === "postprocessing" && state.ppJobId) {
    try {
      const response = await fetch(`${getBaseUrl()}/pp/status/${state.ppJobId}`, {
        signal: AbortSignal.timeout(15000),
      });

      if (!response.ok) {
        return { status: "IN_PROGRESS", logs: "Checking post-processing status..." };
      }

      const data = await response.json() as any;

      if (data.status === "COMPLETED" && data.output_files?.length > 0) {
        ppStateMap.delete(taskId);
        return {
          status: "COMPLETED",
          audio_path: data.output_files[0],
          logs: data.logs,
        };
      }

      if (data.status === "FAILED") {
        ppStateMap.delete(taskId);
        return {
          status: "FAILED",
          error: data.error || "Post-processing failed",
          logs: data.logs,
        };
      }

      return { status: "IN_PROGRESS", logs: data.logs || "Post-processing in progress..." };
    } catch (e: any) {
      return { status: "IN_PROGRESS", logs: `PP status check unavailable: ${e.message}` };
    }
  }

  return { status: "IN_PROGRESS" };
}

function buildDefaultLyrics(prompt: string): string {
  return `[verse]\n${prompt}\n\n[chorus]\n${prompt}`;
}
