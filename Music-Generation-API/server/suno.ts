import { log } from "./index";

const SUNO_API_KEY = process.env.SUNO_API_KEY;
const BASE_URL = "https://api.sunoapi.org";

export function isConfigured(): boolean {
  return Boolean(SUNO_API_KEY);
}

interface SunoTrack {
  id: string;
  audioUrl?: string;
  streamAudioUrl?: string;
  imageUrl?: string;
  prompt?: string;
  modelName?: string;
  title?: string;
  tags?: string;
  createTime?: string;
  duration?: number;
}

interface SunoRecordInfo {
  taskId: string;
  status: string;
  response?: {
    taskId: string;
    sunoData?: SunoTrack[];
  };
  errorCode?: string | null;
  errorMessage?: string | null;
}

export async function generateMusic(params: {
  prompt: string;
  lyrics?: string;
  tags?: string;
  title?: string;
  duration?: number;
  model?: string;
  make_instrumental?: boolean;
  negativeTags?: string;
}): Promise<{ taskId: string }> {
  if (!isConfigured()) {
    throw new Error("SUNO_API_KEY not configured");
  }

  const hasCustomFields = Boolean(params.lyrics?.trim() || params.tags?.trim() || params.title?.trim());
  const isInstrumental = params.make_instrumental || false;

  const modelMap: Record<string, string> = {
    "chirp-crow": "V5",
    "chirp-v4": "V4",
    "chirp-v4-5": "V4_5",
    "V5": "V5",
    "V4": "V4",
    "V4_5": "V4_5",
    "V4_5PLUS": "V4_5PLUS",
    "V4_5ALL": "V4_5ALL",
  };
  const model = modelMap[params.model || "V5"] || "V5";

  const body: Record<string, any> = {
    customMode: hasCustomFields,
    instrumental: isInstrumental,
    model,
    callBackUrl: "https://httpbin.org/post",
  };

  if (hasCustomFields) {
    body.prompt = params.lyrics?.trim() || "";
    body.style = params.tags || params.prompt;
    body.title = params.title || "Untitled";
    if (params.negativeTags) {
      body.negativeTags = params.negativeTags;
    }
  } else {
    body.prompt = params.prompt;
  }

  log(`Suno v1 request: ${JSON.stringify(body)}`, "suno");

  const response = await fetch(`${BASE_URL}/api/v1/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${SUNO_API_KEY}`,
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30000),
  });

  if (!response.ok) {
    const text = await response.text();
    log(`Suno API error: ${response.status} ${text}`, "suno");
    throw new Error(`Suno API error: ${response.status} - ${text}`);
  }

  const result = await response.json();
  log(`Suno generate response: ${JSON.stringify(result)}`, "suno");

  if (result.code !== 200) {
    throw new Error(`Suno API error: ${result.code} - ${result.msg}`);
  }

  if (!result.data?.taskId) {
    throw new Error("Suno returned no taskId");
  }

  return { taskId: result.data.taskId };
}

export async function getTaskStatus(taskId: string): Promise<SunoRecordInfo> {
  if (!isConfigured()) {
    throw new Error("SUNO_API_KEY not configured");
  }

  const response = await fetch(`${BASE_URL}/api/v1/generate/record-info?taskId=${encodeURIComponent(taskId)}`, {
    headers: {
      "Authorization": `Bearer ${SUNO_API_KEY}`,
    },
    signal: AbortSignal.timeout(15000),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Suno status error: ${response.status} - ${text}`);
  }

  const result = await response.json();

  if (result.code !== 200) {
    throw new Error(`Suno status error: ${result.code} - ${result.msg}`);
  }

  return result.data as SunoRecordInfo;
}

export async function pollUntilComplete(
  taskId: string,
  maxWaitMs: number = 600000,
  intervalMs: number = 10000,
): Promise<{ audio_url: string; track: SunoTrack }> {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    try {
      const info = await getTaskStatus(taskId);
      log(`Suno poll: taskId=${taskId} status=${info.status}`, "suno");

      if (info.status === "SUCCESS" || info.status === "FIRST_SUCCESS") {
        const tracks = info.response?.sunoData;
        if (tracks && tracks.length > 0) {
          const track = tracks[0];
          const audioUrl = track.audioUrl || track.streamAudioUrl;
          if (audioUrl) {
            log(`Suno completed: ${track.id} -> ${audioUrl}`, "suno");
            return { audio_url: audioUrl, track };
          }
        }
        log(`Suno status is ${info.status} but no audio URL yet, continuing poll`, "suno");
      }

      if (info.status === "CREATE_TASK_FAILED" || info.status === "GENERATE_AUDIO_FAILED") {
        throw new Error(`Suno generation failed: ${info.status} - ${info.errorMessage || "unknown error"}`);
      }

      if (info.status === "SENSITIVE_WORD_ERROR") {
        throw new Error(`Suno rejected content: contains prohibited words`);
      }

      if (info.status === "CALLBACK_EXCEPTION") {
        const tracks = info.response?.sunoData;
        if (tracks && tracks.length > 0) {
          const track = tracks[0];
          const audioUrl = track.audioUrl || track.streamAudioUrl;
          if (audioUrl) {
            log(`Suno callback exception but audio available: ${audioUrl}`, "suno");
            return { audio_url: audioUrl, track };
          }
        }
      }
    } catch (err: any) {
      if (err.message.includes("Suno generation failed") || err.message.includes("Suno rejected")) throw err;
      log(`Suno poll error (retrying): ${err.message}`, "suno");
    }

    await new Promise(r => setTimeout(r, intervalMs));
  }

  throw new Error("Suno generation timed out after 10 minutes");
}

export async function downloadAudio(audioUrl: string): Promise<Buffer> {
  log(`Downloading Suno audio: ${audioUrl}`, "suno");

  const response = await fetch(audioUrl, {
    signal: AbortSignal.timeout(60000),
  });

  if (!response.ok) {
    throw new Error(`Failed to download Suno audio: ${response.status}`);
  }

  return Buffer.from(await response.arrayBuffer());
}

export async function getCredits(): Promise<any> {
  if (!isConfigured()) {
    return { error: "SUNO_API_KEY not configured", credits_left: null };
  }

  try {
    const response = await fetch(`${BASE_URL}/api/v1/generate/credit`, {
      headers: {
        "Authorization": `Bearer ${SUNO_API_KEY}`,
      },
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      const text = await response.text();
      log(`Suno credits check failed: ${response.status} ${text}`, "suno");
      return { error: `Suno API returned ${response.status}`, credits_left: null, raw: text };
    }

    const result = await response.json();
    if (result.code === 200) {
      return { credits_left: result.data };
    }
    return { error: result.msg, credits_left: null };
  } catch (err: any) {
    log(`Suno credits error: ${err.message}`, "suno");
    return { error: err.message, credits_left: null };
  }
}
