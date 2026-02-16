import type { MusicEngine, EngineInfo, GenerateParams, TaskResult, TaskStatus, AudioResult } from "./base";
import * as diffrhythm from "../diffrhythm";

export class DiffRhythmEngine implements MusicEngine {
  getInfo(): EngineInfo {
    return {
      id: "diffrhythm",
      name: "DiffRhythm",
      description: "Blazingly fast full-song generation using latent diffusion. Generates 4:45 songs in ~30 seconds. Supports text prompts and lyrics. Runs on RunPod Serverless — GPU starts only when needed. Apache 2.0 license.",
      maxDuration: 285,
      supportedParams: [
        "prompt", "lyrics", "duration", "style", "seed",
      ],
    };
  }

  isConfigured(): boolean {
    return diffrhythm.isConfigured();
  }

  async checkHealth(): Promise<boolean> {
    return diffrhythm.checkHealth();
  }

  async submitTask(params: GenerateParams): Promise<TaskResult> {
    const result = await diffrhythm.submitTask({
      prompt: params.prompt,
      lyrics: params.lyrics,
      duration: params.duration,
      style: params.style || params.tags,
      seed: params.seed,
    });
    return { taskId: result.task_id };
  }

  async queryTaskStatus(taskId: string): Promise<TaskStatus> {
    const result = await diffrhythm.queryTaskStatus(taskId);
    if (result.status === "COMPLETED" && result.audio_base64) {
      return {
        status: "COMPLETED",
        audio_path: taskId,
      };
    }
    return {
      status: result.status as TaskStatus["status"],
      error: result.error,
    };
  }

  async fetchAudio(audioPath: string): Promise<AudioResult> {
    return diffrhythm.fetchAudio(audioPath);
  }

  async getDiagnostics(): Promise<Record<string, any>> {
    return diffrhythm.getPodDiagnostics();
  }
}
