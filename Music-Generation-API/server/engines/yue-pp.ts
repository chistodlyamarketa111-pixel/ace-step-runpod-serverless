import type { MusicEngine, EngineInfo, GenerateParams, TaskResult, TaskStatus, AudioResult } from "./base";
import * as yue from "../yue";

export class YuEPPEngine implements MusicEngine {
  getInfo(): EngineInfo {
    return {
      id: "yue-pp",
      name: "YuE + Post-Processing",
      description: "YuE generation with post-processing pipeline: RVC voice conversion (optional) + mastering (loudness normalization, EQ, compression, limiting). Higher quality output.",
      maxDuration: 300,
      supportedParams: [
        "prompt", "lyrics", "style", "seed",
      ],
    };
  }

  isConfigured(): boolean {
    return yue.isConfigured();
  }

  async checkHealth(): Promise<boolean> {
    return yue.checkHealth();
  }

  async submitTask(params: GenerateParams): Promise<TaskResult> {
    const result = await yue.submitTaskWithPP({
      prompt: params.prompt,
      lyrics: params.lyrics,
      duration: params.duration,
      style: params.style,
      genre: params.tags || params.style,
      seed: params.seed,
      num_segments: params.duration ? Math.max(1, Math.ceil(params.duration / 30)) : 2,
    });
    return { taskId: result.task_id };
  }

  async queryTaskStatus(taskId: string): Promise<TaskStatus> {
    const result = await yue.queryTaskStatusPP(taskId);
    return {
      status: result.status as TaskStatus["status"],
      audio_path: result.audio_path,
      error: result.error,
    };
  }

  async fetchAudio(audioPath: string): Promise<AudioResult> {
    return yue.fetchAudio(audioPath);
  }

  async getDiagnostics(): Promise<Record<string, any>> {
    return yue.getPodDiagnostics();
  }
}
