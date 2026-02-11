/**
 * HeartMuLa engine adapter.
 * Delegates all calls to the existing server/heartmula.ts module.
 */

import type { MusicEngine, EngineInfo, GenerateParams, TaskResult, TaskStatus, AudioResult } from "./base";
import * as heartmula from "../heartmula";

export class HeartMuLaEngine implements MusicEngine {
  getInfo(): EngineInfo {
    return {
      id: "heartmula",
      name: "HeartMuLa",
      description: "Full song generation with vocals & lyrics. Supports tags, negative tags, and durations up to 300s.",
      maxDuration: 300,
      supportedParams: [
        "prompt", "lyrics", "duration", "tags", "negative_tags",
        "title", "seed", "temperature", "cfg_scale", "topk",
      ],
    };
  }

  isConfigured(): boolean {
    return heartmula.isConfigured();
  }

  async checkHealth(): Promise<boolean> {
    return heartmula.checkHealth();
  }

  async submitTask(params: GenerateParams): Promise<TaskResult> {
    const result = await heartmula.submitTask({
      prompt: params.prompt,
      lyrics: params.lyrics,
      duration: params.duration,
      tags: params.tags,
      negative_tags: params.negative_tags,
      title: params.title,
      seed: params.seed,
      temperature: params.temperature,
      cfg_scale: params.cfg_scale,
      topk: params.topk,
    });
    return { taskId: result.job_id };
  }

  async queryTaskStatus(taskId: string): Promise<TaskStatus> {
    const result = await heartmula.queryJobStatus(taskId);
    return {
      status: result.status as TaskStatus["status"],
      audio_path: result.audio_path,
      error: result.error,
      progress: result.progress,
    };
  }

  async fetchAudio(audioPath: string): Promise<AudioResult> {
    return heartmula.fetchAudio(audioPath);
  }

  async getDiagnostics(): Promise<Record<string, any>> {
    return heartmula.getPodDiagnostics();
  }
}
