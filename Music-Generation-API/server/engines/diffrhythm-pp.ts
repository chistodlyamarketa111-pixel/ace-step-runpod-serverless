import type { MusicEngine, EngineInfo, GenerateParams, TaskResult, TaskStatus, AudioResult } from "./base";
import * as diffrhythm from "../diffrhythm";

export class DiffRhythmPPEngine implements MusicEngine {
  getInfo(): EngineInfo {
    return {
      id: "diffrhythm-pp",
      name: "DiffRhythm + Post-Processing",
      description: "Modular pipeline: DiffRhythm generation → Demucs stem separation → stem remixing → Matchering mastering. Higher quality output through specialized processing at each stage. Apache 2.0 license.",
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
    const result = await diffrhythm.submitPipeline({
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
    return {
      status: result.status as TaskStatus["status"],
      audio_path: result.audio_path,
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
