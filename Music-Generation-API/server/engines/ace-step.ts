/**
 * ACE-Step engine adapter.
 * Delegates all calls to the existing server/runpod.ts module.
 */

import type { MusicEngine, EngineInfo, GenerateParams, TaskResult, TaskStatus, AudioResult } from "./base";
import * as runpod from "../runpod";

export class AceStepEngine implements MusicEngine {
  getInfo(): EngineInfo {
    return {
      id: "ace-step",
      name: "ACE-Step v1.5",
      description: "Instrumental & vocal music generation with fine-grained DiT controls. Supports up to 600s duration.",
      maxDuration: 600,
      supportedParams: [
        "prompt", "lyrics", "duration", "style", "instrument",
        "bpm", "key_scale", "time_signature", "vocal_language",
        "inference_steps", "guidance_scale", "thinking", "shift",
        "infer_method", "use_adg", "batch_size", "seed", "audio_format", "model",
      ],
    };
  }

  isConfigured(): boolean {
    return runpod.isConfigured();
  }

  async checkHealth(): Promise<boolean> {
    return runpod.checkHealth();
  }

  async submitTask(params: GenerateParams): Promise<TaskResult> {
    const result = await runpod.submitTask(params);
    return { taskId: result.task_id };
  }

  async queryTaskStatus(taskId: string): Promise<TaskStatus> {
    const result = await runpod.queryTaskStatus(taskId);
    return {
      status: result.status as TaskStatus["status"],
      audio_path: result.audio_path,
      error: result.error,
    };
  }

  async fetchAudio(audioPath: string): Promise<AudioResult> {
    return runpod.fetchAudio(audioPath);
  }

  async getDiagnostics(): Promise<Record<string, any>> {
    return runpod.getPodDiagnostics();
  }
}
