/**
 * Универсальный интерфейс для любого движка генерации музыки.
 *
 * Чтобы добавить новую модель:
 * 1. Создать файл в server/engines/ (например, my-model.ts)
 * 2. Реализовать класс, который implements MusicEngine
 * 3. Зарегистрировать его в server/engines/registry.ts → initializeEngines()
 */

export interface EngineInfo {
  id: string;
  name: string;
  description: string;
  maxDuration: number;
  supportedParams: string[];
}

export interface GenerateParams {
  prompt: string;
  lyrics?: string;
  duration?: number;
  style?: string;
  instrument?: string;
  tags?: string;
  negative_tags?: string;
  title?: string;
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
  temperature?: number;
  cfg_scale?: number;
  topk?: number;
}

export interface TaskResult {
  taskId: string;
}

export interface TaskStatus {
  status: "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED";
  audio_path?: string;
  error?: string;
  progress?: number;
}

export interface AudioResult {
  buffer: Buffer;
  contentType: string;
}

export interface MusicEngine {
  getInfo(): EngineInfo;
  isConfigured(): boolean;
  checkHealth(): Promise<boolean>;
  submitTask(params: GenerateParams): Promise<TaskResult>;
  queryTaskStatus(taskId: string): Promise<TaskStatus>;
  fetchAudio(audioPath: string): Promise<AudioResult>;
  getDiagnostics(): Promise<Record<string, any>>;
}
