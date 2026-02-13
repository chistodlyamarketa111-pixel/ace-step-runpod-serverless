import type { MusicEngine } from "./base";
import { AceStepEngine } from "./ace-step";
import { HeartMuLaEngine } from "./heartmula";
import { YuEEngine } from "./yue";
import { YuEPPEngine } from "./yue-pp";
import { log } from "../index";

class EngineRegistry {
  private engines = new Map<string, MusicEngine>();

  register(engine: MusicEngine): void {
    const info = engine.getInfo();
    this.engines.set(info.id, engine);
    log(
      `Engine registered: ${info.id} (${info.name}) — configured: ${engine.isConfigured()}`,
      "registry",
    );
  }

  get(id: string): MusicEngine | undefined {
    return this.engines.get(id);
  }

  getAll(): MusicEngine[] {
    return Array.from(this.engines.values());
  }

  getAllInfo() {
    return this.getAll().map((e) => ({
      ...e.getInfo(),
      configured: e.isConfigured(),
    }));
  }
}

export const registry = new EngineRegistry();

export function initializeEngines(): void {
  registry.register(new AceStepEngine());
  registry.register(new HeartMuLaEngine());
  registry.register(new YuEEngine());
  registry.register(new YuEPPEngine());

  const total = registry.getAll().length;
  const configured = registry.getAll().filter((e) => e.isConfigured()).length;
  log(`${configured}/${total} engines configured and ready`, "registry");
}
