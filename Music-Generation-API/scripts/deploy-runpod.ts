const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;

if (!RUNPOD_API_KEY) {
  console.error("RUNPOD_API_KEY is not set. Cannot deploy.");
  process.exit(1);
}

const BASE_URL = "https://rest.runpod.io/v1";

const ENGINE_CONFIGS = {
  "ace-step": {
    name: "ace-step-music-gen",
    imageName: "valyriantech/ace-step-1.5:latest",
    gpuTypeIds: [
      "NVIDIA L40S",
      "NVIDIA RTX A6000",
      "NVIDIA A100 80GB PCIe",
      "NVIDIA A100-SXM4-80GB",
    ],
    containerDiskInGb: 100,
    envVar: "RUNPOD_POD_ID",
    searchTerms: ["ace-step-music-gen", "ace-step"],
    ports: ["8000/http"],
  },
  heartmula: {
    name: "heartmula-music-gen",
    imageName: "ambsd/heartmula-studio:latest",
    gpuTypeIds: [
      "NVIDIA L40S",
      "NVIDIA RTX A6000",
      "NVIDIA A100 80GB PCIe",
      "NVIDIA A100-SXM4-80GB",
    ],
    containerDiskInGb: 100,
    envVar: "HEARTMULA_POD_ID",
    searchTerms: ["heartmula-music-gen", "heartmula"],
    ports: ["8000/http"],
  },
  yue: {
    name: "yue-music-gen",
    imageName: "alissonpereiraanjos/yue-interface:latest",
    gpuTypeIds: [
      "NVIDIA L40S",
      "NVIDIA RTX A6000",
      "NVIDIA A100 80GB PCIe",
      "NVIDIA A100-SXM4-80GB",
    ],
    containerDiskInGb: 100,
    envVar: "YUE_POD_ID",
    searchTerms: ["yue-music-gen", "yue"],
    ports: ["7860/http"],
  },
} as const;

type EngineName = keyof typeof ENGINE_CONFIGS;

const HEARTMULA_GPU_SETTINGS = {
  quantization_4bit: "false",
  sequential_offload: "false",
  torch_compile: false,
};

async function listPods(): Promise<any[]> {
  const res = await fetch(`${BASE_URL}/pods`, {
    headers: { Authorization: `Bearer ${RUNPOD_API_KEY}` },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to list pods: ${res.status} ${text}`);
  }
  return res.json();
}

async function createPod(engine: EngineName): Promise<any> {
  const config = ENGINE_CONFIGS[engine];
  const payload = {
    name: config.name,
    imageName: config.imageName,
    gpuTypeIds: config.gpuTypeIds,
    gpuCount: 1,
    containerDiskInGb: config.containerDiskInGb,
    volumeInGb: 0,
    ports: config.ports,
    cloudType: "SECURE",
    allowedCudaVersions: ["12.8", "12.9"],
    env: engine === "yue" ? { DOWNLOAD_MODELS: "all" } : {},
  };

  console.log(
    `Creating ${engine} pod with config:`,
    JSON.stringify(payload, null, 2),
  );

  const res = await fetch(`${BASE_URL}/pods`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to create pod: ${res.status} ${text}`);
  }

  return res.json();
}

async function getPod(podId: string): Promise<any> {
  const res = await fetch(`${BASE_URL}/pods/${podId}`, {
    headers: { Authorization: `Bearer ${RUNPOD_API_KEY}` },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to get pod: ${res.status} ${text}`);
  }
  return res.json();
}

async function waitForPod(podId: string, maxWaitMs = 300000): Promise<any> {
  const start = Date.now();
  console.log(`Waiting for pod ${podId} to be ready...`);

  while (Date.now() - start < maxWaitMs) {
    const pod = await getPod(podId);
    console.log(
      `  Status: ${pod.desiredStatus} | Runtime: ${pod.runtime?.uptimeInSeconds || 0}s`,
    );

    if (pod.runtime && pod.runtime.uptimeInSeconds > 0) {
      return pod;
    }

    await new Promise((r) => setTimeout(r, 10000));
  }

  throw new Error("Timed out waiting for pod to start");
}

async function waitForApiReady(
  podId: string,
  engine: EngineName,
  maxWaitMs = 180000,
): Promise<boolean> {
  const port = engine === "yue" ? "7860" : "8000";
  const baseUrl = `https://${podId}-${port}.proxy.runpod.net`;
  const start = Date.now();
  console.log(`Waiting for API to become ready at ${baseUrl}...`);

  while (Date.now() - start < maxWaitMs) {
    try {
      if (engine === "yue") {
        const res = await fetch(`${baseUrl}/`, {
          signal: AbortSignal.timeout(10000),
        });
        if (res.ok) {
          console.log("  Gradio API is ready!");
          return true;
        }
      } else {
        const res = await fetch(`${baseUrl}/health`, {
          signal: AbortSignal.timeout(5000),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.status === "ok") {
            console.log("  API is ready!");
            return true;
          }
        }
      }
    } catch {}
    console.log("  API not ready yet, retrying in 15s...");
    await new Promise((r) => setTimeout(r, 15000));
  }

  console.warn("  API did not become ready within timeout.");
  return false;
}

async function enforceGpuOnlySettings(podId: string): Promise<void> {
  const baseUrl = `https://${podId}-8000.proxy.runpod.net`;

  console.log(
    "\n=== ENFORCING GPU-ONLY SETTINGS (CPU fallback is NEVER acceptable) ===",
  );

  try {
    const currentRes = await fetch(`${baseUrl}/settings/gpu`, {
      signal: AbortSignal.timeout(10000),
    });
    if (!currentRes.ok) {
      console.error(`  Failed to read GPU settings: HTTP ${currentRes.status}`);
      return;
    }
    const current = await currentRes.json();
    console.log(`  Current settings: ${JSON.stringify(current)}`);

    const needsUpdate =
      current.quantization_4bit !== HEARTMULA_GPU_SETTINGS.quantization_4bit ||
      current.sequential_offload !==
        HEARTMULA_GPU_SETTINGS.sequential_offload ||
      current.torch_compile !== HEARTMULA_GPU_SETTINGS.torch_compile;

    if (!needsUpdate) {
      console.log("  GPU settings already correct - GPU-only mode confirmed.");
      return;
    }

    console.log("  Updating GPU settings to enforce GPU-only mode...");
    const updateRes = await fetch(`${baseUrl}/settings/gpu`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(HEARTMULA_GPU_SETTINGS),
      signal: AbortSignal.timeout(10000),
    });

    if (!updateRes.ok) {
      const text = await updateRes.text();
      console.error(`  Failed to update GPU settings: ${text}`);
      return;
    }

    const updated = await updateRes.json();
    console.log(`  GPU settings enforced: ${JSON.stringify(updated)}`);
    console.log("  GPU-ONLY mode is now active. CPU fallback disabled.");
  } catch (e: any) {
    console.error(`  Error enforcing GPU settings: ${e.message}`);
    console.error(
      "  IMPORTANT: Manually enforce GPU settings after pod is fully ready!",
    );
  }
}

async function main() {
  const engine = (process.argv[2] || "ace-step") as EngineName;

  if (!ENGINE_CONFIGS[engine]) {
    console.error(`Unknown engine: ${engine}. Use: ace-step | heartmula | yue`);
    process.exit(1);
  }

  const config = ENGINE_CONFIGS[engine];
  console.log(`=== RunPod ${engine.toUpperCase()} Deployment ===\n`);

  console.log("Checking existing pods...");
  const existingPods = await listPods();
  const existingPod = existingPods.find(
    (p: any) =>
      config.searchTerms.some((term) => p.name === term) ||
      (p.imageName &&
        config.searchTerms.some((term) => p.imageName.includes(term))),
  );

  if (existingPod) {
    console.log(`\nFound existing ${engine} pod: ${existingPod.id}`);
    console.log(`  Name: ${existingPod.name}`);
    console.log(`  Status: ${existingPod.desiredStatus}`);
    console.log(`  GPU: ${existingPod.machine?.gpuDisplayName || "N/A"}`);
    console.log(
      `\nPod API URL: https://${existingPod.id}-8000.proxy.runpod.net`,
    );
    console.log(`\nSet this as ${config.envVar}: ${existingPod.id}`);

    if (engine === "heartmula") {
      const ready = await waitForApiReady(existingPod.id, engine, 30000);
      if (ready) await enforceGpuOnlySettings(existingPod.id);
    }
    if (engine === "yue") {
      await waitForApiReady(existingPod.id, engine, 30000);
    }
    return;
  }

  console.log(`No existing ${engine} pod found. Creating one...\n`);

  const pod = await createPod(engine);
  console.log(`\nPod created successfully!`);
  console.log(`  ID: ${pod.id}`);
  console.log(`  Name: ${pod.name}`);
  console.log(`  GPU: ${pod.gpu?.displayName || "N/A"}`);
  console.log(`  Cost: $${pod.costPerHr}/hr`);

  console.log("\nWaiting for pod to start (this can take a few minutes)...\n");
  const readyPod = await waitForPod(pod.id);

  console.log("\n=== Pod is RUNNING ===");
  console.log(`Pod API URL: https://${readyPod.id}-8000.proxy.runpod.net`);
  console.log(`\nIMPORTANT: Set this environment variable:`);
  console.log(`  ${config.envVar} = ${readyPod.id}`);

  if (engine === "heartmula") {
    console.log(
      "\nWaiting for HeartMuLa API to load model (may take a few minutes)...",
    );
    const apiReady = await waitForApiReady(readyPod.id, engine);
    if (apiReady) {
      await enforceGpuOnlySettings(readyPod.id);
    } else {
      console.log("\nWARNING: Could not auto-enforce GPU settings.");
      console.log(
        "After the API is ready, run: curl -X POST <app-url>/api/heartmula/enforce-gpu",
      );
    }
  }

  if (engine === "yue") {
    console.log(
      "\nWaiting for YuE Gradio API to load models (this can take 5-10 minutes)...",
    );
    const apiReady = await waitForApiReady(readyPod.id, engine, 600000);
    if (apiReady) {
      console.log("  YuE Gradio interface is ready!");
    } else {
      console.log(
        "\nWARNING: YuE API did not become ready within timeout. Model may still be downloading.",
      );
    }
  }

  const port = engine === "yue" ? "7860" : "8000";
  console.log(
    `\nThe model may need additional time to load after the pod starts.`,
  );
  console.log(`Check at: https://${readyPod.id}-${port}.proxy.runpod.net/`);
}

main().catch((err) => {
  console.error("Deployment failed:", err.message);
  process.exit(1);
});
