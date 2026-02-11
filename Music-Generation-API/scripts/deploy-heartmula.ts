const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;

if (!RUNPOD_API_KEY) {
  console.error("RUNPOD_API_KEY is not set. Cannot deploy.");
  process.exit(1);
}

const BASE_URL = "https://rest.runpod.io/v1";

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

async function createPod(): Promise<any> {
  const payload = {
    name: "heartmula-music-gen",
    imageName: "ambsd/heartmula-studio:latest",
    gpuTypeIds: ["NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    gpuCount: 1,
    containerDiskInGb: 100,
    volumeInGb: 0,
    ports: ["8000/http"],
    cloudType: "SECURE",
    allowedCudaVersions: ["12.8", "12.9"],
    env: {},
  };

  console.log("Creating HeartMuLa pod with config:", JSON.stringify(payload, null, 2));

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

async function waitForPod(podId: string, maxWaitMs = 600000): Promise<any> {
  const start = Date.now();
  console.log(`Waiting for pod ${podId} to be ready (up to ${maxWaitMs / 60000} min)...`);

  while (Date.now() - start < maxWaitMs) {
    const pod = await getPod(podId);
    console.log(`  Status: ${pod.desiredStatus} | Runtime: ${pod.runtime?.uptimeInSeconds || 0}s`);

    if (pod.runtime && pod.runtime.uptimeInSeconds > 0) {
      return pod;
    }

    await new Promise((r) => setTimeout(r, 15000));
  }

  throw new Error("Timed out waiting for pod to start");
}

async function waitForHealthy(podId: string, maxWaitMs = 600000): Promise<boolean> {
  const start = Date.now();
  const url = `https://${podId}-8000.proxy.runpod.net/health`;
  console.log(`\nWaiting for HeartMuLa API to become healthy at ${url}...`);

  while (Date.now() - start < maxWaitMs) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(10000) });
      if (res.ok) {
        const data = await res.json();
        console.log(`  Health response: ${JSON.stringify(data)}`);
        if (data.status === "ok" || data.status === "healthy") {
          return true;
        }
      }
      console.log(`  Not ready yet (HTTP ${res.status})`);
    } catch (e: any) {
      console.log(`  Not ready yet (${e.message})`);
    }
    await new Promise((r) => setTimeout(r, 20000));
  }

  return false;
}

async function main() {
  console.log("=== RunPod HeartMuLa Deployment ===\n");

  console.log("Checking existing pods...");
  const existingPods = await listPods();
  console.log(`Found ${existingPods.length} existing pods`);

  const heartmulaPod = existingPods.find(
    (p: any) =>
      p.name === "heartmula-music-gen" ||
      (p.imageName && p.imageName.includes("heartmula"))
  );

  if (heartmulaPod) {
    console.log(`\nFound existing HeartMuLa pod: ${heartmulaPod.id}`);
    console.log(`  Name: ${heartmulaPod.name}`);
    console.log(`  Status: ${heartmulaPod.desiredStatus}`);
    console.log(`  GPU: ${heartmulaPod.machine?.gpuDisplayName || "N/A"}`);
    console.log(`\nPod API URL: https://${heartmulaPod.id}-8000.proxy.runpod.net`);
    console.log(`\n>>> HEARTMULA_POD_ID = ${heartmulaPod.id}`);

    const healthy = await waitForHealthy(heartmulaPod.id, 120000);
    if (healthy) {
      console.log("\nHeartMuLa API is healthy and ready!");
    } else {
      console.log("\nHeartMuLa API is not yet responding. It may still be loading models.");
    }
    return;
  }

  console.log("No existing HeartMuLa pod found. Creating one...\n");

  const pod = await createPod();
  console.log(`\nPod created successfully!`);
  console.log(`  ID: ${pod.id}`);
  console.log(`  Name: ${pod.name}`);
  console.log(`  GPU: ${pod.gpu?.displayName || "N/A"}`);
  console.log(`  Cost: $${pod.costPerHr}/hr`);

  console.log("\nWaiting for pod to start (this can take several minutes)...\n");
  const readyPod = await waitForPod(pod.id);

  console.log("\n=== Pod is RUNNING ===");
  console.log(`Pod API URL: https://${readyPod.id}-8000.proxy.runpod.net`);

  console.log("\nNow waiting for HeartMuLa API to load models (this may take 5-10 min)...");
  const healthy = await waitForHealthy(readyPod.id, 600000);

  if (healthy) {
    console.log("\n=== HeartMuLa API is READY ===");
  } else {
    console.log("\n=== WARNING: HeartMuLa API did not become healthy in time ===");
    console.log("The models may still be loading. Check manually.");
  }

  console.log(`\n>>> HEARTMULA_POD_ID = ${readyPod.id}`);
  console.log(`\nSet this as a secret in your project to enable HeartMuLa generation.`);
}

main().catch((err) => {
  console.error("Deployment failed:", err.message);
  process.exit(1);
});
