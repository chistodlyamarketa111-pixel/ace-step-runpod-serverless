const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
if (!RUNPOD_API_KEY) {
  console.error("RUNPOD_API_KEY not set");
  process.exit(1);
}

const GRAPHQL_URL = "https://api.runpod.io/graphql";
const DOCKER_IMAGE = "ruslanmusin/ace-step-serverless:latest";

async function gql(query: string, variables: Record<string, any> = {}) {
  const res = await fetch(GRAPHQL_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({ query, variables }),
  });
  const json = await res.json();
  if (json.errors) {
    console.error("GraphQL errors:", JSON.stringify(json.errors, null, 2));
    throw new Error(json.errors[0]?.message || "GraphQL error");
  }
  return json.data;
}

const action = process.argv[2] || "help";

async function createPod() {
  const gpuId = process.argv[3] || "NVIDIA RTX A5000";
  const imageName = process.argv[4] || DOCKER_IMAGE;

  console.log(`Creating pod with GPU: ${gpuId}`);
  console.log(`Docker image: ${imageName}`);

  const query = `
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        desiredStatus
        imageName
        machine {
          gpuDisplayName
        }
        runtime {
          uptimeInSeconds
          ports {
            ip
            isIpPublic
            privatePort
            publicPort
            type
          }
        }
      }
    }
  `;

  const startupScript = `bash -c '
set -e
echo "=== Starting ACE-Step HTTP server ==="
pip install -q huggingface_hub 2>/dev/null || true
cd /workspace
if [ ! -f /workspace/http_server.py ]; then
  echo "http_server.py not found, copying from Docker image..."
  cp /app/http_server.py /workspace/http_server.py 2>/dev/null || echo "No /app/http_server.py in image"
fi
ACESTEP_CHECKPOINT_DIR=/workspace/checkpoints ACESTEP_CPU_OFFLOAD=true PORT=8888 python3 -u http_server.py
'`;

  const input = {
    name: "ace-step-test",
    imageName,
    gpuTypeId: gpuId,
    gpuCount: 1,
    volumeInGb: 20,
    containerDiskInGb: 30,
    minVcpuCount: 4,
    minMemoryInGb: 16,
    ports: "8888/http",
    dockerArgs: startupScript,
    env: [
      { key: "ACESTEP_CHECKPOINT_DIR", value: "/workspace/checkpoints" },
      { key: "ACESTEP_CPU_OFFLOAD", value: "true" },
    ],
  };

  const data = await gql(query, { input });
  const pod = data.podFindAndDeployOnDemand;

  console.log("\n=== Pod Created ===");
  console.log(`ID: ${pod.id}`);
  console.log(`Name: ${pod.name}`);
  console.log(`Status: ${pod.desiredStatus}`);
  console.log(`GPU: ${pod.machine?.gpuDisplayName || gpuId}`);
  console.log(`Image: ${pod.imageName}`);
  console.log(`\nPod ID: ${pod.id}`);
  console.log(`Monitor: https://www.runpod.io/console/pods/${pod.id}`);
  console.log(`\nDocker image has all deps pre-installed.`);
  console.log(`Models on volume /workspace/checkpoints (persist across stop/resume).`);
  console.log(`If models missing, they download from HuggingFace on first run (~10 min).`);
  console.log(`\nTest URL: https://${pod.id}-8888.proxy.runpod.net/health`);
  console.log(`\nCommands:`);
  console.log(`  Stop:    cd Music-Generation-API && npx tsx scripts/create-test-pod.ts stop ${pod.id}`);
  console.log(`  Resume:  cd Music-Generation-API && npx tsx scripts/create-test-pod.ts resume ${pod.id}`);
  console.log(`  Status:  cd Music-Generation-API && npx tsx scripts/create-test-pod.ts status ${pod.id}`);
  console.log(`  Test:    cd Music-Generation-API && npx tsx scripts/create-test-pod.ts test ${pod.id}`);
}

async function getPodStatus(podId?: string) {
  const id = podId || process.argv[3];
  if (!id) {
    console.error("Usage: npx tsx scripts/create-test-pod.ts status <pod-id>");
    process.exit(1);
  }

  const query = `
    query pod($input: PodFilter!) {
      pod(input: $input) {
        id
        name
        desiredStatus
        lastStatusChange
        imageName
        machine {
          gpuDisplayName
        }
        runtime {
          uptimeInSeconds
          gpus {
            id
            gpuUtilPercent
            memoryUtilPercent
          }
          ports {
            ip
            isIpPublic
            privatePort
            publicPort
            type
          }
        }
      }
    }
  `;

  const data = await gql(query, { input: { podId: id } });
  const pod = data.pod;

  if (!pod) {
    console.log("Pod not found");
    return;
  }

  console.log("\n=== Pod Status ===");
  console.log(`ID: ${pod.id}`);
  console.log(`Name: ${pod.name}`);
  console.log(`Status: ${pod.desiredStatus}`);
  console.log(`GPU: ${pod.machine?.gpuDisplayName}`);
  console.log(`Uptime: ${pod.runtime?.uptimeInSeconds || 0}s`);

  if (pod.runtime?.ports) {
    console.log("\nPorts:");
    for (const p of pod.runtime.ports) {
      console.log(`  ${p.privatePort} -> ${p.ip}:${p.publicPort} (${p.type})`);
    }
    console.log(`\nProxy URL: https://${id}-8888.proxy.runpod.net`);
  }

  if (pod.runtime?.gpus) {
    console.log("\nGPUs:");
    for (const g of pod.runtime.gpus) {
      console.log(`  ${g.id}: util=${g.gpuUtilPercent}%, mem=${g.memoryUtilPercent}%`);
    }
  }

  try {
    const healthRes = await fetch(`https://${id}-8888.proxy.runpod.net/health`, { signal: AbortSignal.timeout(10000) });
    const health = await healthRes.json();
    console.log(`\nServer health: ${JSON.stringify(health)}`);
  } catch {
    console.log(`\nServer not responding yet (may still be starting).`);
  }
}

async function stopPod() {
  const id = process.argv[3];
  if (!id) {
    console.error("Usage: npx tsx scripts/create-test-pod.ts stop <pod-id>");
    process.exit(1);
  }

  const query = `
    mutation stopPod($input: PodStopInput!) {
      podStop(input: $input) {
        id
        desiredStatus
      }
    }
  `;

  const data = await gql(query, { input: { podId: id } });
  console.log(`Pod ${id} stopped. Status: ${data.podStop.desiredStatus}`);
  console.log(`Volume /workspace preserved (models saved).`);
  console.log(`\nTo resume: npx tsx scripts/create-test-pod.ts resume ${id}`);
}

async function resumePod() {
  const id = process.argv[3];
  if (!id) {
    console.error("Usage: npx tsx scripts/create-test-pod.ts resume <pod-id>");
    process.exit(1);
  }

  const query = `
    mutation resumePod($input: PodResumeInput!) {
      podResume(input: $input) {
        id
        desiredStatus
        imageName
        machine {
          gpuDisplayName
        }
      }
    }
  `;

  const data = await gql(query, { input: { podId: id, gpuCount: 1 } });
  const pod = data.podResume;
  console.log(`Pod ${id} resumed.`);
  console.log(`Status: ${pod.desiredStatus}`);
  console.log(`GPU: ${pod.machine?.gpuDisplayName}`);
  console.log(`\nHTTP server starts automatically (~1-2 min).`);
  console.log(`Test URL: https://${id}-8888.proxy.runpod.net/health`);
}

async function terminatePod() {
  const id = process.argv[3];
  if (!id) {
    console.error("Usage: npx tsx scripts/create-test-pod.ts terminate <pod-id>");
    process.exit(1);
  }

  const query = `
    mutation terminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
  `;

  await gql(query, { input: { podId: id } });
  console.log(`Pod ${id} terminated (deleted permanently, volume lost).`);
}

async function listPods() {
  const query = `
    query myPods {
      myself {
        pods {
          id
          name
          desiredStatus
          imageName
          machine {
            gpuDisplayName
          }
          runtime {
            uptimeInSeconds
          }
        }
      }
    }
  `;

  const data = await gql(query);
  const pods = data.myself?.pods || [];

  if (pods.length === 0) {
    console.log("No pods found.");
    return;
  }

  console.log("\n=== Your Pods ===\n");
  console.log("ID".padEnd(28) + "Name".padEnd(20) + "Status".padEnd(12) + "GPU".padEnd(22) + "Uptime");
  console.log("-".repeat(90));

  for (const p of pods) {
    const uptime = p.runtime?.uptimeInSeconds ? `${Math.round(p.runtime.uptimeInSeconds / 60)}min` : "-";
    console.log(
      `${p.id.padEnd(28)}${(p.name || "").padEnd(20)}${p.desiredStatus.padEnd(12)}${(p.machine?.gpuDisplayName || "").padEnd(22)}${uptime}`
    );
  }
}

async function listGpus() {
  const query = `
    query gpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        lowestPrice(input: { gpuCount: 1 }) {
          minimumBidPrice
          uninterruptablePrice
        }
      }
    }
  `;

  const data = await gql(query);
  const gpus = data.gpuTypes
    .filter((g: any) => g.lowestPrice?.uninterruptablePrice && g.memoryInGb >= 24)
    .sort((a: any, b: any) => (a.lowestPrice?.uninterruptablePrice || 999) - (b.lowestPrice?.uninterruptablePrice || 999));

  console.log("\n=== Available GPUs (24GB+ VRAM) ===\n");
  console.log("ID".padEnd(30) + "VRAM".padEnd(8) + "Price/hr".padEnd(12) + "Community".padEnd(12) + "Secure");
  console.log("-".repeat(75));

  for (const g of gpus) {
    const price = g.lowestPrice?.uninterruptablePrice?.toFixed(2) || "N/A";
    console.log(
      `${g.id.padEnd(30)}${(g.memoryInGb + "GB").padEnd(8)}$${price.padEnd(11)}${(g.communityCloud ? "Yes" : "No").padEnd(12)}${g.secureCloud ? "Yes" : "No"}`
    );
  }
}

async function runTest(podId: string, model: string) {
  console.log(`\n--- Testing model: ${model} ---`);

  const baseUrl = `https://${podId}-8888.proxy.runpod.net`;

  const testPayload = {
    prompt: "A dreamy lo-fi hip hop beat with warm piano chords and gentle boom-bap rhythm",
    lyrics: "[Instrumental]",
    duration: 30,
    model,
    audio_format: "mp3",
    guidance_scale: 7.0,
    seed: 42,
  };

  console.log(`Sending request to ${baseUrl}/generate ...`);
  const start = Date.now();

  try {
    const res = await fetch(`${baseUrl}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(testPayload),
      signal: AbortSignal.timeout(600000),
    });

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    const result = await res.json() as any;

    if (result.error) {
      console.log(`ERROR (${elapsed}s): ${result.error}`);
    } else if (result.audio_base64) {
      const audioSize = Math.round((result.audio_base64.length * 3) / 4 / 1024);
      console.log(`SUCCESS (${elapsed}s): ${audioSize}KB audio, format=${result.audio_format || "wav"}`);
      if (result.duration) console.log(`  Duration: ${result.duration}s`);
      if (result.model) console.log(`  Model used: ${result.model}`);
    } else {
      console.log(`UNEXPECTED (${elapsed}s):`, JSON.stringify(result).slice(0, 500));
    }
  } catch (e: any) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    console.log(`FAILED (${elapsed}s): ${e.message}`);
  }
}

async function testAllModels() {
  const podId = process.argv[3];
  if (!podId) {
    console.error("Usage: npx tsx scripts/create-test-pod.ts test <pod-id>");
    process.exit(1);
  }

  const baseUrl = `https://${podId}-8888.proxy.runpod.net`;
  try {
    const healthRes = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(10000) });
    const health = await healthRes.json();
    console.log(`Server health: ${JSON.stringify(health)}`);
  } catch (e: any) {
    console.log(`Server not ready: ${e.message}`);
    return;
  }

  const models = ["acestep-v15-turbo", "acestep-v15-sft", "acestep-v15-base", "acestep-v15-turbo-shift3"];
  console.log(`Testing ${models.length} models on pod ${podId}...`);

  for (const model of models) {
    await runTest(podId, model);
  }

  console.log("\n=== All tests complete ===");
}

async function testSingleModel() {
  const podId = process.argv[3];
  const model = process.argv[4] || "acestep-v15-turbo";
  if (!podId) {
    console.error("Usage: npx tsx scripts/create-test-pod.ts test1 <pod-id> [model]");
    console.error("Models: acestep-v15-turbo, acestep-v15-sft, acestep-v15-base, acestep-v15-turbo-shift3");
    process.exit(1);
  }

  await runTest(podId, model);
}

async function deployServer() {
  const podId = process.argv[3];
  if (!podId) {
    console.error("Usage: npx tsx scripts/create-test-pod.ts deploy <pod-id>");
    console.error("Uploads http_server.py to the pod and restarts the server.");
    process.exit(1);
  }

  const baseUrl = `https://${podId}-8888.proxy.runpod.net`;
  const fs = await import("fs");
  const path = await import("path");

  const serverPath = path.resolve(import.meta.dirname || __dirname, "../docker/ace-step/http_server.py");
  if (!fs.existsSync(serverPath)) {
    console.error(`http_server.py not found at ${serverPath}`);
    process.exit(1);
  }

  const code = fs.readFileSync(serverPath, "utf-8");
  const encoded = Buffer.from(code).toString("base64");
  const chunks: string[] = [];
  const CHUNK_SIZE = 50000;
  for (let i = 0; i < encoded.length; i += CHUNK_SIZE) {
    chunks.push(encoded.slice(i, i + CHUNK_SIZE));
  }

  console.log(`Uploading http_server.py (${code.length} bytes, ${chunks.length} chunks)...`);

  for (let i = 0; i < chunks.length; i++) {
    const payload = { chunk: chunks[i], index: i, total: chunks.length, filename: "/workspace/http_server.py" };
    const res = await fetch(`${baseUrl}/upload-chunk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(30000),
    });
    const result = await res.json() as any;
    if (result.error) {
      console.error(`Chunk ${i} failed: ${result.error}`);
      return;
    }
    console.log(`  Chunk ${i + 1}/${chunks.length} uploaded`);
  }

  console.log(`\nServer updated. Restart pod to apply:`);
  console.log(`  npx tsx scripts/create-test-pod.ts stop ${podId}`);
  console.log(`  npx tsx scripts/create-test-pod.ts resume ${podId}`);
}

switch (action) {
  case "create":
    createPod();
    break;
  case "list":
    listPods();
    break;
  case "status":
    getPodStatus();
    break;
  case "stop":
    stopPod();
    break;
  case "resume":
    resumePod();
    break;
  case "terminate":
    terminatePod();
    break;
  case "gpus":
    listGpus();
    break;
  case "test":
    testAllModels();
    break;
  case "test1":
    testSingleModel();
    break;
  case "deploy":
    deployServer();
    break;
  default:
    console.log(`
ACE-Step Pod Manager
Docker image: ${DOCKER_IMAGE}

Usage: npx tsx scripts/create-test-pod.ts <command> [args]

Commands:
  create [gpu-id] [image]      - Create pod (default GPU: NVIDIA RTX A5000)
  list                         - List all your pods
  status <pod-id>              - Get pod status, ports, and health
  stop <pod-id>                - Stop pod (volume preserved, no GPU charges)
  resume <pod-id>              - Resume stopped pod
  terminate <pod-id>           - Delete pod permanently (volume lost!)
  gpus                         - List available GPUs with pricing
  test <pod-id>                - Test all 4 models (30s each)
  test1 <pod-id> [model]       - Test single model
  deploy <pod-id>              - Upload http_server.py to pod

Models: acestep-v15-turbo, acestep-v15-sft, acestep-v15-base, acestep-v15-turbo-shift3

Examples:
  npx tsx scripts/create-test-pod.ts create "NVIDIA RTX A5000"
  npx tsx scripts/create-test-pod.ts test kows2z63ecabf4
  npx tsx scripts/create-test-pod.ts test1 kows2z63ecabf4 acestep-v15-sft
  npx tsx scripts/create-test-pod.ts deploy kows2z63ecabf4
`);
}
