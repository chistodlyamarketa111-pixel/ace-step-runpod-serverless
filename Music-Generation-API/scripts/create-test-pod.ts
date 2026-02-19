const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
if (!RUNPOD_API_KEY) {
  console.error("RUNPOD_API_KEY not set");
  process.exit(1);
}

const GRAPHQL_URL = "https://api.runpod.io/graphql";

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

const action = process.argv[2] || "create";

const REPLIT_SERVER_URL = "https://30ae1522-eccc-49e0-b335-f4d16f2f3093-00-3ixp57dkvat9h.picard.replit.dev";

async function createPod() {
  const gpuId = process.argv[3] || "NVIDIA RTX A5000";
  const imageName = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04";

  console.log(`Creating pod with GPU: ${gpuId}`);
  console.log(`Docker image: ${imageName}`);

  const startupScript = `bash -c 'apt-get update && apt-get install -y ffmpeg libsndfile1 && pip install flask huggingface_hub soundfile && pip install git+https://github.com/ACE-Step/ACE-Step.git && python3 -c "from huggingface_hub import snapshot_download as dl; import os; d=\\"/workspace/checkpoints\\"; os.makedirs(d,exist_ok=True); dl(\\"ACE-Step/Ace-Step1.5\\",local_dir=d); dl(\\"ACE-Step/acestep-v15-sft\\",local_dir=d+\\"/acestep-v15-sft\\"); dl(\\"ACE-Step/acestep-v15-base\\",local_dir=d+\\"/acestep-v15-base\\"); dl(\\"ACE-Step/acestep-v15-turbo-shift3\\",local_dir=d+\\"/acestep-v15-turbo-shift3\\"); print(\\"MODELS DONE\\")" && python3 -c "import urllib.request; urllib.request.urlretrieve(\\"${REPLIT_SERVER_URL}/raw/http_server.py\\",\\"/app/http_server.py\\"); print(\\"SERVER DOWNLOADED\\")" && cd /app && ACESTEP_CHECKPOINT_DIR=/workspace/checkpoints python3 -u http_server.py'`;

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

  const input = {
    name: "ace-step-test",
    imageName,
    gpuTypeId: gpuId,
    gpuCount: 1,
    volumeInGb: 0,
    containerDiskInGb: 100,
    minVcpuCount: 4,
    minMemoryInGb: 16,
    ports: "8888/http",
    dockerArgs: startupScript,
    env: [
      { key: "ACESTEP_CHECKPOINT_DIR", value: "/workspace/checkpoints" },
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
  console.log(`\nPod ID for env: ${pod.id}`);
  console.log(`\nMonitor at: https://www.runpod.io/console/pods/${pod.id}`);
  console.log(`\nThe pod will automatically:`);
  console.log(`1. Install dependencies (ffmpeg, ace-step, flask)`);
  console.log(`2. Download all 4 models from HuggingFace`);
  console.log(`3. Download server script from Replit`);
  console.log(`4. Start HTTP server on port 8888`);
  console.log(`\nThis process takes ~10-15 minutes. Check logs in RunPod console.`);
}

async function getPodStatus(podId?: string) {
  const id = podId || process.argv[3];
  if (!id) {
    console.error("Usage: tsx scripts/create-test-pod.ts status <pod-id>");
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
  }

  if (pod.runtime?.gpus) {
    console.log("\nGPUs:");
    for (const g of pod.runtime.gpus) {
      console.log(`  ${g.id}: util=${g.gpuUtilPercent}%, mem=${g.memoryUtilPercent}%`);
    }
  }
}

async function stopPod() {
  const id = process.argv[3];
  if (!id) {
    console.error("Usage: tsx scripts/create-test-pod.ts stop <pod-id>");
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
}

async function terminatePod() {
  const id = process.argv[3];
  if (!id) {
    console.error("Usage: tsx scripts/create-test-pod.ts terminate <pod-id>");
    process.exit(1);
  }

  const query = `
    mutation terminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
  `;

  await gql(query, { input: { podId: id } });
  console.log(`Pod ${id} terminated.`);
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

  const statusData = await gql(`
    query pod($input: PodFilter!) {
      pod(input: $input) {
        runtime {
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
  `, { input: { podId } });

  const ports = statusData.pod?.runtime?.ports;
  if (!ports?.length) {
    console.log("Pod not ready yet, no ports available");
    return;
  }

  const httpPort = ports.find((p: any) => p.privatePort === 8888);
  if (!httpPort) {
    console.log("HTTP port 8888 not found");
    return;
  }

  const baseUrl = `https://${podId}-8888.proxy.runpod.net`;
  console.log(`Endpoint: ${baseUrl}`);

  const testPayload = {
    input: {
      prompt: "A dreamy lo-fi hip hop beat with warm piano chords and gentle boom-bap rhythm",
      lyrics: "[Instrumental]",
      duration: 30,
      model,
      task_type: "text2music",
      audio_format: "mp3",
      guidance_scale: 7.0,
      thinking: true,
      seed: 42,
    },
  };

  console.log(`Sending request...`);
  const start = Date.now();

  try {
    const res = await fetch(`${baseUrl}/runsync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(testPayload),
      signal: AbortSignal.timeout(600000),
    });

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    const result = await res.json();

    if (result.output?.error) {
      console.log(`ERROR (${elapsed}s): ${result.output.error}`);
    } else if (result.output?.audio_base64) {
      const audioSize = Math.round((result.output.audio_base64.length * 3) / 4 / 1024);
      console.log(`SUCCESS (${elapsed}s): ${audioSize}KB audio, format=${result.output.audio_format}`);
      console.log(`  Duration: ${result.output.duration}s, Sample rate: ${result.output.sample_rate}`);
      if (result.output.model) console.log(`  Model used: ${result.output.model}`);
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
    console.error("Usage: tsx scripts/create-test-pod.ts test <pod-id>");
    process.exit(1);
  }

  const models = ["acestep-v15-sft", "acestep-v15-base", "acestep-v15-turbo-shift3"];
  console.log(`Testing ${models.length} models on pod ${podId}...`);

  for (const model of models) {
    await runTest(podId, model);
  }

  console.log("\n=== All tests complete ===");
}

switch (action) {
  case "create":
    createPod();
    break;
  case "status":
    getPodStatus();
    break;
  case "stop":
    stopPod();
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
  default:
    console.log(`
Usage: tsx scripts/create-test-pod.ts <command> [args]

Commands:
  gpus                         - List available GPUs with pricing
  create [gpu-id] [image]      - Create a test pod
  status <pod-id>              - Get pod status
  test <pod-id>                - Test all 3 new models
  stop <pod-id>                - Stop pod (saves state)
  terminate <pod-id>           - Terminate pod (deletes)
`);
}
