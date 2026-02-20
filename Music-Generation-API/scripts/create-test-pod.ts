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
  const gpuId = process.argv[3] || "NVIDIA RTX 4090";
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

  const input = {
    name: "ace-step-test",
    imageName,
    gpuTypeId: gpuId,
    gpuCount: 1,
    volumeInGb: 0,
    containerDiskInGb: 50,
    minVcpuCount: 4,
    minMemoryInGb: 16,
    ports: "8888/http",
    dockerArgs: "bash -c 'cd /app && ACESTEP_CPU_OFFLOAD=true python3 -u http_server.py'",
    env: [
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
  console.log(`\nHTTP server will start on port 8888 in ~1-2 minutes.`);
  console.log(`Test URL: https://${pod.id}-8888.proxy.runpod.net/health`);
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
    console.log(`\nTest URL: https://${id}-8888.proxy.runpod.net/health`);
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
  console.log(`\nHTTP server will be ready in ~1-2 minutes.`);
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
  console.log(`Pod ${id} terminated (deleted permanently).`);
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
  console.log(`Endpoint: ${baseUrl}`);

  try {
    const healthRes = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(10000) });
    const health = await healthRes.json();
    console.log(`Health: ${JSON.stringify(health)}`);
  } catch (e: any) {
    console.log(`Server not ready: ${e.message}`);
    return;
  }

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
    console.error("Usage: npx tsx scripts/create-test-pod.ts test <pod-id>");
    process.exit(1);
  }

  const models = ["turbo", "acestep-v15-sft", "acestep-v15-base", "acestep-v15-turbo-shift3"];
  console.log(`Testing ${models.length} models on pod ${podId}...`);

  for (const model of models) {
    await runTest(podId, model);
  }

  console.log("\n=== All tests complete ===");
}

async function testSingleModel() {
  const podId = process.argv[3];
  const model = process.argv[4] || "turbo";
  if (!podId) {
    console.error("Usage: npx tsx scripts/create-test-pod.ts test1 <pod-id> [model]");
    console.error("Models: turbo, acestep-v15-sft, acestep-v15-base, acestep-v15-turbo-shift3");
    process.exit(1);
  }

  await runTest(podId, model);
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
  default:
    console.log(`
ACE-Step Pod Manager
Docker image: ${DOCKER_IMAGE}

Usage: npx tsx scripts/create-test-pod.ts <command> [args]

Commands:
  create [gpu-id]              - Create pod (default: NVIDIA RTX 4090)
  list                         - List all your pods
  status <pod-id>              - Get pod status and ports
  stop <pod-id>                - Stop pod (keeps it, no charges)
  resume <pod-id>              - Resume stopped pod
  terminate <pod-id>           - Delete pod permanently
  gpus                         - List available GPUs with pricing
  test <pod-id>                - Test all 4 models
  test1 <pod-id> [model]       - Test single model

Models: turbo, acestep-v15-sft, acestep-v15-base, acestep-v15-turbo-shift3
`);
}
