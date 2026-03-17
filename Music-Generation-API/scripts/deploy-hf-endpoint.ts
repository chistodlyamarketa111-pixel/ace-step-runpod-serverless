/**
 * Deploy ACE-Step v1.5 to Hugging Face Inference Endpoints.
 *
 * Step 1: Creates a model repo on HF with handler.py + requirements.txt
 * Step 2: Installs ACE-Step source code into the repo
 * Step 3: Creates an Inference Endpoint
 *
 * Usage:
 *   npx tsx scripts/deploy-hf-endpoint.ts
 *
 * Environment:
 *   HF_API_TOKEN — Hugging Face token (required)
 */

const HF_TOKEN = process.env.HF_API_TOKEN;
if (!HF_TOKEN) {
  console.error("ERROR: HF_API_TOKEN not set");
  process.exit(1);
}

const HF_API = "https://huggingface.co/api";

const REPO_NAME = "ace-step-v15-endpoint";
const ENDPOINT_NAME = "ace-step-v15";

async function hfFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const headers: Record<string, string> = {
    "Authorization": `Bearer ${HF_TOKEN}`,
    ...(options.headers as Record<string, string> || {}),
  };
  return fetch(url, { ...options, headers });
}

async function getUsername(): Promise<string> {
  const res = await hfFetch("https://huggingface.co/api/whoami-v2");
  if (!res.ok) throw new Error(`Auth failed: ${res.status} ${await res.text()}`);
  const data = await res.json() as any;
  return data.name;
}

async function createRepo(username: string): Promise<string> {
  const repoId = `${username}/${REPO_NAME}`;
  console.log(`Creating repo: ${repoId}`);

  const res = await hfFetch(`${HF_API}/repos/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      type: "model",
      name: REPO_NAME,
      private: true,
    }),
  });

  if (res.ok) {
    console.log(`Repo created: ${repoId}`);
  } else {
    const text = await res.text();
    if (text.includes("already") || res.status === 409) {
      console.log(`Repo already exists: ${repoId}`);
    } else {
      throw new Error(`Failed to create repo: ${res.status} ${text}`);
    }
  }

  return repoId;
}

async function uploadFile(repoId: string, filePath: string, content: string): Promise<void> {
  console.log(`  Uploading ${filePath}...`);

  const url = `${HF_API}/models/${repoId}/commit/main`;

  const res = await hfFetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      summary: `Upload ${filePath}`,
      files: [{
        path: filePath,
        content: content,
      }],
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload ${filePath} failed: ${res.status} ${text}`);
  }
}

async function uploadFiles(repoId: string, files: Array<{path: string, content: string}>): Promise<void> {
  const url = `${HF_API}/models/${repoId}/commit/main`;

  const ndjson_parts: string[] = [];
  ndjson_parts.push(JSON.stringify({
    key: "header",
    value: { summary: "Upload ACE-Step handler files" },
  }));

  for (const f of files) {
    ndjson_parts.push(JSON.stringify({
      key: "file",
      value: {
        content: f.content,
        path: f.path,
        encoding: "utf-8",
      },
    }));
  }

  const body = ndjson_parts.join("\n");

  const res = await hfFetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/x-ndjson" },
    body: body,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed: ${res.status} ${text.substring(0, 500)}`);
  }
}

import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function uploadRepoFiles(repoId: string): Promise<void> {
  console.log("\nUploading files to HF repo...");

  const dockerDir = path.resolve(__dirname, "../docker/ace-step-hf");

  const handlerContent = fs.readFileSync(path.join(dockerDir, "handler.py"), "utf-8");
  const requirementsContent = fs.readFileSync(path.join(dockerDir, "requirements.txt"), "utf-8");

  const readmeContent = `---
library_name: custom
tags:
  - music-generation
  - ace-step
  - endpoints-template
---

# ACE-Step v1.5 — Inference Endpoint

AI music generation model. Uses custom EndpointHandler.

## Models
- acestep-v15-turbo (8 steps, fastest)
- acestep-v15-sft (32 steps, best quality)
- acestep-v15-base (50 steps)
- acestep-v15-turbo-shift3 (8 steps)

## API

POST request with JSON body:
\`\`\`json
{
  "inputs": {
    "prompt": "upbeat electronic dance music",
    "lyrics": "[verse]\\nHello world...",
    "duration": 30,
    "model": "acestep-v15-turbo",
    "audio_format": "wav",
    "inference_steps": 8
  }
}
\`\`\`

Response:
\`\`\`json
{
  "audio_base64": "...",
  "audio_format": "wav",
  "model": "acestep-v15-turbo",
  "generation_time": 12.5,
  "duration": 30
}
\`\`\`
`;
  console.log("  Uploading handler.py, requirements.txt, README.md...");
  await uploadFiles(repoId, [
    { path: "handler.py", content: handlerContent },
    { path: "requirements.txt", content: requirementsContent },
    { path: "README.md", content: readmeContent },
  ]);

  console.log("Files uploaded successfully!");
}

async function createEndpoint(repoId: string): Promise<any> {
  console.log(`\nCreating Inference Endpoint: ${ENDPOINT_NAME}`);
  console.log(`  Repo: ${repoId}`);
  console.log(`  GPU: nvidia-a10g (~$1/hr)`);
  console.log(`  Scale-to-zero: enabled`);

  const username = await getUsername();
  const res = await hfFetch(`https://api.endpoints.huggingface.cloud/v2/endpoint/${username}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: ENDPOINT_NAME,
      model: {
        repository: repoId,
        task: "custom",
        framework: "custom",
        image: {
          huggingface: {},
        },
      },
      provider: {
        vendor: "aws",
        region: "us-east-1",
      },
      compute: {
        accelerator: "gpu",
        instanceType: "nvidia-a10g",
        instanceSize: "x1",
        scaling: {
          minReplica: 0,
          maxReplica: 1,
          scaleToZeroTimeout: 15,
        },
      },
      type: "protected",
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    if (text.includes("already exists") || res.status === 409) {
      console.log("Endpoint already exists. Fetching info...");
      return getEndpointInfo();
    }
    throw new Error(`Create endpoint failed: ${res.status} ${text}`);
  }

  return res.json();
}

async function getEndpointInfo(): Promise<any> {
  const username = await getUsername();
  const res = await hfFetch(
    `https://api.endpoints.huggingface.cloud/v2/endpoint/${username}/${ENDPOINT_NAME}`
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Get endpoint failed: ${res.status} ${text}`);
  }
  return res.json();
}

async function waitForEndpoint(username: string, maxWaitMs = 900000): Promise<string> {
  console.log("\nWaiting for endpoint to become ready...");
  console.log("(This may take 10-20 minutes — model needs to download + load)");

  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    try {
      const res = await hfFetch(
        `https://api.endpoints.huggingface.cloud/v2/endpoint/${username}/${ENDPOINT_NAME}`
      );
      if (res.ok) {
        const data = await res.json() as any;
        const status = data.status?.state || data.state || "unknown";
        const url = data.status?.url || data.url || "";

        process.stdout.write(`\r  Status: ${status}   `);

        if (status === "running" && url) {
          console.log(`\n\nEndpoint READY!`);
          return url;
        }
        if (status === "failed") {
          throw new Error("Endpoint failed to start. Check HF dashboard.");
        }
      }
    } catch (e: any) {
      if (e.message.includes("failed")) throw e;
    }

    await new Promise(r => setTimeout(r, 15000));
  }

  throw new Error("Timeout waiting for endpoint");
}

async function main() {
  console.log("=== ACE-Step v1.5 → HF Inference Endpoints ===\n");

  const username = await getUsername();
  console.log(`Logged in as: ${username}\n`);

  const repoId = await createRepo(username);

  await uploadRepoFiles(repoId);

  let endpoint: any;
  try {
    endpoint = await createEndpoint(repoId);
    console.log("\nEndpoint created:", JSON.stringify(endpoint, null, 2));
  } catch (e: any) {
    console.error(`\nError creating endpoint: ${e.message}`);
    console.log("\nFiles are uploaded to the HF repo. You can create the endpoint manually:");
    console.log(`  1. Go to https://ui.endpoints.huggingface.co/`);
    console.log(`  2. Click "+ New Endpoint"`);
    console.log(`  3. Select model: ${repoId}`);
    console.log(`  4. GPU: A10G`);
    console.log(`  5. Enable scale-to-zero`);
    console.log(`\nOr set HF_ENDPOINT_URL manually once deployed.`);
    return;
  }

  try {
    const url = await waitForEndpoint(username);
    console.log(`\nEndpoint URL: ${url}`);
    console.log(`\nSet these in your Replit secrets:`);
    console.log(`  HF_ENDPOINT_URL=${url}`);
    console.log(`\nTest:`);
    console.log(`  curl -X POST ${url} \\`);
    console.log(`    -H "Authorization: Bearer $HF_API_TOKEN" \\`);
    console.log(`    -H "Content-Type: application/json" \\`);
    console.log(`    -d '{"inputs":{"prompt":"blues guitar","duration":10}}'`);
  } catch (e: any) {
    console.log(`\n${e.message}`);
    console.log("Check status at: https://ui.endpoints.huggingface.co/");
    const endpointUrl = endpoint?.status?.url || endpoint?.url;
    if (endpointUrl) {
      console.log(`\nEndpoint URL (when ready): ${endpointUrl}`);
      console.log(`Set: HF_ENDPOINT_URL=${endpointUrl}`);
    }
  }
}

main().catch(e => {
  console.error("Fatal error:", e.message);
  process.exit(1);
});
