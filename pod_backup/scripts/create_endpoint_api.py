#!/usr/bin/env python3
"""Create RunPod Serverless endpoint via API.

Usage:
    RUNPOD_API_KEY=rpa_... python3 create_endpoint_api.py --image ghcr.io/USER/REPO:latest

Or to create a template first, then endpoint:
    RUNPOD_API_KEY=rpa_... python3 create_endpoint_api.py --image ghcr.io/USER/REPO:latest --create-template
"""

import os
import sys
import json
import argparse
import urllib.request

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
API_URL = "https://api.runpod.io/graphql"


def graphql(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def create_template(image_name, template_name="ace-step-macan-lora"):
    query = """
    mutation saveTemplate($input: SaveTemplateInput!) {
        saveTemplate(input: $input) {
            id
            name
        }
    }
    """
    variables = {
        "input": {
            "name": template_name,
            "imageName": image_name,
            "dockerArgs": "",
            "containerDiskInGb": 50,
            "volumeInGb": 0,
            "env": [],
            "isServerless": True,
        }
    }
    result = graphql(query, variables)
    if "errors" in result:
        print(f"Template error: {result['errors'][0]['message']}")
        return None
    tpl = result["data"]["saveTemplate"]
    print(f"Template created: {tpl['id']} ({tpl['name']})")
    return tpl["id"]


def create_endpoint(image_name, template_id=None, endpoint_name="ace-step-macan-lora"):
    query = """
    mutation saveEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
        }
    }
    """
    inp = {
        "name": endpoint_name,
        "gpuIds": "AMPERE_48",
        "workersMin": 0,
        "workersMax": 1,
        "idleTimeout": 5,
        "flashBoot": True,
        "volumeInGb": 0,
        "env": [],
    }
    if template_id:
        inp["templateId"] = template_id
    else:
        inp["dockerImage"] = image_name

    variables = {"input": inp}
    result = graphql(query, variables)

    if "errors" in result:
        print(f"Endpoint error: {result['errors'][0]['message']}")
        return None

    ep = result["data"]["saveEndpoint"]
    print(f"\nEndpoint created!")
    print(f"  ID:   {ep['id']}")
    print(f"  Name: {ep['name']}")
    print(f"  Run:  https://api.runpod.ai/v2/{ep['id']}/run")
    print(f"  Sync: https://api.runpod.ai/v2/{ep['id']}/runsync")
    print()
    print("Example request:")
    print(f'  curl -X POST "https://api.runpod.ai/v2/{ep["id"]}/runsync" \\')
    print(f'    -H "Authorization: Bearer $RUNPOD_API_KEY" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f"    -d '{{\"input\": {{")
    print(f'      "caption": "voice_macan, Russian hip-hop/rap, emotional male vocals, atmospheric pads, 808 bass, trap beat 85 BPM",')
    print(f'      "lyrics": "Ваш текст здесь",')
    print(f'      "duration": 60,')
    print(f'      "seed": 42')
    print(f"    }}}}'")
    return ep["id"]


def main():
    parser = argparse.ArgumentParser(description="Create RunPod Serverless endpoint")
    parser.add_argument("--image", required=True, help="Docker image URL")
    parser.add_argument("--create-template", action="store_true", help="Create template first")
    parser.add_argument("--name", default="ace-step-macan-lora", help="Endpoint name")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        sys.exit(1)

    template_id = None
    if args.create_template:
        template_id = create_template(args.image, f"{args.name}-template")
        if not template_id:
            sys.exit(1)

    endpoint_id = create_endpoint(args.image, template_id, args.name)
    if not endpoint_id:
        sys.exit(1)


if __name__ == "__main__":
    main()
