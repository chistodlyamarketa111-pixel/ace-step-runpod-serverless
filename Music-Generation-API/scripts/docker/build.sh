#!/bin/bash
set -e

REGISTRY="${DOCKER_REGISTRY:-your-dockerhub-username}"

show_help() {
    echo "Usage: ./build.sh [command] [engine]"
    echo ""
    echo "Commands:"
    echo "  base              Build the base image with shared dependencies"
    echo "  engine <name>     Build a specific engine image"
    echo "  push-base         Build and push the base image"
    echo "  push <name>       Build and push a specific engine image"
    echo "  all               Build base + all engine images"
    echo ""
    echo "Engines:"
    echo "  diffrhythm        DiffRhythm (latent diffusion)"
    echo ""
    echo "Environment:"
    echo "  DOCKER_REGISTRY   Docker Hub username (default: your-dockerhub-username)"
    echo ""
    echo "Examples:"
    echo "  DOCKER_REGISTRY=myuser ./build.sh base"
    echo "  DOCKER_REGISTRY=myuser ./build.sh engine diffrhythm"
    echo "  DOCKER_REGISTRY=myuser ./build.sh push diffrhythm"
}

build_base() {
    echo "Building base image: ${REGISTRY}/music-gen-base:latest"
    docker build -t "${REGISTRY}/music-gen-base:latest" -f Dockerfile.base .
    echo "Done: ${REGISTRY}/music-gen-base:latest"
}

build_engine() {
    local engine=$1
    local dockerfile="Dockerfile.${engine}"
    local worker="${engine}_serverless_worker.py"

    if [ ! -f "$dockerfile" ]; then
        # check parent scripts dir
        if [ -f "../${engine}_serverless_worker.py" ]; then
            worker="../${engine}_serverless_worker.py"
        fi
        if [ ! -f "$dockerfile" ]; then
            echo "Error: ${dockerfile} not found"
            exit 1
        fi
    fi

    # copy worker to build context if needed
    local worker_file="${engine}_serverless_worker.py"
    if [ ! -f "$worker_file" ] && [ -f "../${worker_file}" ]; then
        cp "../${worker_file}" "./${worker_file}"
    fi

    # update base image reference in Dockerfile
    sed "s|your-dockerhub-username|${REGISTRY}|g" "$dockerfile" > "/tmp/${dockerfile}"

    echo "Building engine image: ${REGISTRY}/music-gen-${engine}:latest"
    docker build -t "${REGISTRY}/music-gen-${engine}:latest" -f "/tmp/${dockerfile}" .
    echo "Done: ${REGISTRY}/music-gen-${engine}:latest"
}

case "${1}" in
    base)
        build_base
        ;;
    engine)
        [ -z "$2" ] && echo "Error: specify engine name" && exit 1
        build_engine "$2"
        ;;
    push-base)
        build_base
        docker push "${REGISTRY}/music-gen-base:latest"
        echo "Pushed: ${REGISTRY}/music-gen-base:latest"
        ;;
    push)
        [ -z "$2" ] && echo "Error: specify engine name" && exit 1
        build_engine "$2"
        docker push "${REGISTRY}/music-gen-${2}:latest"
        echo "Pushed: ${REGISTRY}/music-gen-${2}:latest"
        ;;
    all)
        build_base
        for df in Dockerfile.*; do
            engine="${df#Dockerfile.}"
            [ "$engine" = "base" ] && continue
            [ "$engine" = "example" ] && continue
            build_engine "$engine"
        done
        ;;
    *)
        show_help
        ;;
esac
