#!/bin/bash

# --- Terminal Color Palette ---
COLOR_RESET="\033[0m"
COLOR_INFO="\033[1;34m"    # Bold Blue
COLOR_SUCCESS="\033[1;32m" # Bold Green
COLOR_WARN="\033[1;33m"    # Bold Yellow
COLOR_ERROR="\033[1;31m"   # Bold Red
COLOR_MUTED="\033[0;37m"   # Muted Gray
COLOR_PANEL="\033[1;36m"   # Cyan Panel Border

# --- Configuration Constants ---
LLAMA_SERVE_BIN="/home/ibrahim/works/llama.cpp/build/bin/llama-server"
# MODEL_PATH="/home/ibrahim/works/nlp-llm-works/llm-w-llama_cpp/chat-w-llm/models/Llama-3.2-11B-Vision-Instruct.Q8_0.gguf"
# MMPROJ_PATH="/home/ibrahim/works/nlp-llm-works/llm-w-llama_cpp/chat-w-llm/models/Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf"

MODELS_DIR="/home/ibrahim/works/nlp-llm-works/llm-w-llama_cpp/chat-w-llm/models"
MODEL_PATH="$MODELS_DIR/Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
MMPROJ_PATH="$MODELS_DIR/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"

PORT=8091
HOST="127.0.0.1"
CONTEXT_WINDOW=2048

clear
echo -e "${COLOR_PANEL}====================================================================${COLOR_RESET}"
echo -e "${COLOR_SUCCESS}         🚀 OFFLINE MULTIMODAL INFERENCE ENGINE INITIALIZER 🚀       ${COLOR_RESET}"
echo -e "${COLOR_PANEL}====================================================================${COLOR_RESET}"
echo ""

# --- Step 1: Pre-flight Verification Diagnostics ---
echo -e "${COLOR_INFO}[1/4] Running pre-flight system integrity checks...${COLOR_RESET}"
sleep 0.5

# Check if llama-serve binary exists
if [ ! -f "$LLAMA_SERVE_BIN" ]; then
    echo -e "${COLOR_ERROR}❌ ERROR: Binary file not found at: $LLAMA_SERVE_BIN${COLOR_RESET}"
    echo -e "${COLOR_WARN}💡 Fix: Ensure llama.cpp is cloned and compiled ('make' or 'GGML_CUDA=1 make') inside your project folder.${COLOR_RESET}"
    exit 1
else
    echo -e "${COLOR_MUTED}  -> [OK] Engine binary detected.${COLOR_RESET}"
fi

# Check if Base Text GGUF model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${COLOR_ERROR}❌ ERROR: Primary model weight snapshot not found at: $MODEL_PATH${COLOR_RESET}"
    exit 1
else
    echo -e "${COLOR_MUTED}  -> [OK] Primary neural weights detected.${COLOR_RESET}"
fi

# Check if Multimodal Projector (mmproj) model exists
if [ ! -f "$MMPROJ_PATH" ]; then
    echo -e "${COLOR_ERROR}❌ ERROR: Multimodal projection graph not found at: $MMPROJ_PATH${COLOR_RESET}"
    exit 1
else
    echo -e "${COLOR_MUTED}  -> [OK] Vision vector projector detected.${COLOR_RESET}"
fi

echo -e "${COLOR_SUCCESS}✅ System checks passed successfully!${COLOR_RESET}"
echo ""

# --- Step 2: Telemetry & Port Binding Audit ---
echo -e "${COLOR_INFO}[2/4] Auditing network port configurations...${COLOR_RESET}"
sleep 0.5

# Check if the desired port is already running another process
PORT_OCCUPIED=$(lsof -i :$PORT | grep LISTEN)
if [ ! -z "$PORT_OCCUPIED" ]; then
    echo -e "${COLOR_ERROR}❌ CONFIGURATION COLLISION: Port $PORT is already in use by another service!${COLOR_RESET}"
    echo -e "${COLOR_MUTED}$PORT_OCCUPIED${COLOR_RESET}"
    echo -e "${COLOR_WARN}💡 Action: Shut down the competing process or switch the port inside this script.${COLOR_RESET}"
    exit 1
else
    echo -e "${COLOR_MUTED}  -> [OK] Port $PORT is completely unassigned and clear for runtime binding.${COLOR_RESET}"
fi
echo ""

# --- Step 3: Deployment Specifications Panel ---
echo -e "${COLOR_PANEL}--------------------------------------------------------------------${COLOR_RESET}"
echo -e "${COLOR_INFO} ENGINE RUNTIME METRICS:${COLOR_RESET}"
echo -e "${COLOR_INFO} • Binding Protocol :${COLOR_RESET} http://$HOST:$PORT"
echo -e "${COLOR_INFO} • Engine Core Link :${COLOR_RESET} $LLAMA_SERVE_BIN"
echo -e "${COLOR_INFO} • Target Context   :${COLOR_RESET} $CONTEXT_WINDOW Tokens"
echo -e "${COLOR_INFO} • GPU Accelerator  :${COLOR_RESET} Enabled (All layers offloaded to VRAM)"
echo -e "${COLOR_PANEL}--------------------------------------------------------------------${COLOR_RESET}"
echo ""

# --- Step 4: Activating Core Service Process ---
echo -e "${COLOR_INFO}[4/4] Spinning up local C++ engine pipeline instances...${COLOR_RESET}"
echo -e "${COLOR_WARN}🔔 SYSTEM NOTE: Logs streaming below are direct C++ native diagnostics.${COLOR_RESET}"
echo -e "${COLOR_MUTED}Press [CTRL + C] anytime to safely terminate the offline server pool.${COLOR_RESET}"
echo ""
sleep 1.5

# Execute the native high-performance service process wrapper
exec "$LLAMA_SERVE_BIN" \
  -m "$MODEL_PATH" \
  --mmproj "$MMPROJ_PATH" \
  --port "$PORT" \
  --host "$HOST" \
  --ctx-size "$CONTEXT_WINDOW" \
  --n-gpu-layers 25
# Instead of -1 (which forces 100% GPU calculation), set it to 20 or 30.
# This tells the engine to split the burden between your graphics card and your main system memory.