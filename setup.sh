#!/usr/bin/env bash
# =============================================================================
# LLM-Eval Setup Script
# Run this once from your terminal to install dependencies and verify setup.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== LLM-Eval Setup ==="
echo ""

# --- 1. Check Python version ---
python_version=$(python3 --version 2>&1)
echo "[1/4] Python: $python_version"

# --- 2. Install dependencies ---
echo "[2/4] Installing dependencies..."
pip3 install openai anthropic python-dotenv rich --quiet
echo "      Dependencies installed."

# --- 3. Verify .env exists and key is set ---
echo "[3/4] Checking .env..."
if [ ! -f ".env" ]; then
    echo "      ERROR: .env file not found. Copy env.example to .env and fill in your keys."
    exit 1
fi

if grep -q "REPLACE_WITH_YOUR_KEY" .env; then
    echo ""
    echo "  !! ACTION REQUIRED: Open .env and replace ANTHROPIC_API_KEY value."
    echo "     The Opus judge requires a valid Anthropic API key to score responses."
    echo ""
fi

# --- 4. Check LM Studio connectivity ---
echo "[4/4] Checking LM Studio local server..."
lm_url=$(grep LM_STUDIO_BASE_URL .env | cut -d'=' -f2)
lm_url="${lm_url:-http://localhost:1234/v1}"

if curl -s --connect-timeout 3 "${lm_url}/models" > /dev/null 2>&1; then
    echo "      LM Studio reachable at: $lm_url"
    echo ""
    echo "      Loaded models:"
    curl -s "${lm_url}/models" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    if models:
        for m in models:
            print(f'        - {m[\"id\"]}')
    else:
        print('        (no models loaded — load a model in LM Studio first)')
except Exception as e:
    print(f'        Could not parse model list: {e}')
"
else
    echo ""
    echo "  !! LM Studio server not reachable at: $lm_url"
    echo "     Make sure LM Studio is open and the local server is started:"
    echo "     LM Studio > Settings (gear icon) > Local Server > Start Server"
    echo ""
fi

echo ""
echo "=== Setup complete. ==="
echo ""
echo "Usage:"
echo "  # Discover loaded models (verify LM Studio is configured)"
echo "  python3 harness.py list-models"
echo ""
echo "  # Run full eval against LM Studio models only"
echo "  python3 harness.py run --backends lm_studio"
echo ""
echo "  # Run a single category (faster)"
echo "  python3 harness.py run --backends lm_studio --categories vuln_analysis"
echo ""
echo "  # Run all backends (LM Studio + Anthropic)"
echo "  python3 harness.py run"
echo ""
