#!/bin/bash
set -e
echo "=== Starting MediVan ==="

# Check Tailscale
if command -v tailscale &> /dev/null; then
    TS_IP=$(tailscale ip -4 2>/dev/null || echo "not connected")
    echo "Tailscale IP: $TS_IP"
else
    echo "Tailscale: not installed (local access only)"
fi

# Get local IP
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
echo "Local IP: $LOCAL_IP"

# Start server
echo ""
echo "Starting MediVan server..."
echo "  Local:     http://$LOCAL_IP:8000"
if [ "$TS_IP" != "not connected" ] && [ -n "$TS_IP" ]; then
    echo "  Tailscale: http://$TS_IP:8000"
fi
echo ""

cd "$(dirname "$0")/.."
export MOCK_MODE=${MOCK_MODE:-true}
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
