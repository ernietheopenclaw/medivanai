#!/bin/bash
set -e
echo "=== MediVan Tailscale Setup ==="

# Install Tailscale
if ! command -v tailscale &> /dev/null; then
    echo "Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# Start Tailscale
echo "Starting Tailscale..."
sudo tailscale up

# Get IP
TS_IP=$(tailscale ip -4)
echo ""
echo "=== Tailscale Ready ==="
echo "Tailscale IP: $TS_IP"
echo "MediVan URL:  http://$TS_IP:8000"
echo ""
echo "Optional: Enable HTTPS with:"
echo "  sudo tailscale cert $TS_IP"
echo ""
echo "On your phone:"
echo "  1. Install Tailscale app"
echo "  2. Join the same Tailnet"
echo "  3. Open http://$TS_IP:8000 in Chrome"
