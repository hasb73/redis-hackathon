#!/bin/bash

# Grafana Anomaly Detection Dashboard Stop Script

set -e

echo "🛑 Stopping Grafana Anomaly Detection Dashboards..."

# Stop Grafana using Docker Compose
docker-compose -f docker-compose-grafana.yml down

echo "✅ Grafana has been stopped successfully!"
echo ""
echo "💡 To start again, run: ./start-grafana.sh"
echo "🗑️  To remove all data, run: docker-compose -f docker-compose-grafana.yml down -v"