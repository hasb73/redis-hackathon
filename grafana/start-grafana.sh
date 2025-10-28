#!/bin/bash

# Grafana Anomaly Detection Dashboard Startup Script

set -e

echo "🚀 Starting Grafana Anomaly Detection Dashboards..."

# Check if anomaly_detection.db exists
if [ ! -f "../anomaly_detection.db" ]; then
    echo "❌ Error: anomaly_detection.db not found in parent directory"
    echo "Please ensure the SQLite database file is present in the project root"
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p dashboards
mkdir -p provisioning/datasources
mkdir -p provisioning/dashboards

# Set proper permissions for the database file
echo "🔐 Setting database permissions..."
chmod 644 ../anomaly_detection.db

# Start Grafana using Docker Compose
echo "🐳 Starting Grafana container..."
docker-compose -f docker-compose-grafana.yml up -d

# Wait for Grafana to be ready
echo "⏳ Waiting for Grafana to be ready..."
timeout=60
counter=0

while [ $counter -lt $timeout ]; do
    if curl -s -f http://localhost:3000/api/health > /dev/null 2>&1; then
        echo "✅ Grafana is ready!"
        break
    fi
    
    echo "Waiting for Grafana... ($counter/$timeout)"
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -ge $timeout ]; then
    echo "❌ Timeout waiting for Grafana to start"
    echo "Check logs with: docker-compose -f docker-compose-grafana.yml logs grafana"
    exit 1
fi

echo ""
echo "🎉 Grafana Anomaly Detection Dashboards are now running!"
echo ""
echo "📊 Access Grafana at: http://localhost:3000"
echo "👤 Username: admin"
echo "🔑 Password: admin123"
echo ""
echo "🛠️  Useful commands:"
echo "  View logs:    docker-compose -f docker-compose-grafana.yml logs -f grafana"
echo "  Stop Grafana: docker-compose -f docker-compose-grafana.yml down"
echo "  Restart:      docker-compose -f docker-compose-grafana.yml restart"
echo ""