#!/bin/bash

# Test script for Grafana setup

set -e

echo "🧪 Testing Grafana Docker setup..."

# Test 1: Check if required files exist
echo "📋 Checking required files..."
required_files=(
    "docker-compose-grafana.yml"
    "../anomaly_detection.db"
    "provisioning/datasources/sqlite.yml"
    "provisioning/dashboards/dashboards.yml"
    "start-grafana.sh"
    "stop-grafana.sh"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Test 2: Validate Docker Compose configuration
echo ""
echo "🐳 Validating Docker Compose configuration..."
if docker-compose -f docker-compose-grafana.yml config --quiet; then
    echo "✅ Docker Compose configuration is valid"
else
    echo "❌ Docker Compose configuration is invalid"
    exit 1
fi

# Test 3: Check if scripts are executable
echo ""
echo "🔐 Checking script permissions..."
if [ -x "start-grafana.sh" ]; then
    echo "✅ start-grafana.sh is executable"
else
    echo "❌ start-grafana.sh is not executable"
    exit 1
fi

if [ -x "stop-grafana.sh" ]; then
    echo "✅ stop-grafana.sh is executable"
else
    echo "❌ stop-grafana.sh is not executable"
    exit 1
fi

# Test 4: Check database file permissions
echo ""
echo "📊 Checking database file..."
if [ -r "../anomaly_detection.db" ]; then
    echo "✅ anomaly_detection.db is readable"
else
    echo "❌ anomaly_detection.db is not readable"
    exit 1
fi

# Test 5: Validate provisioning files
echo ""
echo "⚙️  Validating provisioning configurations..."

# Check datasource config
if grep -q "frser-sqlite-datasource" provisioning/datasources/sqlite.yml; then
    echo "✅ SQLite datasource configuration is correct"
else
    echo "❌ SQLite datasource configuration is missing or incorrect"
    exit 1
fi

# Check dashboard config
if grep -q "anomaly-dashboards" provisioning/dashboards/dashboards.yml; then
    echo "✅ Dashboard provisioning configuration is correct"
else
    echo "❌ Dashboard provisioning configuration is missing or incorrect"
    exit 1
fi

echo ""
echo "🎉 All tests passed! Grafana setup is ready."
echo ""
echo "🚀 To start Grafana, run: ./start-grafana.sh"
echo "📊 Access URL: http://localhost:3000"
echo "👤 Username: admin"
echo "🔑 Password: admin123"