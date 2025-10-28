#!/bin/bash

# Comprehensive Grafana setup verification script

set -e

echo "🔍 Verifying Grafana Anomaly Detection Setup..."

# Test 1: Check if Grafana is running
echo ""
echo "1️⃣ Checking Grafana container status..."
if docker-compose -f docker-compose-grafana.yml ps | grep -q "healthy"; then
    echo "✅ Grafana container is running and healthy"
else
    echo "❌ Grafana container is not healthy"
    exit 1
fi

# Test 2: Check HTTP endpoint
echo ""
echo "2️⃣ Testing Grafana HTTP endpoint..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health)
if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Grafana API is responding (HTTP $HTTP_STATUS)"
else
    echo "❌ Grafana API is not responding (HTTP $HTTP_STATUS)"
    exit 1
fi

# Test 3: Check authentication
echo ""
echo "3️⃣ Testing Grafana authentication..."
AUTH_TEST=$(curl -s -u admin:admin123 http://localhost:3000/api/user | grep -o '"login":"admin"' || echo "failed")
if [ "$AUTH_TEST" = '"login":"admin"' ]; then
    echo "✅ Authentication working correctly"
else
    echo "❌ Authentication failed"
    exit 1
fi

# Test 4: Check SQLite datasource
echo ""
echo "4️⃣ Verifying SQLite datasource configuration..."
DATASOURCE_TEST=$(curl -s -u admin:admin123 http://localhost:3000/api/datasources | grep -o '"name":"AnomalyDetectionDB"' || echo "failed")
if [ "$DATASOURCE_TEST" = '"name":"AnomalyDetectionDB"' ]; then
    echo "✅ SQLite datasource is configured"
else
    echo "❌ SQLite datasource not found"
    exit 1
fi

# Test 5: Check SQLite plugin
echo ""
echo "5️⃣ Checking SQLite plugin installation..."
PLUGIN_TEST=$(curl -s -u admin:admin123 http://localhost:3000/api/plugins | grep -o '"id":"frser-sqlite-datasource"' || echo "failed")
if [ "$PLUGIN_TEST" = '"id":"frser-sqlite-datasource"' ]; then
    echo "✅ SQLite plugin is installed and active"
else
    echo "❌ SQLite plugin not found"
    exit 1
fi

# Test 6: Test database connectivity
echo ""
echo "6️⃣ Testing database connectivity..."
DB_TEST=$(curl -s -u admin:admin123 -H "Content-Type: application/json" -X POST "http://localhost:3000/api/ds/query" -d '{
  "queries": [
    {
      "datasource": {"uid": "P7BEE95809231D024"},
      "rawSql": "SELECT 1 as test_connection;",
      "format": "table"
    }
  ]
}' | grep -o '"status":200' || echo "failed")

if [ "$DB_TEST" = '"status":200' ]; then
    echo "✅ Database connectivity test passed"
else
    echo "❌ Database connectivity test failed"
    exit 1
fi

# Test 7: Check volume mounts
echo ""
echo "7️⃣ Verifying volume mounts..."
VOLUME_TEST=$(docker-compose -f docker-compose-grafana.yml exec -T grafana ls -la /var/lib/grafana/data/anomaly_detection.db 2>/dev/null | grep -o "anomaly_detection.db" || echo "failed")
if [ "$VOLUME_TEST" = "anomaly_detection.db" ]; then
    echo "✅ Database file is properly mounted"
else
    echo "❌ Database file mount failed"
    exit 1
fi

echo ""
echo "🎉 All tests passed! Grafana setup is working correctly."
echo ""
echo "📊 Access Information:"
echo "  URL: http://localhost:3000"
echo "  Username: admin"
echo "  Password: admin123"
echo "  Datasource: AnomalyDetectionDB (SQLite)"
echo ""
echo "🛠️  Management Commands:"
echo "  View logs: docker-compose -f docker-compose-grafana.yml logs -f grafana"
echo "  Restart: docker-compose -f docker-compose-grafana.yml restart"
echo "  Stop: ./stop-grafana.sh"