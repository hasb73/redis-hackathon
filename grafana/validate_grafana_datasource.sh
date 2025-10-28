#!/bin/bash

# Grafana SQLite Data Source Validation Script
# This script validates the Grafana SQLite data source configuration

set -e

echo "=== Grafana SQLite Data Source Validation ==="
echo "Timestamp: $(date)"

# Configuration variables
GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="admin123"
DB_PATH="./anomaly_detection.db"
CONTAINER_DB_PATH="/var/lib/grafana/data/anomaly_detection.db"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    local status=$1
    local message=$2
    case $status in
        "success")
            echo -e "${GREEN}✓${NC} $message"
            ;;
        "warning")
            echo -e "${YELLOW}⚠${NC} $message"
            ;;
        "error")
            echo -e "${RED}✗${NC} $message"
            ;;
    esac
}

# Function to check if Grafana is running
check_grafana_status() {
    echo "Checking Grafana service status..."
    
    if docker ps | grep -q "anomaly-grafana"; then
        print_status "success" "Grafana container is running"
        
        # Check if Grafana API is responding
        if curl -s -f "$GRAFANA_URL/api/health" > /dev/null 2>&1; then
            print_status "success" "Grafana API is responding"
            return 0
        else
            print_status "warning" "Grafana container running but API not responding yet"
            echo "Waiting for Grafana to start up..."
            sleep 10
            
            if curl -s -f "$GRAFANA_URL/api/health" > /dev/null 2>&1; then
                print_status "success" "Grafana API is now responding"
                return 0
            else
                print_status "error" "Grafana API still not responding"
                return 1
            fi
        fi
    else
        print_status "error" "Grafana container is not running"
        echo "Please start Grafana using: docker-compose -f docker-compose-grafana.yml up -d"
        return 1
    fi
}

# Function to validate database file accessibility
validate_database_access() {
    echo "Validating database file access..."
    
    # Check if database file exists
    if [ -f "$DB_PATH" ]; then
        print_status "success" "Database file exists: $DB_PATH"
    else
        print_status "error" "Database file not found: $DB_PATH"
        return 1
    fi
    
    # Check database file permissions
    if [ -r "$DB_PATH" ]; then
        print_status "success" "Database file is readable"
    else
        print_status "error" "Database file is not readable"
        return 1
    fi
    
    # Check if database is accessible from container
    if docker exec anomaly-grafana test -f "$CONTAINER_DB_PATH" 2>/dev/null; then
        print_status "success" "Database file accessible from Grafana container"
    else
        print_status "error" "Database file not accessible from Grafana container"
        echo "Check volume mount configuration in docker-compose-grafana.yml"
        return 1
    fi
    
    # Test database connectivity from container
    if docker exec anomaly-grafana sqlite3 "$CONTAINER_DB_PATH" "SELECT COUNT(*) FROM anomaly_detections;" > /dev/null 2>&1; then
        local record_count=$(docker exec anomaly-grafana sqlite3 "$CONTAINER_DB_PATH" "SELECT COUNT(*) FROM anomaly_detections;")
        print_status "success" "Database query successful from container ($record_count records)"
    else
        print_status "error" "Cannot query database from container"
        return 1
    fi
}

# Function to test Grafana data source API
test_datasource_api() {
    echo "Testing Grafana data source API..."
    
    # Get authentication token or use basic auth
    local auth_header="Authorization: Basic $(echo -n "$GRAFANA_USER:$GRAFANA_PASSWORD" | base64)"
    
    # List all data sources
    local datasources_response=$(curl -s -H "$auth_header" "$GRAFANA_URL/api/datasources")
    
    if echo "$datasources_response" | jq -e '.[] | select(.name=="AnomalyDetectionDB")' > /dev/null 2>&1; then
        print_status "success" "AnomalyDetectionDB data source found in Grafana"
        
        # Get data source details
        local datasource_id=$(echo "$datasources_response" | jq -r '.[] | select(.name=="AnomalyDetectionDB") | .id')
        print_status "success" "Data source ID: $datasource_id"
        
        # Test data source connection
        local test_response=$(curl -s -H "$auth_header" -X POST "$GRAFANA_URL/api/datasources/$datasource_id/health")
        
        if echo "$test_response" | jq -e '.status == "OK"' > /dev/null 2>&1; then
            print_status "success" "Data source connection test passed"
            return 0
        else
            print_status "error" "Data source connection test failed"
            echo "Response: $test_response"
            return 1
        fi
    else
        print_status "error" "AnomalyDetectionDB data source not found"
        echo "Available data sources:"
        echo "$datasources_response" | jq -r '.[].name' | sed 's/^/  - /'
        return 1
    fi
}

# Function to test sample queries
test_sample_queries() {
    echo "Testing sample queries through Grafana API..."
    
    local auth_header="Authorization: Basic $(echo -n "$GRAFANA_USER:$GRAFANA_PASSWORD" | base64)"
    
    # Get data source ID
    local datasources_response=$(curl -s -H "$auth_header" "$GRAFANA_URL/api/datasources")
    local datasource_uid=$(echo "$datasources_response" | jq -r '.[] | select(.name=="AnomalyDetectionDB") | .uid')
    
    if [ "$datasource_uid" = "null" ] || [ -z "$datasource_uid" ]; then
        print_status "error" "Could not get data source UID"
        return 1
    fi
    
    # Test queries
    local test_queries=(
        "SELECT COUNT(*) as total FROM anomaly_detections"
        "SELECT COUNT(*) as metrics FROM performance_metrics"
        "SELECT predicted_label, COUNT(*) as count FROM anomaly_detections GROUP BY predicted_label"
    )
    
    for query in "${test_queries[@]}"; do
        local query_payload=$(jq -n --arg uid "$datasource_uid" --arg sql "$query" '{
            "queries": [
                {
                    "datasource": {"uid": $uid},
                    "rawSql": $sql,
                    "format": "table"
                }
            ]
        }')
        
        local query_response=$(curl -s -H "$auth_header" -H "Content-Type: application/json" \
            -X POST "$GRAFANA_URL/api/ds/query" -d "$query_payload")
        
        if echo "$query_response" | jq -e '.results' > /dev/null 2>&1; then
            print_status "success" "Query executed: ${query:0:50}..."
        else
            print_status "error" "Query failed: ${query:0:50}..."
            echo "Response: $query_response"
        fi
    done
}

# Function to validate provisioning configuration
validate_provisioning() {
    echo "Validating provisioning configuration..."
    
    # Check if provisioning files exist
    if [ -f "grafana/provisioning/datasources/sqlite.yml" ]; then
        print_status "success" "Data source provisioning file exists"
        
        # Validate YAML syntax
        if python3 -c "import yaml; yaml.safe_load(open('grafana/provisioning/datasources/sqlite.yml'))" 2>/dev/null; then
            print_status "success" "Provisioning YAML syntax is valid"
        else
            print_status "error" "Provisioning YAML syntax is invalid"
            return 1
        fi
    else
        print_status "error" "Data source provisioning file not found"
        return 1
    fi
    
    # Check if provisioning directory is mounted in container
    if docker exec anomaly-grafana test -d "/etc/grafana/provisioning/datasources" 2>/dev/null; then
        print_status "success" "Provisioning directory mounted in container"
    else
        print_status "error" "Provisioning directory not mounted in container"
        return 1
    fi
}

# Main validation function
main() {
    local overall_success=true
    
    echo "Starting comprehensive Grafana SQLite data source validation..."
    echo
    
    # Run validation steps
    if ! validate_database_access; then
        overall_success=false
    fi
    echo
    
    if ! validate_provisioning; then
        overall_success=false
    fi
    echo
    
    if ! check_grafana_status; then
        overall_success=false
    fi
    echo
    
    if ! test_datasource_api; then
        overall_success=false
    fi
    echo
    
    if ! test_sample_queries; then
        overall_success=false
    fi
    echo
    
    # Summary
    echo "=== Validation Summary ==="
    if [ "$overall_success" = true ]; then
        print_status "success" "All validation tests passed"
        echo "Grafana SQLite data source is properly configured and ready for use"
        echo
        echo "Next steps:"
        echo "  1. Access Grafana at: $GRAFANA_URL"
        echo "  2. Login with: $GRAFANA_USER / $GRAFANA_PASSWORD"
        echo "  3. Create dashboards using the AnomalyDetectionDB data source"
        return 0
    else
        print_status "error" "Some validation tests failed"
        echo "Please review the errors above and fix the configuration"
        return 1
    fi
}

# Check dependencies
if ! command -v jq &> /dev/null; then
    print_status "error" "jq is required but not installed"
    echo "Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    print_status "error" "Docker is required but not installed"
    exit 1
fi

# Run main validation
main
exit $?