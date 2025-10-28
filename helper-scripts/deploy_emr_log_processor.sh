#!/bin/bash
"""
EMR HDFS Log Processor Deployment Script
Deploys real-time HDFS log processing on EMR DataNode
"""

set -e

# Configuration
LOG_DIR="/var/log/hadoop-hdfs"
SERVICE_USER="hadoop"
INSTALL_DIR="/opt/hdfs-anomaly-detection"
SERVICE_NAME="hdfs-log-processor"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m' 
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Detect HDFS DataNode log file
detect_hdfs_log_file() {
    print_status "Detecting HDFS DataNode log file..."
    
    # Common patterns for HDFS DataNode logs
    LOG_PATTERNS=(
        "${LOG_DIR}/hadoop-hdfs-datanode-*.log"
        "${LOG_DIR}/hadoop-hdfs-datanode*.log"
        "/var/log/hadoop-hdfs/datanode/*.log"
        "/var/log/hadoop/hdfs/datanode/*.log"
    )
    
    for pattern in "${LOG_PATTERNS[@]}"; do
        matches=($(ls $pattern 2>/dev/null || true))
        if [ ${#matches[@]} -gt 0 ]; then
            HDFS_LOG_FILE="${matches[0]}"
            print_success "Found HDFS DataNode log: $HDFS_LOG_FILE"
            return 0
        fi
    done
    
    print_error "HDFS DataNode log file not found. Please specify manually."
    exit 1
}


# Setup application directory
setup_app_directory() {
    print_status "Setting up application directory..."
    
    # Create directory structure
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$INSTALL_DIR/logs"
    mkdir -p "$INSTALL_DIR/config"
    
    # Set permissions
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    chmod 755 "$INSTALL_DIR"
    
    print_success "Application directory created: $INSTALL_DIR"
}


# Create configuration file
create_config() {
    print_status "Creating configuration file..."
    
    cat > "$INSTALL_DIR/config/processor_config.json" << EOF
{
    "log_file_path": "$HDFS_LOG_FILE",
    "kafka_servers": "localhost:9092",
    "kafka_topic": "logs",
    "scoring_service_url": "http://localhost:8003",
    "initial_lines": 0,
    "stats_interval": 60,
    "log_level": "INFO",
    "description": "initial_lines=0 means only NEW log entries are processed, not existing ones"
}
EOF
    
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/config/processor_config.json"
    
    print_success "Configuration file created"
}

# Deploy application files
deploy_app_files() {
    print_status "Deploying application files..."
    
    # Check if hdfs_production_log_processor.py exists in current directory
    if [ ! -f "hdfs_production_log_processor.py" ]; then
        print_error "hdfs_production_log_processor.py not found in current directory"
        print_error "Please run this script from the directory containing the processor script"
        exit 1
    fi
    
    # Copy application files
    cp hdfs_production_log_processor.py "$INSTALL_DIR/"
    
    # Set permissions
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/hdfs_production_log_processor.py"
    chmod 755 "$INSTALL_DIR/hdfs_production_log_processor.py"
    
    print_success "Application files deployed"
}

# Create systemd service
create_systemd_service() {
    print_status "Creating systemd service..."
    
    cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=HDFS Real-time Log Processor for Anomaly Detection
After=network.target
Wants=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 hdfs_production_log_processor.py $HDFS_LOG_FILE localhost:9092 logs http://localhost:8003
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
TimeoutStartSec=30
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    systemctl daemon-reload
    
    print_success "Systemd service created: $SERVICE_NAME"
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > "$INSTALL_DIR/start_processor.sh" << EOF
#!/bin/bash
# HDFS Log Processor Startup Script

INSTALL_DIR="$INSTALL_DIR"
CONFIG_FILE="\$INSTALL_DIR/config/processor_config.json"

# Load configuration
if [ -f "\$CONFIG_FILE" ]; then
    LOG_FILE=\$(python3 -c "import json; print(json.load(open('\$CONFIG_FILE'))['log_file_path'])")
    KAFKA_SERVERS=\$(python3 -c "import json; print(json.load(open('\$CONFIG_FILE'))['kafka_servers'])")
    KAFKA_TOPIC=\$(python3 -c "import json; print(json.load(open('\$CONFIG_FILE'))['kafka_topic'])")
    SCORING_SERVICE=\$(python3 -c "import json; print(json.load(open('\$CONFIG_FILE'))['scoring_service_url'])")
else
    LOG_FILE="$HDFS_LOG_FILE"
    KAFKA_SERVERS="localhost:9092"
    KAFKA_TOPIC="logs"
    SCORING_SERVICE="http://localhost:8003"
fi

echo "Starting HDFS Log Processor..."
echo "Log File: \$LOG_FILE"
echo "Kafka: \$KAFKA_SERVERS"
echo "Topic: \$KAFKA_TOPIC"
echo "Scoring Service: \$SCORING_SERVICE"

cd "\$INSTALL_DIR"
/usr/bin/python3 hdfs_production_log_processor.py "\$LOG_FILE" "\$KAFKA_SERVERS" "\$KAFKA_TOPIC" "\$SCORING_SERVICE"
EOF
    
    chmod +x "$INSTALL_DIR/start_processor.sh"
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/start_processor.sh"
    
    print_success "Startup script created"
}

# Validate installation
validate_installation() {
    print_status "Validating installation..."
    
    # Check Python
    if ! python3 --version; then
        print_error "Python 3 not found"
        exit 1
    fi
    
    # Check log file access
    if ! sudo -u "$SERVICE_USER" test -r "$HDFS_LOG_FILE"; then
        print_warning "Log file not readable by $SERVICE_USER user"
        print_warning "You may need to adjust permissions: chmod 644 $HDFS_LOG_FILE"
    fi
    
    print_success "Installation validation completed"
    print_warning "Remember to install Python dependencies manually:"
    print_warning "pip3 install kafka-python requests pandas urllib3 watchdog"
}

# Main deployment function
main() {
    print_status "Starting HDFS Log Processor deployment on EMR..."
    
    # Parse command line arguments
    if [ $# -gt 0 ]; then
        HDFS_LOG_FILE="$1"
        print_status "Using specified log file: $HDFS_LOG_FILE"
    else
        detect_hdfs_log_file
    fi
    
    # Run deployment steps
    check_root
    setup_app_directory
    deploy_app_files
    create_config
    create_systemd_service
    create_startup_script
    validate_installation
    
    print_success "HDFS Log Processor deployment completed!"
    print_status ""
    print_warning "IMPORTANT: Install Python dependencies first:"
    print_warning "pip3 install kafka-python requests pandas urllib3 watchdog"
    print_status ""
    print_status "Next steps:"
    print_status "1. Install Python dependencies (see above)"
    print_status "2. Start the service: systemctl start $SERVICE_NAME"
    print_status "3. Enable auto-start: systemctl enable $SERVICE_NAME"
    print_status "4. Check status: systemctl status $SERVICE_NAME"
    print_status "5. View logs: journalctl -u $SERVICE_NAME -f"
    print_status ""
    print_status "Manual start: $INSTALL_DIR/start_processor.sh"
    print_status "Configuration: $INSTALL_DIR/config/processor_config.json"
    print_status ""
    print_success "The processor will ONLY process NEW log entries as they arrive in:"
    print_success "$HDFS_LOG_FILE"
}

# Run main function
main "$@"
