# Helper Scripts

## Overview
Utility scripts for system management, debugging, and operational tasks.

## Files
- `qdrant_entries.py` - View and inspect Qdrant database entries
- `qdrant_embeddings.py` - Analyze and debug embeddings in Qdrant
- `manage_qdrant.py` - Comprehensive Qdrant database management
- `view_qdrant.py` - Interactive Qdrant data visualization
- `analyze_embeddings.py` - Embedding quality analysis and metrics
- `clear_qdrant.py` - Database cleanup and reset utilities
- `deploy_emr_log_processor.sh` - Automated EMR deployment script
- `stop_scoring_service.sh` - Service shutdown utility
- `commands.sh` - Common operational commands
- `scoring.log` - Scoring service operation logs

## Features
- **Database Management**: Complete Qdrant CRUD operations
- **Embedding Analysis**: Vector quality assessment and visualization
- **Deployment Automation**: One-click EMR and service deployment
- **System Monitoring**: Service health checks and log analysis
- **Debugging Tools**: Troubleshooting utilities for development

## Usage
```bash
# View Qdrant entries
python3 helper-scripts/qdrant_entries.py

# Analyze embedding quality
python3 helper-scripts/analyze_embeddings.py

# Clear Qdrant database
python3 helper-scripts/clear_qdrant.py

# Deploy to EMR
./helper-scripts/deploy_emr_log_processor.sh

# Stop scoring service
./helper-scripts/stop_scoring_service.sh
```

## Categories
- **Qdrant Tools**: Database management and analysis
- **Deployment Scripts**: Automation for cloud deployment
- **Service Management**: Start/stop utilities for services
- **Analysis Tools**: Data quality and performance analysis
