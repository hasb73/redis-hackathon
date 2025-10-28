# Grafana Provisioning

## Overview
Automated Grafana configuration for zero-touch dashboard and datasource deployment.

## Directories
- `dashboards/` - Dashboard provisioning configuration
- `datasources/` - Datasource connection configuration

## Files
- `dashboards/dashboards.yml` - Dashboard auto-discovery configuration
- `datasources/sqlite.yml` - SQLite database connection settings

## Features
- **Automatic Discovery**: Dashboards loaded on Grafana startup
- **Database Integration**: Pre-configured SQLite datasource
- **Zero Configuration**: No manual setup required
- **Version Control**: Configuration as code

## Usage
Provisioning is automatically activated when Grafana starts via Docker Compose:
```bash
docker-compose up -d grafana
```

## Configuration
- **Dashboard Path**: `/var/lib/grafana/dashboards/`
- **Datasource**: SQLite connection to anomaly detection database
- **Auto-reload**: Changes detected and applied automatically

## Validation
Test provisioning setup:
```bash
./grafana/test-grafana-setup.sh
```
