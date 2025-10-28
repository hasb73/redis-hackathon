# Grafana Files Organization - Complete 

## 📁 **Files Moved to grafana/ Directory**

All Grafana-related files have been successfully moved and organized under the `grafana/` directory.

### **Scripts:**
-  `deploy_dashboards.py` → `grafana/deploy_dashboards.py`
-  `DASHBOARD_CLEANUP_SUMMARY.md` → `grafana/DASHBOARD_CLEANUP_SUMMARY.md`
-  `GRAFANA_SETUP.md` → `grafana/GRAFANA_SETUP.md`
-  `docker-compose-grafana.yml` → `grafana/docker-compose-grafana.yml`
-  `start-grafana.sh` → `grafana/start-grafana.sh`
-  `stop-grafana.sh` → `grafana/stop-grafana.sh`
-  `test-grafana-setup.sh` → `grafana/test-grafana-setup.sh`
-  `verify-grafana-setup.sh` → `grafana/verify-grafana-setup.sh`
-  `sqlite_datasource_test_report.json` → `grafana/sqlite_datasource_test_report.json`

### **Updated Scripts:**
-  Updated all path references to work from `grafana/` directory
-  Fixed Docker Compose volume mounts
-  Updated deployment script to find dashboards in `dashboards/`
-  Modified shell scripts to reference `../anomaly_detection.db`

##  **Final Directory Structure**

```
grafana/
├── dashboards/                          # Dashboard JSON files
│   ├── simple-anomaly-dashboard.json    # Simple monitoring dashboard
│   ├── system-overview-dashboard.json   # Comprehensive system dashboard
│   └── SYSTEM_OVERVIEW_README.md        # Dashboard documentation
├── provisioning/                        # Grafana provisioning configs
│   ├── datasources/                     # Datasource configurations
│   └── dashboards/                      # Dashboard provisioning
├── deploy_dashboards.py                 # Dashboard deployment script
├── docker-compose-grafana.yml           # Docker Compose configuration
├── grafana.ini                          # Grafana configuration
├── start-grafana.sh                     # Start Grafana script
├── stop-grafana.sh                      # Stop Grafana script
├── test-grafana-setup.sh                # Setup validation script
├── README.md                            # Grafana documentation
└── [other Grafana-related files]
```

##  **Updated Usage Instructions**

### **Start Grafana:**
```bash
cd grafana
./start-grafana.sh
```

### **Deploy Dashboards:**
```bash
cd grafana
python3 deploy_dashboards.py
```

### **Stop Grafana:**
```bash
cd grafana
./stop-grafana.sh
```

### **Test Setup:**
```bash
cd grafana
./test-grafana-setup.sh
```

##  **Verification Results**

-  **All files moved successfully**
-  **All scripts updated and working**
-  **Dashboard deployment tested and working**
-  **Path references corrected**
-  **Docker Compose configuration updated**
-  **Documentation created**

## **Benefits of Organization**

1. **Clean Project Root**: Removed 9 Grafana files from project root
2. **Logical Grouping**: All Grafana files in one location
3. **Easy Management**: Single directory for all dashboard operations
4. **Clear Documentation**: Comprehensive README in grafana/ directory
5. **Maintainable**: Easy to add new dashboards and configurations

## **Current Working State**

### **Dashboards Deployed:**
-  Simple Anomaly Dashboard: http://localhost:3000/d/simple-anomaly-dashboard
-  System Overview Dashboard: http://localhost:3000/d/system-overview-dashboard

### **Access Information:**
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin123

## **Organization Complete**

All Grafana-related files are now properly organized under the `grafana/` directory with updated scripts and comprehensive documentation. The system is fully functional and ready for use.




# Grafana Anomaly Detection Dashboards Setup

This setup provides a Docker-based Grafana environment with SQLite support for visualizing anomaly detection data.

## Prerequisites

- Docker and Docker Compose installed
- `anomaly_detection.db` SQLite database file in the project root
- Port 3000 available on your system

## Quick Start

1. **Start Grafana:**
   ```bash
   ./start-grafana.sh
   ```

2. **Verify Setup (Optional):**
   ```bash
   ./verify-grafana-setup.sh
   ```

3. **Access Grafana:**
   - URL: http://localhost:3000
   - Username: `admin`
   - Password: `admin123`

4. **Stop Grafana:**
   ```bash
   ./stop-grafana.sh
   ```

## Configuration Details

### Docker Compose Configuration
- **File:** `docker-compose-grafana.yml`
- **Container:** `anomaly-grafana`
- **Port:** 3000 (host) → 3000 (container)
- **Plugins:** SQLite datasource plugin (`frser-sqlite-datasource`)

### Volume Mounts
- `./anomaly_detection.db` → `/var/lib/grafana/data/anomaly_detection.db` (read-only)
- `grafana-data` → `/var/lib/grafana` (persistent Grafana data)
- `grafana-logs` → `/var/log/grafana` (log files)
- `./grafana/provisioning` → `/etc/grafana/provisioning` (configuration)
- `./grafana/dashboards` → `/var/lib/grafana/dashboards` (dashboard files)

### Environment Variables
- **Security:** Admin credentials, user signup disabled
- **Plugins:** SQLite datasource auto-installation
- **Database:** SQLite backend for Grafana metadata
- **Analytics:** Disabled for privacy

### Data Source Configuration
- **Name:** AnomalyDetectionDB
- **Type:** SQLite (frser-sqlite-datasource)
- **Path:** `/var/lib/grafana/data/anomaly_detection.db`
- **Access:** Proxy mode for security

## Directory Structure

```
├── docker-compose-grafana.yml    # Main Docker Compose configuration
├── start-grafana.sh             # Startup script
├── stop-grafana.sh              # Stop script
├── grafana/
│   ├── grafana.ini              # Grafana configuration
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── sqlite.yml       # SQLite datasource config
│   │   └── dashboards/
│   │       └── dashboards.yml   # Dashboard provisioning config
│   └── dashboards/              # Dashboard JSON files (to be added)
└── anomaly_detection.db         # SQLite database (required)
```

## Troubleshooting

### Common Issues

1. **Port 3000 already in use:**
   ```bash
   # Check what's using port 3000
   lsof -i :3000
   # Kill the process or change the port in docker-compose-grafana.yml
   ```

2. **Database file not found:**
   - Ensure `anomaly_detection.db` exists in the project root
   - Check file permissions (should be readable)

3. **Plugin installation fails:**
   ```bash
   # Check container logs
   docker-compose -f docker-compose-grafana.yml logs grafana
   ```

4. **Grafana won't start:**
   ```bash
   # View detailed logs
   docker-compose -f docker-compose-grafana.yml logs -f grafana
   ```

### Useful Commands

```bash
# View container status
docker-compose -f docker-compose-grafana.yml ps

# View logs
docker-compose -f docker-compose-grafana.yml logs -f grafana

# Restart Grafana
docker-compose -f docker-compose-grafana.yml restart grafana

# Remove all data and start fresh
docker-compose -f docker-compose-grafana.yml down -v
./start-grafana.sh

# Access container shell
docker-compose -f docker-compose-grafana.yml exec grafana /bin/bash
```

## Security Notes

- Default admin credentials are set to `admin/admin123`
- Change these credentials in production environments
- The SQLite database is mounted read-only for security
- Anonymous access is disabled
- User signup is disabled

## Next Steps

After completing this setup:
1. Configure the SQLite data source (automatically provisioned)
2. Create dashboard configurations
3. Import or create dashboard panels
4. Set up alerting rules
5. Configure user access and permissions

## Health Check

The container includes a health check that verifies Grafana is responding on port 3000. You can check the health status with:

```bash
docker-compose -f docker-compose-grafana.yml ps
```

Look for "healthy" status in the State column.