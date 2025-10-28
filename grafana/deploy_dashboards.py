#!/usr/bin/env python3
"""
General Dashboard Deployment Script
Deploys all dashboards from dashboards/ directory to Grafana

Usage: Run from the grafana/ directory
    cd grafana
    python3 deploy_dashboards.py
"""

import os
import json
import requests
import glob

def deploy_dashboard(dashboard_file):
    """Deploy a single dashboard file to Grafana"""
    
    try:
        with open(dashboard_file, 'r') as f:
            dashboard_data = json.load(f)
        
        # Prepare the payload for Grafana API
        payload = {
            "dashboard": dashboard_data,
            "overwrite": True
        }
        
        # Deploy to Grafana
        response = requests.post(
            "http://localhost:3000/api/dashboards/db",
            json=payload,
            auth=('admin', 'admin123'),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            dashboard_name = dashboard_data.get('title', 'Unknown')
            print(f"✅ Deployed: {dashboard_name}")
            print(f"   URL: http://localhost:3000{result['url']}")
            return True
        else:
            print(f"❌ Failed to deploy {dashboard_file}: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error deploying {dashboard_file}: {e}")
        return False

def deploy_all_dashboards():
    """Deploy all dashboards from grafana/dashboards directory"""
    
    print("🚀 Deploying Dashboards to Grafana")
    print("=" * 50)
    
    # Find all JSON dashboard files
    dashboard_files = glob.glob("dashboards/*.json")
    
    if not dashboard_files:
        print("❌ No dashboard files found in dashboards/")
        return
    
    print(f"📊 Found {len(dashboard_files)} dashboard file(s):")
    for file in dashboard_files:
        print(f"   • {file}")
    
    print(f"\n🔄 Deploying dashboards...")
    
    deployed_count = 0
    for dashboard_file in dashboard_files:
        if deploy_dashboard(dashboard_file):
            deployed_count += 1
    
    print(f"\n📋 Deployment Summary:")
    print(f"   ✅ Successfully deployed: {deployed_count}/{len(dashboard_files)}")
    
    if deployed_count > 0:
        print(f"\n🌐 Access Grafana:")
        print(f"   URL: http://localhost:3000")
        print(f"   Username: admin")
        print(f"   Password: admin123")
    
    return deployed_count

def main():
    """Main deployment function"""
    
    # Check if Grafana is running
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Grafana is not responding. Please start Grafana first.")
            print("💡 Run: ./start-grafana.sh")
            return
    except:
        print("❌ Cannot connect to Grafana. Please start Grafana first.")
        print("💡 Run: ./start-grafana.sh")
        return
    
    # Deploy all dashboards
    deploy_all_dashboards()

if __name__ == "__main__":
    main()
