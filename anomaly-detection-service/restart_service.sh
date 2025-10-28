#!/bin/bash

# HDFS Anomaly Detection Engine V2 - Background Starter Script
# This script starts the enhanced scoring service in the background
echo "stopping amomaly detection on 8003"

lsof -ti:8003 | xargs kill -9

# Run in background and redirect output to log file
nohup /usr/bin/python3 ./anomaly-detection-service/anomaly_detection_service.py --background 2>&1 &

sleep 3

curl  -X POST localhost:8003/cache/clear