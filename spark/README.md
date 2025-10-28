# Cloud Deployment

## Overview
This contains preprocessing and spark scripts to be used on the HDFS cluster's datanode in AWS EMR


## Files


- `spark_job.py` -  custom spark job is created which acts as a consumer to the logs exposed on “logs” kafka topic.
- - Consume the log entries from kafka in batches and generate the embeddings by calling the embedding service
- - Store the embeddings in qdrant DB volume



## Dependencies

- Anomaly detection service should be running on port 8000 with the loaded ensemble model
- Qdrant should be available and running on port  6333
- Kafka broker (producer) should be available and running on port 9092
- Embedding service should be available and running on port 8000
- Redis caching should be initialized
- Grafana is available to visualise the performance




## Usage
```bash
# Run the log preprocessor that listens to the hadoop logs
hdfs_production_log_processor.py /var/log/hadoop-hdfs/hadoop-hdfs-datanode-ip-172-31-39-152.eu-west-1.compute.internal.log

# Run production Spark job
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0 spark_job.py

#Restart the anomaly detection service
sh restart_anomaly_detection_service.sh 


## run HDFS commands to generate anomalies 

hadoop daemonlog -setlevel 0.0.0.0:9864 org.apache.hadoop.hdfs.server.datanode.BlockReceiver DEBUG


#ON data node 

echo "Test data for corruption simulation" > /tmp/testfile.txt
hdfs dfs -mkdir -p /test/corruption
hdfs dfs -put /tmp/testfile.txt /test/corruption/testfile.txt
hdfs fsck /test/corruption/testfile.txt -files -blocks -locations

#Add corruption to block 

sudo dd if=/dev/urandom of=/mnt/hdfs/current/BP-904282469-172.31.36.192-1758638658492/current/finalized/subdir0/subdir0/blk_1073742024 bs=1 count=10 seek=10 conv=notrunc

#Command to cat corrupted file

hdfs dfs -cat /test/corruption/testfile.txt

#Log Line generated in /var/log/hadoop-hdfs/hadoop-hdfs-datanode-ip-172-31-39-152.eu-west-1.compute.internal.log


2025-09-25 09:30:54,737 DEBUG org.apache.hadoop.hdfs.server.datanode.DataNode.clienttrace (DataXceiver for client DFSClient_NONMAPREDUCE_-56011767_1 at /172.31.36.192:38172 [Sending block BP-904282469-172.31.36.192-1758638658492:blk_1073742023_1199]): src: /172.31.39.152:9866, dest: /172.31.36.192:38172, volume: , bytes: 40, op: HDFS_READ, cliID: DFSClient_NONMAPREDUCE_-56011767_1, offset: 0, srvID: 4afa7677-552a-4246-84f6-54ae00a35b76, blockid: BP-904282469-172.31.36.192-1758638658492:blk_1073742023_1199, duration(ns): 119030
2025-09-25 09:30:54,745 DEBUG org.apache.hadoop.hdfs.server.datanode.DataNode (DataXceiver for client DFSClient_NONMAPREDUCE_-56011767_1 at /172.31.36.192:38172 [Sending block BP-904282469-172.31.36.192-1758638658492:blk_1073742023_1199]): Error reading client status response. Will close connection.
java.net.SocketException: Connection reset
        at java.base/sun.nio.ch.SocketChannelImpl.throwConnectionReset(SocketChannelImpl.java:394)
        at java.base/sun.nio.ch.SocketChannelImpl.read(SocketChannelImpl.java:426)
        at org.apache.hadoop.net.SocketInputStream$Reader.performIO(SocketInputStream.java:57)
        at org.apache.hadoop.net.SocketIOWithTimeout.doIO(SocketIOWithTimeout.java:141)
        at org.apache.hadoop.net.SocketInputStream.read(SocketInputStream.java:161)
        at org.apache.hadoop.net.SocketInputStream.read(SocketInputStream.java:131)

```
