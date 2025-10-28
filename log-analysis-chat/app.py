#!/usr/bin/env python3
"""
Redis AI-Powered Log Analysis Chat Interface with RedisVL Integration
Intelligent chat assistant for log analysis and anomaly detection insights
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import redis
import json
import asyncio
import sqlite3
import time
from datetime import datetime, timedelta
import numpy as np
import logging
from pathlib import Path
import re
import requests

# RedisVL imports for vector similarity search
try:
    from redisvl.index import SearchIndex
    from redisvl.query import VectorQuery
    from redisvl.query.filter import Tag
    REDISVL_AVAILABLE = True
except ImportError:
    REDISVL_AVAILABLE = False
    print("Warning: RedisVL not available. Vector similarity search disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str
    timestamp: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    analysis_data: Optional[Dict[str, Any]] = None
    suggestions: List[str] = []
    timestamp: str

class LogAnalysisChatBot:
    """Redis AI-powered chat bot for log analysis with RedisVL integration"""
    
    def __init__(self):
        # Redis connections - use service names in Docker
        self.redis_stack = redis.Redis(host='redis-stack', port=6379, decode_responses=True)
        self.redis_ai = redis.Redis(host='redis-ai', port=6379, decode_responses=False)
        
        # Database connection - fix path for Docker environment
        self.db_path = "/app/anomaly-detection-service/anomaly_detection.db"
        
        # RedisVL configuration (matching your existing setup)
        self.redis_index_name = "logs_embeddings"
        self.redis_key_prefix = "log_entry:"
        self.embedding_service_url = "http://embedding:8000"
        
        # Initialize RedisVL search index
        self.search_index = None
        self.search_index_available = False
        self.init_redisvl()
        
        # Chat patterns and responses
        self.chat_patterns = {
            'anomaly_count': r'(?i)(how many|count|number).*anomal',
            'latest_anomalies': r'(?i)(latest|recent|new).*anomal',
            'anomaly_types': r'(?i)(what types?|kinds?).*anomal',
            'log_summary': r'(?i)(summary|overview|status).*log',
            'performance': r'(?i)(performance|speed|latency|time)',
            'accuracy': r'(?i)(accuracy|precision|recall|f1)',
            'help': r'(?i)(help|what can you|commands|options)',
            'trends': r'(?i)(trend|pattern|over time)',
            'errors': r'(?i)(error|failed|problem|issue)',
            'system_health': r'(?i)(health|status|system)',
            'similar_logs': r'(?i)(similar|like|find.*similar|pattern|search.*for)',
            'vector_search': r'(?i)(vector|semantic|embedding|similarity)',
            'query_logs': r'(?i)(show.*logs?|find.*logs?|get.*logs?|logs?.*contain|logs?.*with)',
        }
        
        # Initialize AI models status
        self.models_loaded = self.check_redis_ai_models()
        
        logger.info("🤖 Log Analysis Chat Bot initialized with RedisVL integration")
    
    def init_redisvl(self):
        """Initialize RedisVL search index for vector similarity search"""
        if not REDISVL_AVAILABLE:
            logger.warning("RedisVL not available - vector similarity search disabled")
            return
        
        try:
            # Use raw Redis client for RedisVL (no decode_responses)
            redis_raw = redis.Redis(host='redis-stack', port=6379, decode_responses=False)
            
            # Try to check if index exists first
            try:
                # Check if index exists using FT.INFO
                redis_raw.execute_command('FT.INFO', self.redis_index_name)
                logger.info(f"✅ RedisVL index '{self.redis_index_name}' exists with documents")
                
                # Index exists - we'll use direct Redis commands instead of SearchIndex
                # because SearchIndex requires schema definition
                self.search_index = redis_raw
                self.search_index_available = True
                
            except redis.ResponseError as e:
                if "Unknown index name" in str(e):
                    logger.warning(f"⚠️ RedisVL index '{self.redis_index_name}' does not exist yet")
                    self.search_index = None
                    self.search_index_available = False
                else:
                    raise e
            
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to RedisVL index: {e}")
            self.search_index = None
            self.search_index_available = False
    
    def check_redis_ai_models(self) -> bool:
        """Check if Redis AI models are loaded"""
        try:
            # Try to check if models are loaded - Redis AI might have different commands
            # Try different approaches for Redis AI availability
            try:
                # Method 1: Try AI.MODELLIST
                models = self.redis_ai.execute_command('AI.MODELLIST')
                logger.info("✅ Redis AI available - found models")
                return True
            except:
                try:
                    # Method 2: Try AI._LIST
                    models = self.redis_ai.execute_command('AI._LIST')
                    logger.info("✅ Redis AI available")
                    return True
                except:
                    # Method 3: Just try to connect to Redis AI port
                    self.redis_ai.ping()
                    logger.info("✅ Redis AI server available")
                    return True
        except Exception as e:
            logger.warning(f"⚠️ Redis AI not available: {e}")
            return False
    
    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly statistics from database"""
        try:
            if not Path(self.db_path).exists():
                logger.warning(f"Database not found at {self.db_path}")
                return {
                    'total_anomalies': 0,
                    'total_logs': 0,
                    'recent_anomalies': 0,
                    'anomaly_rate': 0.0,
                    'database_status': 'not_found'
                }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='anomaly_detections'")
            if not cursor.fetchone():
                conn.close()
                return {
                    'total_anomalies': 0,
                    'total_logs': 0,
                    'recent_anomalies': 0,
                    'anomaly_rate': 0.0,
                    'database_status': 'no_tables'
                }
            
            # Total anomalies (predicted_label = 1)
            cursor.execute("SELECT COUNT(*) FROM anomaly_detections WHERE predicted_label = 1")
            total_anomalies = cursor.fetchone()[0]
            
            # Total logs
            cursor.execute("SELECT COUNT(*) FROM anomaly_detections")
            total_logs = cursor.fetchone()[0]
            
            # Recent anomalies (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            cursor.execute(
                "SELECT COUNT(*) FROM anomaly_detections WHERE predicted_label = 1 AND timestamp > ?",
                (one_hour_ago.isoformat(),)
            )
            recent_anomalies = cursor.fetchone()[0]
            
            # Anomaly rate
            anomaly_rate = (total_anomalies / total_logs * 100) if total_logs > 0 else 0
            
            conn.close()
            
            return {
                'total_anomalies': total_anomalies,
                'total_logs': total_logs,
                'recent_anomalies': recent_anomalies,
                'anomaly_rate': round(anomaly_rate, 2),
                'database_status': 'available'
            }
            
        except Exception as e:
            logger.error(f"Error getting anomaly stats: {e}")
            return {
                'total_anomalies': 0,
                'total_logs': 0,
                'recent_anomalies': 0,
                'anomaly_rate': 0.0,
                'database_status': 'error',
                'error': str(e)
            }
    
    def get_latest_anomalies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get latest anomalies"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, text, anomaly_score, confidence 
                FROM anomaly_detections 
                WHERE predicted_label = 1 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            anomalies = []
            for row in cursor.fetchall():
                anomalies.append({
                    'timestamp': row[0],
                    'log_text': row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                    'anomaly_score': round(row[2], 3),
                    'confidence': round(row[3], 3)
                })
            
            conn.close()
            return anomalies
            
        except Exception as e:
            logger.error(f"Error getting latest anomalies: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Try to get from Redis first
            metrics = self.redis_stack.hgetall("performance_metrics")
            if metrics:
                return {
                    'avg_processing_time': float(metrics.get('avg_processing_time_ms', 0)),
                    'total_predictions': int(metrics.get('total_predictions', 0)),
                    'cache_hit_rate': float(metrics.get('cache_hit_rate', 0)),
                    'accuracy': float(metrics.get('accuracy', 0))
                }
            
            # Fallback to database calculation
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT AVG(processing_time_ms) FROM anomaly_detections WHERE processing_time_ms > 0")
            avg_time = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM anomaly_detections")
            total_predictions = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'avg_processing_time': round(avg_time, 2),
                'total_predictions': total_predictions,
                'cache_hit_rate': 0.0,  # Not available from DB
                'accuracy': 0.0  # Would need ground truth
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using the embedding service"""
        try:
            response = requests.post(
                f"{self.embedding_service_url}/embed",
                json={"texts": [text]},
                timeout=10
            )
            if response.status_code == 200:
                embeddings = response.json().get("embeddings", [])
                return embeddings[0] if embeddings else None
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
        return None
    
    def vector_similarity_search(self, query_text: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search using direct Redis FT.SEARCH"""
        try:
            logger.info(f"Vector search query: '{query_text}'")
            
            # Get embedding for query text
            query_embedding = self.get_embedding(query_text)
            if not query_embedding:
                logger.error("Could not get embedding for query")
                return []
            
            logger.info(f"Got embedding with {len(query_embedding)} dimensions")
            
            # Use direct Redis FT.SEARCH (most reliable method)
            try:
                redis_raw = redis.Redis(host='redis-stack', port=6379, decode_responses=False)
                redis_raw.ping()
                logger.info("Redis connection successful")
                
                # Convert embedding to bytes for Redis
                embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
                
                # Perform KNN search using FT.SEARCH
                search_result = redis_raw.execute_command(
                    'FT.SEARCH', self.redis_index_name,
                    f'*=>[KNN {num_results} @embedding $vec AS score]',
                    'PARAMS', '2', 'vec', embedding_bytes,
                    'SORTBY', 'score',
                    'RETURN', '4', 'text', 'timestamp', 'label', 'score',
                    'DIALECT', '2'
                )
                
                # Parse results
                formatted_results = []
                if search_result and len(search_result) > 1:
                    num_results_found = search_result[0]
                    logger.info(f"Found {num_results_found} results from vector search")
                    
                    for i in range(1, len(search_result), 2):
                        if i + 1 < len(search_result):
                            doc_id = search_result[i].decode('utf-8') if isinstance(search_result[i], bytes) else search_result[i]
                            fields = search_result[i + 1]
                            
                            # Parse fields
                            field_dict = {}
                            for j in range(0, len(fields), 2):
                                if j + 1 < len(fields):
                                    key = fields[j].decode('utf-8') if isinstance(fields[j], bytes) else fields[j]
                                    value = fields[j + 1]
                                    if isinstance(value, bytes):
                                        try:
                                            value = value.decode('utf-8')
                                        except:
                                            value = str(value)
                                    field_dict[key] = value
                            
                            # Calculate similarity score (1 - distance for COSINE)
                            distance = float(field_dict.get('score', 1.0))
                            similarity = 1.0 - distance
                            
                            formatted_results.append({
                                'log_text': field_dict.get('text', ''),
                                'timestamp': field_dict.get('timestamp', ''),
                                'anomaly_score': 0.0,
                                'predicted_label': int(field_dict.get('label', 0)),
                                'similarity_score': similarity
                            })
                
                logger.info(f"Vector search returned {len(formatted_results)} results")
                return formatted_results
                
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                import traceback
                traceback.print_exc()
                return []
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []
    
    def find_anomaly_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar anomaly patterns using vector search"""
        try:
            # Get recent anomalies from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT text, anomaly_score 
                FROM anomaly_detections 
                WHERE predicted_label = 1 
                ORDER BY timestamp DESC 
                LIMIT 3
            """)
            
            recent_anomalies = cursor.fetchall()
            conn.close()
            
            if not recent_anomalies:
                return []
            
            # Use most recent anomaly for pattern search
            recent_anomaly_text = recent_anomalies[0][0]
            
            # Find similar patterns
            similar_logs = self.vector_similarity_search(recent_anomaly_text, limit)
            
            return similar_logs
            
        except Exception as e:
            logger.error(f"Error finding anomaly patterns: {e}")
            return []
    
    def analyze_message(self, message: str) -> ChatResponse:
        """Analyze user message and generate appropriate response"""
        message_lower = message.lower()
        
        # Determine intent
        intent = self.classify_intent(message)
        
        # Generate response based on intent
        if intent == 'anomaly_count':
            return self.handle_anomaly_count_query(message)
        elif intent == 'latest_anomalies':
            return self.handle_latest_anomalies_query(message)
        elif intent == 'performance':
            return self.handle_performance_query(message)
        elif intent == 'log_summary':
            return self.handle_log_summary_query(message)
        elif intent == 'help':
            return self.handle_help_query(message)
        elif intent == 'system_health':
            return self.handle_system_health_query(message)
        elif intent == 'similar_logs' or intent == 'query_logs':
            return self.handle_similar_logs_query(message)
        elif intent == 'vector_search':
            return self.handle_vector_search_query(message)
        else:
            return self.handle_general_query(message)
    
    def classify_intent(self, message: str) -> str:
        """Classify user intent based on message patterns"""
        for intent, pattern in self.chat_patterns.items():
            if re.search(pattern, message):
                return intent
        return 'general'
    
    def handle_anomaly_count_query(self, message: str) -> ChatResponse:
        """Handle anomaly count queries"""
        stats = self.get_anomaly_stats()
        
        if stats.get('database_status') == 'not_found':
            response = f"📊 **Database Status: Not Found**\n\n"
            response += f"The anomaly detection database hasn't been created yet. This means:\n\n"
            response += f"• No logs have been processed through the system\n"
            response += f"• The Kafka pipeline might not be running\n"
            response += f"• The anomaly detection service needs to be started\n\n"
            response += f"💡 **To get started:**\n"
            response += f"1. Start the anomaly detection service\n"
            response += f"2. Send some log data through Kafka\n"
            response += f"3. Return here to see anomaly statistics!"
            
            suggestions = [
                "Check system health",
                "What can you help me with?",
                "Show performance metrics"
            ]
        elif stats.get('database_status') == 'no_tables':
            response = f"📊 **Database Status: Empty**\n\n"
            response += f"The database exists but contains no log data yet.\n\n"
            response += f"This typically means the system is ready but waiting for log data."
            
            suggestions = [
                "Check system health",
                "What can you help me with?"
            ]
        else:
            response = f"📊 **Anomaly Statistics:**\n\n"
            response += f"• **Total Anomalies:** {stats.get('total_anomalies', 0):,}\n"
            response += f"• **Total Logs Processed:** {stats.get('total_logs', 0):,}\n"
            response += f"• **Anomaly Rate:** {stats.get('anomaly_rate', 0)}%\n"
            response += f"• **Recent Anomalies (1h):** {stats.get('recent_anomalies', 0)}\n\n"
            
            if stats.get('recent_anomalies', 0) > 10:
                response += "⚠️ **Alert:** High number of recent anomalies detected!"
            elif stats.get('recent_anomalies', 0) > 0:
                response += "ℹ️ Some recent anomalous activity detected."
            else:
                response += "✅ No recent anomalies - system looks healthy!"
            
            suggestions = [
                "Show me the latest anomalies",
                "What's the system performance?",
                "Give me a log summary"
            ]
        
        return ChatResponse(
            response=response,
            analysis_data=stats,
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def handle_latest_anomalies_query(self, message: str) -> ChatResponse:
        """Handle latest anomalies queries"""
        anomalies = self.get_latest_anomalies(5)
        
        if not anomalies:
            response = "✅ **No recent anomalies found!**\n\nYour system appears to be running smoothly."
            suggestions = ["Check system performance", "Get log summary"]
        else:
            response = f"🚨 **Latest {len(anomalies)} Anomalies:**\n\n"
            
            for i, anomaly in enumerate(anomalies, 1):
                response += f"**{i}.** `{anomaly['timestamp']}`\n"
                response += f"   📝 {anomaly['log_text']}\n"
                response += f"   🎯 Score: {anomaly['anomaly_score']} | Confidence: {anomaly['confidence']}\n\n"
            
            suggestions = [
                "What caused these anomalies?",
                "Show anomaly trends",
                "Check system health"
            ]
        
        return ChatResponse(
            response=response,
            analysis_data={'anomalies': anomalies},
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def handle_performance_query(self, message: str) -> ChatResponse:
        """Handle performance queries"""
        metrics = self.get_performance_metrics()
        
        response = f"⚡ **System Performance Metrics:**\n\n"
        response += f"• **Average Processing Time:** {metrics.get('avg_processing_time', 0):.1f}ms\n"
        response += f"• **Total Predictions:** {metrics.get('total_predictions', 0):,}\n"
        response += f"• **Cache Hit Rate:** {metrics.get('cache_hit_rate', 0):.1%}\n"
        
        if self.models_loaded:
            response += f"• **Redis AI Status:** ✅ Active\n"
        else:
            response += f"• **Redis AI Status:** ⚠️ Not Available\n"
        
        # Performance assessment
        avg_time = metrics.get('avg_processing_time', 0)
        if avg_time < 50:
            response += f"\n🚀 **Excellent performance!** Very fast processing times."
        elif avg_time < 100:
            response += f"\n✅ **Good performance.** Processing times are acceptable."
        else:
            response += f"\n⚠️ **Performance warning.** Processing times are higher than optimal."
        
        suggestions = [
            "Show system health",
            "Display anomaly trends",
            "Check latest errors"
        ]
        
        return ChatResponse(
            response=response,
            analysis_data=metrics,
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def handle_log_summary_query(self, message: str) -> ChatResponse:
        """Handle log summary queries"""
        stats = self.get_anomaly_stats()
        metrics = self.get_performance_metrics()
        
        response = f"📋 **System Log Summary:**\n\n"
        response += f"**📊 Volume:**\n"
        response += f"• Total logs processed: {stats.get('total_logs', 0):,}\n"
        response += f"• Anomalies detected: {stats.get('total_anomalies', 0):,}\n"
        response += f"• Anomaly rate: {stats.get('anomaly_rate', 0)}%\n\n"
        
        response += f"**⚡ Performance:**\n"
        response += f"• Avg processing time: {metrics.get('avg_processing_time', 0):.1f}ms\n"
        response += f"• Total predictions: {metrics.get('total_predictions', 0):,}\n\n"
        
        response += f"**🕐 Recent Activity:**\n"
        response += f"• Anomalies in last hour: {stats.get('recent_anomalies', 0)}\n"
        
        # Overall health assessment
        recent_anomalies = stats.get('recent_anomalies', 0)
        if recent_anomalies == 0:
            response += f"\n✅ **System Status:** Healthy - No recent issues detected"
        elif recent_anomalies < 5:
            response += f"\n🟡 **System Status:** Monitor - Some anomalous activity"
        else:
            response += f"\n🔴 **System Status:** Alert - High anomalous activity"
        
        suggestions = [
            "Show latest anomalies",
            "Check performance details",
            "What can you help me with?"
        ]
        
        return ChatResponse(
            response=response,
            analysis_data={**stats, **metrics},
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def handle_system_health_query(self, message: str) -> ChatResponse:
        """Handle system health queries"""
        # Check Redis connections
        redis_stack_ok = False
        redis_ai_ok = False
        
        try:
            self.redis_stack.ping()
            redis_stack_ok = True
        except:
            pass
        
        try:
            self.redis_ai.ping()
            redis_ai_ok = True
        except:
            pass
        
        # Check database
        db_ok = Path(self.db_path).exists()
        
        response = f"🏥 **System Health Check:**\n\n"
        response += f"**🔧 Infrastructure:**\n"
        response += f"• Redis Stack: {'✅ Online' if redis_stack_ok else '❌ Offline'}\n"
        response += f"• Redis AI: {'✅ Online' if redis_ai_ok else '❌ Offline'}\n"
        response += f"• Database: {'✅ Available' if db_ok else '❌ Missing'}\n"
        response += f"• AI Models: {'✅ Loaded' if self.models_loaded else '⚠️ Not Available'}\n\n"
        
        # Calculate overall health
        health_score = sum([redis_stack_ok, redis_ai_ok, db_ok]) / 3 * 100
        
        if health_score == 100:
            response += f"🟢 **Overall Health:** Excellent ({health_score:.0f}%)"
        elif health_score >= 66:
            response += f"🟡 **Overall Health:** Good ({health_score:.0f}%)"
        else:
            response += f"🔴 **Overall Health:** Issues Detected ({health_score:.0f}%)"
        
        suggestions = [
            "Check performance metrics",
            "Show recent anomalies",
            "Get log summary"
        ]
        
        return ChatResponse(
            response=response,
            analysis_data={
                'redis_stack': redis_stack_ok,
                'redis_ai': redis_ai_ok,
                'database': db_ok,
                'models_loaded': self.models_loaded,
                'health_score': health_score
            },
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def handle_help_query(self, message: str) -> ChatResponse:
        """Handle help queries"""
        response = f"🤖 **Log Analysis Assistant - Help**\n\n"
        response += f"I can help you analyze your logs and anomalies! Here's what you can ask:\n\n"
        
        response += f"**📊 Anomaly Analysis:**\n"
        response += f"• \"How many anomalies were detected?\"\n"
        response += f"• \"Show me the latest anomalies\"\n"
        response += f"• \"What types of anomalies occurred?\"\n\n"
        
        response += f"**⚡ Performance & Health:**\n"
        response += f"• \"What's the system performance?\"\n"
        response += f"• \"Check system health\"\n"
        response += f"• \"Show processing times\"\n\n"
        
        response += f"**📋 Reports & Summaries:**\n"
        response += f"• \"Give me a log summary\"\n"
        response += f"• \"Show recent trends\"\n"
        response += f"• \"System status overview\"\n\n"
        
        response += f"💡 **Tips:**\n"
        response += f"• Ask in natural language - I understand context!\n"
        response += f"• Use the suggested questions below for quick access\n"
        response += f"• I can analyze patterns and provide insights\n"
        
        if self.models_loaded:
            response += f"• 🚀 Redis AI is active for enhanced analysis!"
        
        suggestions = [
            "How many anomalies were detected?",
            "Show me the latest anomalies",
            "What's the system performance?",
            "Give me a log summary"
        ]
        
        return ChatResponse(
            response=response,
            analysis_data=None,
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def handle_general_query(self, message: str) -> ChatResponse:
        """Handle general/unrecognized queries"""
        response = f"🤔 I'm not sure I understand that specific request.\n\n"
        response += f"I'm specialized in log analysis and anomaly detection. "
        response += f"I can help you with:\n\n"
        response += f"• Anomaly statistics and reports\n"
        response += f"• System performance metrics\n"
        response += f"• Recent log activity\n"
        response += f"• Health status checks\n\n"
        response += f"Try asking me something like \"How many anomalies were detected?\" or \"Show system health\""
        
        suggestions = [
            "How many anomalies were detected?",
            "What's the system performance?",
            "Show me the latest anomalies",
            "What can you help me with?"
        ]
        
        return ChatResponse(
            response=response,
            analysis_data=None,
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def handle_similar_logs_query(self, message: str) -> ChatResponse:
        """Handle queries for similar logs using vector search"""
        # Extract search term from message
        search_text = message
        
        # Try to extract specific log text to search for
        if "like" in message.lower():
            parts = message.lower().split("like")
            if len(parts) > 1:
                search_text = parts[1].strip().strip('"\'')
        elif "similar to" in message.lower():
            parts = message.lower().split("similar to")
            if len(parts) > 1:
                search_text = parts[1].strip().strip('"\'')
        elif "find" in message.lower() and ("error" in message.lower() or "block" in message.lower() or "receiving" in message.lower()):
            # Extract key terms for search
            search_text = message
        
        # Perform vector similarity search
        similar_logs = self.vector_similarity_search(search_text, 5)
        
        if not similar_logs:
            response = "🔍 **No similar logs found**\n\n"
            response += "I couldn't find logs similar to your query. This could be because:\n"
            response += "• The embedding service is not available\n"
            response += "• No similar patterns exist in the log database\n"
            response += "• The Redis VL index needs to be populated with logs\n\n"
            response += "💡 **Tip:** Make sure logs are being processed through the Spark job and stored in Redis.\n\n"
            response += "Try asking about recent anomalies or system status instead."
            
            suggestions = [
                "Show me the latest anomalies",
                "What's the system performance?",
                "Check system health"
            ]
        else:
            response = f"🔍 **Found {len(similar_logs)} Similar Logs:**\n\n"
            
            anomaly_count = sum(1 for log in similar_logs if log['predicted_label'] == 1)
            response += f"📊 **Analysis:** {anomaly_count} anomalies, {len(similar_logs) - anomaly_count} normal logs\n\n"
            
            for i, log in enumerate(similar_logs, 1):
                is_anomaly = "🚨" if log['predicted_label'] == 1 else "✅"
                response += f"**{i}.** {is_anomaly} `{log['timestamp']}`\n"
                response += f"   📝 {log['log_text'][:100]}{'...' if len(log['log_text']) > 100 else ''}\n"
                response += f"   🎯 Similarity: {log['similarity_score']:.3f}\n\n"
            
            response += "✨ **Powered by Redis Vector Search**"
            
            suggestions = [
                "Find more patterns like these",
                "Show anomaly trends",
                "Check system performance"
            ]
        
        return ChatResponse(
            response=response,
            analysis_data={'similar_logs': similar_logs, 'query': search_text},
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
    
    def handle_vector_search_query(self, message: str) -> ChatResponse:
        """Handle specific vector/semantic search queries"""
        # Find anomaly patterns using vector similarity
        pattern_results = self.find_anomaly_patterns(8)
        
        response = f"🧠 **Semantic Log Analysis:**\n\n"
        
        if pattern_results:
            response += f"Found {len(pattern_results)} logs with similar patterns to recent anomalies:\n\n"
            
            anomaly_count = sum(1 for log in pattern_results if log['predicted_label'] == 1)
            normal_count = len(pattern_results) - anomaly_count
            
            response += f"📊 **Pattern Distribution:**\n"
            response += f"• Anomalous patterns: {anomaly_count}\n"
            response += f"• Normal patterns: {normal_count}\n"
            response += f"• Average similarity: {np.mean([log['similarity_score'] for log in pattern_results]):.3f}\n\n"
            
            response += f"🎯 **Top Similar Logs:**\n"
            for i, log in enumerate(pattern_results[:3], 1):
                status = "🚨 Anomaly" if log['predicted_label'] == 1 else "✅ Normal"
                response += f"{i}. {status} (Score: {log['anomaly_score']:.3f})\n"
                response += f"   {log['log_text'][:80]}...\n"
            
            if REDISVL_AVAILABLE:
                response += f"\n🚀 **Vector Search:** Using RedisVL embeddings for semantic analysis"
        else:
            response += "No patterns found. This could indicate:\n"
            response += "• Very few logs in the system\n"
            response += "• Embedding service unavailable\n"
            response += "• RedisVL index not populated\n\n"
            response += "💡 Try sending some logs through the Kafka pipeline first."
        
        suggestions = [
            "Show me the latest anomalies",
            "Find logs similar to an error message",
            "What's the system health?"
        ]
        
        return ChatResponse(
            response=response,
            analysis_data={'pattern_results': pattern_results},
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )

# FastAPI app
app = FastAPI(title="Redis AI Log Analysis Chat", version="1.0.0")

# Initialize chat bot
chat_bot = LogAnalysisChatBot()

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Redis AI Log Analysis Chat</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', system-ui, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh; display: flex; align-items: center; justify-content: center;
            }
            .chat-container { 
                width: 90%; max-width: 800px; height: 80vh; 
                background: white; border-radius: 20px; 
                box-shadow: 0 20px 60px rgba(0,0,0,0.15);
                display: flex; flex-direction: column; overflow: hidden;
            }
            .chat-header { 
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4); 
                color: white; padding: 20px; text-align: center;
                border-radius: 20px 20px 0 0;
            }
            .chat-header h1 { font-size: 1.5em; margin-bottom: 5px; }
            .chat-header p { opacity: 0.9; font-size: 0.9em; }
            .chat-messages { 
                flex: 1; padding: 20px; overflow-y: auto; 
                background: #f8f9fa;
            }
            .message { margin-bottom: 15px; }
            .user-message { 
                background: #4ECDC4; color: white; 
                padding: 12px 18px; border-radius: 18px 18px 5px 18px;
                margin-left: 20%; text-align: right;
            }
            .bot-message { 
                background: white; border: 1px solid #e9ecef;
                padding: 15px 20px; border-radius: 18px 18px 18px 5px;
                margin-right: 10%; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .suggestions { 
                margin-top: 15px; display: flex; flex-wrap: wrap; gap: 8px;
            }
            .suggestion-btn { 
                background: #f1f3f4; border: none; padding: 8px 12px;
                border-radius: 20px; cursor: pointer; font-size: 0.85em;
                transition: all 0.2s; color: #5f6368;
            }
            .suggestion-btn:hover { 
                background: #4ECDC4; color: white; transform: translateY(-1px);
            }
            .chat-input-container { 
                padding: 20px; background: white; 
                border-top: 1px solid #e9ecef;
            }
            .input-group { display: flex; gap: 10px; }
            .chat-input { 
                flex: 1; padding: 15px; border: 2px solid #e9ecef;
                border-radius: 25px; font-size: 1em; outline: none;
                transition: border-color 0.2s;
            }
            .chat-input:focus { border-color: #4ECDC4; }
            .send-btn { 
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                color: white; border: none; padding: 15px 25px;
                border-radius: 25px; cursor: pointer; font-weight: 600;
                transition: transform 0.2s;
            }
            .send-btn:hover { transform: scale(1.05); }
            .send-btn:disabled { 
                background: #ccc; cursor: not-allowed; transform: none;
            }
            .typing { 
                opacity: 0.7; font-style: italic; 
                color: #666; font-size: 0.9em;
            }
            .status-indicator {
                display: inline-block; width: 10px; height: 10px;
                border-radius: 50%; margin-right: 8px;
                background: #4ECDC4; animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1><span class="status-indicator"></span>Redis AI Log Analysis Assistant</h1>
                <p>Intelligent chat interface for anomaly detection and log analysis</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message">
                    <div class="bot-message">
                        <strong>🤖 Welcome to Redis AI Log Analysis Chat!</strong><br><br>
                        I'm your intelligent assistant for analyzing logs and detecting anomalies. 
                        I can help you with:<br><br>
                        • 📊 Anomaly statistics and reports<br>
                        • ⚡ System performance metrics<br>
                        • 🔍 Recent log activity analysis<br>
                        • 🏥 Health status monitoring<br><br>
                        Ask me anything about your logs! Try: "How many anomalies were detected?"
                        
                        <div class="suggestions">
                            <button class="suggestion-btn" onclick="sendSuggestion('How many anomalies were detected?')">Anomaly Count</button>
                            <button class="suggestion-btn" onclick="sendSuggestion('Show me the latest anomalies')">Latest Anomalies</button>
                            <button class="suggestion-btn" onclick="sendSuggestion('What\\'s the system performance?')">Performance</button>
                            <button class="suggestion-btn" onclick="sendSuggestion('Give me a log summary')">Log Summary</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <div class="input-group">
                    <input type="text" class="chat-input" id="messageInput" 
                           placeholder="Ask me about your logs and anomalies..." 
                           onkeypress="handleKeyPress(event)">
                    <button class="send-btn" onclick="sendMessage()" id="sendBtn">Send</button>
                </div>
            </div>
        </div>

        <script>
            let ws = null;
            
            function connectWebSocket() {
                ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onopen = function(event) {
                    console.log('Connected to chat bot');
                };
                
                ws.onmessage = function(event) {
                    const response = JSON.parse(event.data);
                    addBotMessage(response);
                };
                
                ws.onclose = function(event) {
                    console.log('Disconnected from chat bot');
                    setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
                };
            }
            
            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (message) {
                    addUserMessage(message);
                    
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({message: message}));
                    } else {
                        // Fallback to HTTP API
                        sendHttpMessage(message);
                    }
                    
                    input.value = '';
                    showTyping();
                }
            }
            
            async function sendHttpMessage(message) {
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    const data = await response.json();
                    removeTyping();
                    addBotMessage(data);
                } catch (error) {
                    removeTyping();
                    addBotMessage({
                        response: "Sorry, I'm having trouble connecting. Please try again.",
                        suggestions: []
                    });
                }
            }
            
            function sendSuggestion(text) {
                document.getElementById('messageInput').value = text;
                sendMessage();
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            function addUserMessage(message) {
                const messagesDiv = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';
                messageDiv.innerHTML = `<div class="user-message">${escapeHtml(message)}</div>`;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function addBotMessage(response) {
                removeTyping();
                const messagesDiv = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';
                
                let suggestionsHtml = '';
                if (response.suggestions && response.suggestions.length > 0) {
                    suggestionsHtml = '<div class="suggestions">';
                    response.suggestions.forEach(suggestion => {
                        suggestionsHtml += `<button class="suggestion-btn" onclick="sendSuggestion('${escapeHtml(suggestion)}')">${escapeHtml(suggestion)}</button>`;
                    });
                    suggestionsHtml += '</div>';
                }
                
                messageDiv.innerHTML = `
                    <div class="bot-message">
                        ${formatBotResponse(response.response)}
                        ${suggestionsHtml}
                    </div>
                `;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function showTyping() {
                const messagesDiv = document.getElementById('chatMessages');
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message typing';
                typingDiv.id = 'typing-indicator';
                typingDiv.innerHTML = '<div class="bot-message typing">🤖 Analyzing your request...</div>';
                messagesDiv.appendChild(typingDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function removeTyping() {
                const typing = document.getElementById('typing-indicator');
                if (typing) {
                    typing.remove();
                }
            }
            
            function formatBotResponse(text) {
                // Convert markdown-style formatting to HTML
                return text
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/\n/g, '<br>');
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            // Initialize
            connectWebSocket();
            
            // Add event listeners when DOM is ready
            document.getElementById('sendBtn').addEventListener('click', sendMessage);
            document.getElementById('messageInput').addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """HTTP endpoint for chat messages"""
    try:
        response = chat_bot.analyze_message(message.message)
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            response = chat_bot.analyze_message(message_data['message'])
            
            # Send response back
            await websocket.send_text(response.json())
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Redis AI Log Analysis Chat",
        "redis_ai_available": chat_bot.models_loaded,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
