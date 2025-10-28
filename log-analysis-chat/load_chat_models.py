#!/usr/bin/env python3
"""
Simple Redis AI Model Loader for Chat Interface
Loads a basic text classification model for chat intelligence
"""

import redis
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleChatModelLoader:
    """Load simple models into Redis AI for chat intelligence"""
    
    def __init__(self, redis_host='localhost', redis_port=6380):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        logger.info(f"🔗 Connected to Redis AI at {redis_host}:{redis_port}")
    
    def test_connection(self):
        """Test Redis AI connection"""
        try:
            self.redis_client.ping()
            logger.info("✅ Redis AI connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Redis AI connection failed: {e}")
            return False
    
    def create_simple_anomaly_scorer(self):
        """Create a simple anomaly scoring model using TorchScript"""
        
        # Simple TorchScript model for demo purposes
        torch_script = """
def anomaly_score(features):
    # Simple heuristic-based anomaly scoring
    # In a real implementation, this would be a trained model
    
    # Calculate basic statistics
    mean_val = torch.mean(features)
    std_val = torch.std(features)
    
    # Simple anomaly score based on deviation from mean
    anomaly_score = torch.abs(mean_val) / (std_val + 0.001)
    
    # Normalize to 0-1 range
    normalized_score = torch.sigmoid(anomaly_score)
    
    return normalized_score
"""
        
        try:
            # Store the script in Redis AI
            self.redis_client.execute_command(
                'AI.SCRIPTSTORE', 'simple_anomaly_scorer', 'CPU', 'SOURCE', torch_script
            )
            logger.info("✅ Simple anomaly scorer script loaded")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load anomaly scorer: {e}")
            return False
    
    def create_intent_classifier(self):
        """Create a simple intent classification model"""
        
        # Simple intent classification script
        intent_script = """
def classify_intent(text_features):
    # Simple intent classification based on feature patterns
    # This is a demo - real implementation would use trained embeddings
    
    # Calculate feature statistics
    feature_sum = torch.sum(text_features)
    feature_max = torch.max(text_features)
    feature_min = torch.min(text_features)
    
    # Simple decision rules (demo purposes)
    if feature_sum > 100:
        intent = torch.tensor([1.0, 0.0, 0.0])  # anomaly_query
    elif feature_max > 50:
        intent = torch.tensor([0.0, 1.0, 0.0])  # performance_query  
    else:
        intent = torch.tensor([0.0, 0.0, 1.0])  # general_query
    
    return intent
"""
        
        try:
            self.redis_client.execute_command(
                'AI.SCRIPTSTORE', 'intent_classifier', 'CPU', 'SOURCE', intent_script
            )
            logger.info("✅ Intent classifier script loaded")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load intent classifier: {e}")
            return False
    
    def test_models(self):
        """Test the loaded models"""
        logger.info("🧪 Testing loaded models...")
        
        try:
            # Test anomaly scorer
            test_features = np.random.randn(10).astype(np.float32)
            
            # Store test tensor
            self.redis_client.execute_command(
                'AI.TENSORSET', 'test_features', 'FLOAT', '10', *test_features
            )
            
            # Run anomaly scorer
            self.redis_client.execute_command(
                'AI.SCRIPTEXECUTE', 'simple_anomaly_scorer', 'anomaly_score',
                'INPUTS', 'test_features', 'OUTPUTS', 'anomaly_result'
            )
            
            # Get result
            result = self.redis_client.execute_command('AI.TENSORGET', 'anomaly_result', 'VALUES')
            logger.info(f"  ✅ Anomaly scorer test: {result[0]:.4f}")
            
            # Test intent classifier
            text_features = np.random.rand(20).astype(np.float32) * 100
            
            self.redis_client.execute_command(
                'AI.TENSORSET', 'text_features', 'FLOAT', '20', *text_features
            )
            
            self.redis_client.execute_command(
                'AI.SCRIPTEXECUTE', 'intent_classifier', 'classify_intent',
                'INPUTS', 'text_features', 'OUTPUTS', 'intent_result'
            )
            
            intent_result = self.redis_client.execute_command('AI.TENSORGET', 'intent_result', 'VALUES')
            logger.info(f"  ✅ Intent classifier test: {intent_result}")
            
            # Clean up test tensors
            self.redis_client.execute_command('AI.TENSORDEL', 'test_features')
            self.redis_client.execute_command('AI.TENSORDEL', 'anomaly_result')
            self.redis_client.execute_command('AI.TENSORDEL', 'text_features')
            self.redis_client.execute_command('AI.TENSORDEL', 'intent_result')
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model testing failed: {e}")
            return False
    
    def load_all_models(self):
        """Load all chat intelligence models"""
        logger.info("🚀 Loading Redis AI Chat Models")
        logger.info("=" * 40)
        
        if not self.test_connection():
            return False
        
        success = True
        
        # Load anomaly scorer
        if not self.create_simple_anomaly_scorer():
            success = False
        
        # Load intent classifier
        if not self.create_intent_classifier():
            success = False
        
        # Test models
        if success and not self.test_models():
            success = False
        
        if success:
            logger.info("🎉 All chat models loaded successfully!")
        else:
            logger.error("❌ Some models failed to load")
        
        return success

def main():
    """Main loader execution"""
    loader = SimpleChatModelLoader()
    success = loader.load_all_models()
    
    if success:
        print("\n✅ Redis AI Chat Models Ready!")
        print("🚀 You can now start the chat interface:")
        print("   docker-compose up log-analysis-chat")
    else:
        print("\n❌ Model loading failed!")
        print("⚠️ Check Redis AI connection and try again")

if __name__ == "__main__":
    main()
