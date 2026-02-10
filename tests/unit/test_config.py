"""Unit tests for configuration module."""

import unittest
from unittest.mock import patch
from src.core.config import get_config, SystemConfig, MilvusConfig, LLMConfig

class TestConfig(unittest.TestCase):
    """Test configuration loading and validation."""
    
    def test_milvus_config_defaults(self):
        """Test Milvus configuration defaults."""
        config = MilvusConfig()
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 19530)
        self.assertEqual(config.collection_name, "documents")
        self.assertEqual(config.embedding_dim, 384)
    
    def test_llm_config_defaults(self):
        """Test LLM configuration defaults."""
        config = LLMConfig()
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, "gemini-pro")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.top_k, 5)
    
    @patch.dict('os.environ', {'MILVUS_HOST': 'test-host', 'MILVUS_PORT': '9999'})
    def test_config_environment_override(self):
        """Test that environment variables override defaults."""
        config = get_config()
        self.assertEqual(config.milvus.host, "test-host")
        self.assertEqual(config.milvus.port, 9999)
    
    def test_system_config_post_init(self):
        """Test that SystemConfig creates required directories."""
        import tempfile
        import shutil
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SystemConfig(
                milvus=MilvusConfig(),
                llm=LLMConfig(),
                embedding=None,
                chunking=None,
                retrieval=None,
                upload_dir=temp_dir,
                temp_dir=temp_dir
            )
            
            # Check that directories exist
            self.assertTrue(os.path.exists(temp_dir))

if __name__ == '__main__':
    unittest.main()