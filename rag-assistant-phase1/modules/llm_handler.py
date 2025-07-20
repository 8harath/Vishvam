"""
Open Source LLM Integration for RAG Assistant Phase 1

This module provides functionality to integrate various open source LLMs including:
- Hugging Face Transformers models
- Ollama local LLMs
- Memory-efficient loading and inference
"""

import logging
import json
import time
import warnings
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available - Ollama integration will be disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate a response to the given prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available and ready to use."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass


class HuggingFaceLLM(BaseLLM):
    """
    Hugging Face Transformers-based LLM implementation.
    
    Supports various models with memory optimization options.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/DialoGPT-medium",
        device: Optional[str] = None,
        use_quantization: bool = True,
        max_memory_gb: float = 8.0
    ):
        """
        Initialize the Hugging Face LLM.
        
        Args:
            model_name (str): HuggingFace model name/path
            device (str, optional): Device to load model on
            use_quantization (bool): Use 8-bit quantization to reduce memory
            max_memory_gb (float): Maximum memory to use in GB
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.max_memory_gb = max_memory_gb
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        logger.info(f"Initializing HuggingFace LLM: {model_name} on {device}")
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure model loading parameters
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            # Use quantization if enabled and CUDA available
            if self.use_quantization and self.device == "cuda":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("Using 8-bit quantization")
                except Exception as e:
                    logger.warning(f"Quantization setup failed, using standard loading: {e}")
            
            logger.info("Loading model... (this may take a few minutes)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using quantization
            if not (self.use_quantization and self.device == "cuda"):
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully! Memory usage: ~{self._estimate_memory_usage():.1f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            self.is_loaded = False
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate GPU/RAM memory usage in GB."""
        if self.model is None:
            return 0.0
        
        try:
            param_size = sum(p.numel() for p in self.model.parameters())
            # Rough estimation: 4 bytes per parameter for float32, 2 for float16
            bytes_per_param = 2 if self.device == "cuda" else 4
            memory_bytes = param_size * bytes_per_param
            return memory_bytes / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 512, 
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum tokens to generate
            temperature (float): Sampling temperature (0.1-1.0)
            do_sample (bool): Whether to use sampling
            top_p (float): Top-p sampling threshold
        
        Returns:
            str: Generated response
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Check initialization.")
        
        if not prompt.strip():
            return "Please provide a valid question or prompt."
        
        try:
            # Generate response
            logger.info(f"Generating response for prompt: {prompt[:50]}...")
            start_time = time.time()
            
            # Configure generation parameters
            generation_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,  # Only return generated part
                "num_return_sequences": 1
            }
            
            outputs = self.pipeline(prompt, **generation_kwargs)
            response = outputs[0]["generated_text"].strip()
            
            # Clean up the response
            response = self._clean_response(response, prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def _clean_response(self, response: str, original_prompt: str) -> str:
        """Clean up the generated response."""
        # Remove the original prompt if it appears in the response
        if response.startswith(original_prompt):
            response = response[len(original_prompt):].strip()
        
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("[PAD]", "")
        
        # Truncate at sentence boundaries if too long
        if len(response) > 1000:
            sentences = response.split('. ')
            truncated = '. '.join(sentences[:3])
            if not truncated.endswith('.'):
                truncated += '.'
            response = truncated
        
        return response.strip()
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.is_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": self.use_quantization,
            "estimated_memory_gb": round(self._estimate_memory_usage(), 2),
            "vocab_size": len(self.tokenizer) if self.tokenizer else "Unknown",
            "max_position_embeddings": getattr(self.model.config, "max_position_embeddings", "Unknown")
        }


class OllamaLLM(BaseLLM):
    """
    Ollama-based LLM implementation for local models.
    
    Requires Ollama to be running locally.
    """
    
    def __init__(
        self,
        model_name: str = "llama2:7b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama LLM client.
        
        Args:
            model_name (str): Ollama model name
            base_url (str): Ollama server URL
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library required for Ollama integration")
        
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        logger.info(f"Initializing Ollama LLM: {model_name} at {base_url}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            # Try to get model list
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model_name in model_names:
                    logger.info(f"Successfully connected to Ollama. Model '{self.model_name}' is available.")
                    return True
                else:
                    logger.warning(f"Model '{self.model_name}' not found. Available models: {model_names}")
                    return False
            else:
                logger.error(f"Failed to connect to Ollama server: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama server at {self.base_url}: {str(e)}")
            return False
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 512, 
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using Ollama API.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum tokens to generate
            temperature (float): Sampling temperature
        
        Returns:
            str: Generated response
        """
        if not prompt.strip():
            return "Please provide a valid question or prompt."
        
        try:
            logger.info(f"Generating response via Ollama: {prompt[:50]}...")
            start_time = time.time()
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_length,
                }
            }
            
            # Make streaming request to Ollama
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
            
            # Parse streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            full_response += data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            generation_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.2f} seconds")
            
            return full_response.strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {str(e)}")
            return f"I couldn't connect to the Ollama server. Please make sure Ollama is running and the model '{self.model_name}' is available."
        except Exception as e:
            logger.error(f"Unexpected error in Ollama generation: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Ollama server and model are available."""
        return self._test_connection()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama model."""
        try:
            # Get model details from Ollama
            response = requests.get(f"{self.base_url}/api/show", 
                                  json={"name": self.model_name}, 
                                  timeout=5)
            
            if response.status_code == 200:
                model_info = response.json()
                return {
                    "model_name": self.model_name,
                    "base_url": self.base_url,
                    "status": "Available",
                    "model_info": model_info.get("details", {})
                }
            else:
                return {
                    "model_name": self.model_name,
                    "base_url": self.base_url,
                    "status": "Model not found"
                }
                
        except Exception as e:
            return {
                "model_name": self.model_name,
                "base_url": self.base_url,
                "status": f"Error: {str(e)}"
            }


class LLMManager:
    """
    High-level manager for LLM operations.
    
    Handles model selection, fallback, and unified interface.
    """
    
    def __init__(
        self,
        backend: str = "huggingface",
        hf_model: str = "microsoft/DialoGPT-medium",
        ollama_model: str = "llama2:7b",
        ollama_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize LLM Manager.
        
        Args:
            backend (str): Primary backend ("huggingface" or "ollama")
            hf_model (str): Hugging Face model name
            ollama_model (str): Ollama model name
            ollama_url (str): Ollama server URL
        """
        self.backend = backend.lower()
        self.primary_llm = None
        self.fallback_llm = None
        
        logger.info(f"Initializing LLM Manager with backend: {backend}")
        
        try:
            if self.backend == "huggingface":
                self.primary_llm = HuggingFaceLLM(hf_model, **kwargs)
                # Try to set up Ollama as fallback
                if REQUESTS_AVAILABLE:
                    try:
                        self.fallback_llm = OllamaLLM(ollama_model, ollama_url)
                    except Exception:
                        logger.info("Ollama fallback not available")
            
            elif self.backend == "ollama":
                self.primary_llm = OllamaLLM(ollama_model, ollama_url)
                # Try to set up HuggingFace as fallback
                try:
                    self.fallback_llm = HuggingFaceLLM(hf_model, **kwargs)
                except Exception:
                    logger.info("HuggingFace fallback not available")
            
            else:
                raise ValueError(f"Unsupported backend: {backend}")
        
        except Exception as e:
            logger.error(f"Failed to initialize primary LLM: {str(e)}")
            raise RuntimeError(f"LLM Manager initialization failed: {str(e)}")
    
    def query_llm(
        self, 
        prompt: str, 
        max_length: int = 512, 
        temperature: float = 0.7,
        use_fallback: bool = True
    ) -> str:
        """
        Query the LLM with automatic fallback.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum response length
            temperature (float): Generation temperature
            use_fallback (bool): Whether to use fallback if primary fails
        
        Returns:
            str: Generated response
        """
        # Try primary LLM first
        if self.primary_llm and self.primary_llm.is_available():
            try:
                response = self.primary_llm.generate_response(
                    prompt, max_length, temperature
                )
                if response and not response.startswith("I encountered an error"):
                    return response
            except Exception as e:
                logger.warning(f"Primary LLM failed: {str(e)}")
        
        # Try fallback if enabled and available
        if use_fallback and self.fallback_llm and self.fallback_llm.is_available():
            try:
                logger.info("Using fallback LLM")
                response = self.fallback_llm.generate_response(
                    prompt, max_length, temperature
                )
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Fallback LLM also failed: {str(e)}")
        
        # If all else fails, return fallback message
        return "I'm sorry, but I'm currently unable to generate a response. Please check the LLM configuration and try again."
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all available LLMs."""
        status = {
            "backend": self.backend,
            "primary_llm": None,
            "fallback_llm": None
        }
        
        if self.primary_llm:
            status["primary_llm"] = {
                "available": self.primary_llm.is_available(),
                "info": self.primary_llm.get_model_info()
            }
        
        if self.fallback_llm:
            status["fallback_llm"] = {
                "available": self.fallback_llm.is_available(),
                "info": self.fallback_llm.get_model_info()
            }
        
        return status


# Factory function
def create_llm_manager(
    backend: str = "huggingface",
    **kwargs
) -> LLMManager:
    """
    Factory function to create an LLM Manager.
    
    Args:
        backend (str): LLM backend to use
        **kwargs: Additional configuration parameters
    
    Returns:
        LLMManager: Initialized LLM manager
    """
    return LLMManager(backend=backend, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("=== Open Source LLM Integration Demo ===")
    
    # Test different configurations
    test_configs = [
        {"backend": "huggingface", "hf_model": "distilgpt2"},  # Lightweight option
        {"backend": "ollama", "ollama_model": "llama2:7b"}
    ]
    
    test_prompts = [
        "What is machine learning?",
        "Explain neural networks in simple terms.",
        "How does natural language processing work?",
        "Based on the context about transformers, what are attention mechanisms?"
    ]
    
    for config in test_configs:
        print(f"\n=== Testing {config['backend'].upper()} Backend ===")
        
        try:
            # Create LLM manager
            llm_manager = create_llm_manager(**config)
            
            # Get status
            status = llm_manager.get_status()
            print(f"LLM Status: {json.dumps(status, indent=2)}")
            
            # Test queries
            if status.get("primary_llm", {}).get("available", False):
                print("\n--- Testing Queries ---")
                
                for i, prompt in enumerate(test_prompts[:2], 1):  # Test first 2 prompts
                    print(f"\n{i}. Query: '{prompt}'")
                    
                    start_time = time.time()
                    response = llm_manager.query_llm(prompt, max_length=200)
                    query_time = time.time() - start_time
                    
                    print(f"Response ({query_time:.2f}s): {response[:200]}...")
            else:
                print("❌ LLM not available for testing")
        
        except Exception as e:
            print(f"❌ Error testing {config['backend']}: {str(e)}")
    
    print("\n✅ LLM integration testing completed!")
