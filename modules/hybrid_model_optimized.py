# modules/hybrid_model_optimized.py
"""
Memory-optimized HybridLM for RTX 4060 (8GB)
Strategy: Load models lazily, use gradient checkpointing, aggressive garbage collection
"""
from .bridge import Bridge

import torch
from torch import nn
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from .bridge import Bridge
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridLMOptimized(nn.Module):
    """
    Memory-optimized hybrid model for 8GB GPU.
    
    Key optimizations:
    1. Lazy loading - only load decoder when generating
    2. Encoder offloading - move BERT to CPU after encoding
    3. Aggressive memory cleanup
    4. Flash Attention 2 (if available)
    5. Gradient checkpointing
    """
    
    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        decoder_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_4bit: bool = True,
        offload_encoder: bool = True,  # Move BERT to CPU after encoding
        device: str = None
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.offload_encoder = offload_encoder
        self.decoder_loaded = False
        
        # Clear GPU memory before starting
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("=" * 70)
        logger.info("🚀 Initializing Memory-Optimized HybridLM")
        logger.info("=" * 70)
        
        # ============ ENCODER (BERT) - Load immediately ============
        logger.info(f"📥 Loading encoder: {encoder_name}")
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            torch_dtype=torch.float16  # Half precision
        ).to(self.device)
        self.encoder.eval()
        
        # Freeze encoder to save memory
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        encoder_dim = self.encoder.config.hidden_size
        logger.info(f"   ✅ Encoder loaded ({encoder_dim}d)")
        
        # ============ BRIDGE - Small, always in memory ============
        decoder_dim = 4096  # LLaMA 2 7B hidden size
        logger.info(f"🌉 Initializing bridge: {encoder_dim} → {decoder_dim}")
        self.bridge = Bridge(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            dropout=0.1
        ).to(self.device)
        logger.info("   ✅ Bridge initialized")
        
        # ============ DECODER - Load lazily ============
        self.decoder_name = decoder_name
        self.use_4bit = use_4bit
        self.decoder = None
        self.decoder_tokenizer = None
        
        logger.info(f"💾 GPU Memory after encoder: {self._get_gpu_memory():.2f} GB")
        logger.info("=" * 70)
    
    def _get_gpu_memory(self):
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def _load_decoder(self):
        """Load decoder only when needed."""
        if self.decoder_loaded:
            return
        
        logger.info("📥 Loading LLaMA decoder...")
        
        # Offload encoder to CPU to make room
        if self.offload_encoder and self.device == "cuda":
            logger.info("   📤 Offloading encoder to CPU...")
            self.encoder = self.encoder.cpu()
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load tokenizer
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(self.decoder_name)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
            self.decoder_tokenizer.pad_token_id = self.decoder_tokenizer.eos_token_id
        
        # 4-bit quantization config
        if self.use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.decoder_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager",  # Enable if available
                max_memory={0: "6GB"}  # Leave 1GB for overhead
            )
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.decoder_name,
                torch_dtype=torch.float16,
                device_map={"": 0},
                trust_remote_code=True
            )
        
        # Enable gradient checkpointing to save memory
        self.decoder.gradient_checkpointing_enable()
        
        self.decoder_loaded = True
        logger.info(f"   ✅ Decoder loaded")
        logger.info(f"💾 GPU Memory after decoder: {self._get_gpu_memory():.2f} GB")
    
    def _unload_decoder(self):
        """Unload decoder to free memory."""
        if not self.decoder_loaded:
            return
        
        logger.info("🗑️  Unloading decoder...")
        del self.decoder
        self.decoder = None
        self.decoder_loaded = False
        
        # Restore encoder to GPU if it was offloaded
        if self.offload_encoder and self.device == "cuda":
            logger.info("   📥 Restoring encoder to GPU...")
            self.encoder = self.encoder.to(self.device)
        
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"💾 GPU Memory after cleanup: {self._get_gpu_memory():.2f} GB")
    
    @torch.no_grad()
    def encode_contexts(self, texts: list[str], max_length: int = 512) -> dict:
        """
        Encode query + contexts using BERT.
        Note: If encoder was offloaded, temporarily move it to GPU.
        """
        # Ensure encoder is on correct device
        original_device = next(self.encoder.parameters()).device
        if original_device != self.device and self.device == "cuda":
            self.encoder = self.encoder.to(self.device)
        
        # Tokenize
        inputs = self.encoder_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        encoder_outputs = self.encoder(**inputs)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Project through bridge
        bridge_outputs = self.bridge(encoder_hidden_states)
        
        # Move encoder back if needed
        if self.offload_encoder and original_device != self.device:
            self.encoder = self.encoder.cpu()
            torch.cuda.empty_cache()
        
        return {
            "encoder_hidden_states": encoder_hidden_states.cpu(),  # Save to CPU
            "bridge_outputs": bridge_outputs.cpu(),
            "attention_mask": inputs["attention_mask"].cpu()
        }
    
    def construct_prompt(self, query: str, retrieved_docs: list, system_prompt: str = None):
        """Construct LLaMA 2 chat prompt."""
        if system_prompt is None:
            system_prompt = "You are HybridLM, a helpful AI assistant that answers questions accurately and concisely based on the provided context."
        
        # Build context
        context_parts = []
        for i, (title, content) in enumerate(retrieved_docs, 1):
            context_parts.append(f"[{i}] {title}: {content}")
        
        context_str = "\n".join(context_parts)
        
        # LLaMA 2 chat format
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Context:
{context_str}

Question: {query}

Provide a clear and accurate answer based solely on the context above. [/INST]"""
        
        return prompt
    
    @torch.no_grad()
    def generate(
        self,
        query: str,
        retrieved_docs: list[tuple[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        unload_after: bool = True  # Unload decoder after generation
    ) -> str:
        """
        Generate answer using LLaMA decoder.
        
        Args:
            unload_after: If True, unload decoder after generation to free memory
        """
        # Load decoder if not loaded
        if not self.decoder_loaded:
            self._load_decoder()
        
        # Construct prompt
        prompt = self.construct_prompt(query, retrieved_docs)
        
        # Tokenize
        inputs = self.decoder_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        outputs = self.decoder.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            repetition_penalty=1.1  # Reduce repetition
        )
        
        # Decode
        full_text = self.decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after [/INST])
        if "[/INST]" in full_text:
            answer = full_text.split("[/INST]")[-1].strip()
        else:
            answer = full_text
        
        # Optionally unload decoder to free memory
        if unload_after:
            self._unload_decoder()
        
        return answer
    
    def save_pretrained(self, save_path: str):
        """Save bridge weights."""
        self.bridge.save_pretrained(save_path)
    
    def load_pretrained(self, load_path: str):
        """Load bridge weights."""
        self.bridge.load_pretrained(load_path)