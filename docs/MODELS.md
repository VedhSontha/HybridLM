# LLM Models Configuration

Guidelines for model weights:
- Local models: Llama 3.2 (3B parameters) quantized to 4-bit.
- Representation projection: BERT (768d embeddings) projected into LLaMA (4096d space).
- Target device limits: optimized for edge GPUs under 8GB VRAM.
