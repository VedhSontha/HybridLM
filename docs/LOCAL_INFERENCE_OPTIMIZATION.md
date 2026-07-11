# Local Inference GPU Optimization

Minimizing hardware overhead for local LLM runs:
- Quantization format: 4-bit NormalFloat (NF4) model layers.
- CPU memory offloading: inactive layers offloaded to system memory.
- Lazy-loading loads model weights only when processing report requests.
