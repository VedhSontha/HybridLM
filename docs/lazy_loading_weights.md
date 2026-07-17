# Lazy Weight Loading and VRAM Limits

Optimizing model start times on target hardware:
- Model layers are loaded from disk only when an SRE report request is active.
- Unloads inactive layers dynamically to maintain edge memory targets.
