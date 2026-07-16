# GPU Model Weights Offloading

Resource optimization profiles for local execution:
- Layer offloading: inactive transformer layers are written to host RAM.
- Cache settings: model cache weights cleared dynamically on idle loops.
