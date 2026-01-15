# Flight Recommendation Systems

This repository contains implementations of two flight recommendation optimization approaches: **MIPRO** (Multi-task Improvement through Prompt Optimization) and **OPRO** (Optimization by Prompting). Both approaches are designed to generate optimal flight recommendations that satisfy user preferences and constraints using feature-enriched flight data.

## Repository Structure

```
final-code/
├── README.md                 # Project documentation
├── mipro/                    # MIPRO implementation
│   ├── mipro.py             # Main MIPRO optimizer
│   ├── flightdata.py        # Flight data loading and processing
│   ├── load_annotations.py  # Annotation loading utilities
│   └── tokens.py            # Token calculation utilities
└── opro/                     # OPRO implementation
    ├── opro.py              # Main OPRO optimizer
    ├── flightdata.py        # Flight data loading and processing
    └── load_annotations.py  # Annotation loading utilities
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- DSPy framework
- Pandas, NumPy
- Local LLM model (Llama 3.2 3B Instruct recommended)

### SGLang Server Configuration

Before running any of the optimization scripts, launch the SGLang server with the desired model and configuration:

```bash
python3 -m sglang.launch_server \
    --model-path /home/ysonale/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
    --port 30000 \
    --attention-backend triton \
    --sampling-backend pytorch \
    --mem-fraction-static 0.48
```

**Configuration Parameters:**
- `--port`: Server port (default: 30000)
- `--model-path`: Path to the LLM model
- `--mem-fraction-static`: GPU memory allocation (adjust based on available resources)

### Language Model Configuration

Update the language model configuration in both `mipro/mipro.py` and `opro/opro.py`:

```python
lm = dspy.LM(
    model="openai/Llama-3.2-3B-Instruct",
    api_base="http://localhost:30000/v1",
    api_key="dummy",
    max_tokens=1024,
    temperature=0.4,
    top_p=1
)
```

**Configuration Notes:**
- Update `api_base` to match your SGLang server port
- When using Meta Hugging Face models, prefix the model name with `openai/` to ensure compatibility
- Adjust `temperature` and `top_p` parameters as needed for your use case
- The `api_key` value is a placeholder when using local models