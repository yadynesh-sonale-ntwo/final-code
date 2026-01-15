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
- Local LLM model (Llama 3.2 3B Instruct recommended)
- GPU with sufficient VRAM

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### SGLang Server Configuration

Before running any of the optimization scripts, launch the SGLang server with the desired model and configuration:

```bash
python3 -m sglang.launch_server \
    --model-path </path/to/model-name> \
    --port <port-number> \
    --attention-backend triton \
    --sampling-backend pytorch \
    --mem-fraction-static 0.48
```

### Language Model Configuration

Update the language model configuration in both `mipro/mipro.py` and `opro/opro.py`:

```python
lm = dspy.LM(
    model="openai/<model-name>",
    api_base="http://localhost:<port-number>/v1",
    api_key="dummy",
    max_tokens=1024,
    temperature=0.4,
    top_p=1
)
```

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Feature Engineering Setup

Clone and set up the lambdaMART-flight-ranker repository, then run the feature engineering script:

```bash
git clone https://github.com/neuralnurture/lambdaMART-flight-ranker.git
cd lambdaMART-flight-ranker
# Follow the instructions in the repository till you run the below line then stop
python feature_engineering.py
```

### Step 3: Launch SGLang Server

```bash
sglang server start
```

### Step 4: Run OPRO or MIPRO

**To run OPRO:**
```bash
cd opro/
python opro.py
```

**To run MIPRO:**
```bash
cd mipro/
python mipro.py
```
