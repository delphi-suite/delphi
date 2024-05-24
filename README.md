# Setup

1. Clone the repo
```shell
git clone https://github.com/delphi-suite/delphi.git
cd delphi  
```
2. Make & activate python >= 3.10 virtual env
```shell
python3.10 -m venv .venv
source .venv/bin/activate
```
3. Install the project in editable state  
`pip install -e .`  
See `[project.optional-dependencies]` section in `pyproject.toml` for additional dependencies, e.g. you may want to `pip install -e ."[dev,mamba_cuda]"`
4. get your HuggingFace and W&B tokens and put them in the environment variables
```shell
export HF_TOKEN=...
export WANDB_API_KEY=...
```
