# MLGym

## Installation

### 1. Clone and install dependencies

```bash
git clone git@github.com:yogendrahexo/MLGym.git
cd MLGym
git checkout dev

# Create conda environment from the environment.yaml file
conda env create -f environment.yaml
conda activate mlgym
```

Have generated a `gen_requirements.txt` from my environment in case needed.

### 2. Create a .env file

Create a `.env` file in the MLGym directory (`MLGym/.env`) to save all the environment variables including API keys.

```bash
# Env variables
MLGYM_CONFIG_ROOT="<path_to_MLGYM_root>/configs"
MLGYM_TASK_CONFIG_DIR="<path_to_MLGYM_root>/configs/tasks"
MLGYM_WORKSPACE_PATH="<path_to_MLGYM_root>/workspace"
MLGYM_ENV_TIMEOUT=10000
MLGYM_ACTION_SHORT_TIMEOUT=60
MLGYM_ACTION_LONG_TIMEOUT=10000
MLGYM_MODEL_MAX_RETRIES=3

# OpenAI and Anthropic API keys
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""

# Vertex AI
GOOGLE_APPLICATION_CREDENTIALS="security-key.json"
GOOGLE_CLOUD_PROJECT="project-id"
VERTEXAI_PROJECT="project-id"
VERTEXAI_LOCATION="region"

# Azure OpenAI
API_BASE_URL="https://your-endpoint.openai.azure.com/"
API_KEY="your-azure-api-key"
API_VERSION="2024-12-01-preview"
MODEL_NAME="azure/gpt-4o-2024-11-20"

# Cost limits
TOTAL_COST_LIMIT=0.0
PER_INSTANCE_COST_LIMIT=3.0
```

Note: Copy the `.env.template` file and fill in the correct values for your setup.

### 3. Docker setup

Install NVIDIA Container Toolkit if not already installed:

```bash
sudo dnf install -y nvidia-container-toolkit
```

Pull the MLGym docker image:

```bash
docker pull aigym/mlgym-agent:latest
```

Test the container:

```bash
docker run -it --gpus all --name test aigym/mlgym-agent /bin/bash
ls -la
exit
```

## Running MLGym

### Basic usage

Run MLGym with default settings:

```bash
python run.py --task_config_path tasks/imageClassificationCifar10.yaml
```

Default LLM is Vertex Claude Sonnet 3.7.

### Advanced usage

Run with additional parameters:

```bash
python run.py \
  --container_type docker \
  --task_config_path tasks/imageClassificationCifar10.yaml \
  --model litellm:vertex_ai \
  --per_instance_cost_limit 4.00 \
  --agent_config_path configs/agents/default.yaml \
  --temp 1 \
  --gpus 0 \
  --max_steps 50 \
  --aliases_file ./docker/aliases.sh
```

## Troubleshooting

- **Image not found**: If Docker is unable to locate the image after pulling, check if the Docker host is set correctly.

- **NVIDIA CDI spec errors on Linux**: If you encounter errors like `Error: setting up CDI devices: unresolvable CDI devices nvidia.com/gpu=all`, run these commands:
  ```bash
  sudo mkdir /etc/cdi
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  sudo touch /etc/containers/nodocker
  ```

# Trajectories

After running the experiments you can find the trajectories, logs and results in the `trajectories/user/litellm-vertex_ai/../..`

## Visualizing Trajectories

MLGym provides tools to visualize and analyze your experiment trajectories:

### Trajectory Visualizer

View detailed trajectory information with the interactive Streamlit visualizer:

```bash
streamlit run demo/trajectory_visualizer.py -- --trajectory_dir <absolute_path_to_trajectory>
```

### Demo Application

The demo application showcased in our YouTube video is a Streamlit app that replays selected experiments:

```bash
streamlit run demo/demo.py
```

This interactive interface allows you to explore experiment outcomes and agent interactions.

## Commands running on Vignesh Server
```bash
python run.py   --container_type docker   --task_config_path tasks/imageClassificationCifar10.yaml   --model litellm:vertex_ai/claude-3-7-sonnet@20250219 --per_instance_cost_limit 4.00   --agent_config_path configs/agents/default.yaml   --temp 1   --gpus 0   --max_steps 50   --aliases_file ./docker/aliases.sh
```

```bash
python run.py --task_config_path tasks/imageClassificationCifar10.yaml
```