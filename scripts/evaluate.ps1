# PowerShell
param (
    [string]$ExperimentConfig = ""
)

$BaseConfig = "src/config/config.yaml"

if ($ExperimentConfig -ne "") {
    Write-Host "Evaluating with experiment config: $ExperimentConfig"

    python - <<EOF
import yaml
from copy import deepcopy
from src.models.evaluate import evaluate_model

def deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base

with open("$BaseConfig") as f:
    config = yaml.safe_load(f)

with open("$ExperimentConfig") as f:
    exp_cfg = yaml.safe_load(f)

config = deep_update(deepcopy(config), exp_cfg)

evaluate_model(config)
EOF

} else {
    Write-Host "Evaluating with base config only"

    python - <<EOF
import yaml
from src.models.evaluate import evaluate_model

with open("$BaseConfig") as f:
    config = yaml.safe_load(f)

evaluate_model(config)
EOF
}
