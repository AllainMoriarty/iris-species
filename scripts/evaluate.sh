#!/usr/bin/env bash
set -e

BASE_CONFIG="src/config/config.yaml"

if [ -n "$1" ]; then
  EXP_CONFIG="$1"
  echo "Evaluating with experiment config: $EXP_CONFIG"

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

with open("$BASE_CONFIG") as f:
    config = yaml.safe_load(f)

with open("$EXP_CONFIG") as f:
    exp_cfg = yaml.safe_load(f)

config = deep_update(deepcopy(config), exp_cfg)

evaluate_model(config)
EOF

else
  echo "Evaluating with base config only"

  python - <<EOF
import yaml
from src.models.evaluate import evaluate_model

with open("$BASE_CONFIG") as f:
    config = yaml.safe_load(f)

evaluate_model(config)
EOF
fi
