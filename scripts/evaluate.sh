#!/bin/bash
python -c "from src.models.evaluate import evaluate_model; import yaml; evaluate_model(yaml.safe_load(open('src/config/config.yaml')))"