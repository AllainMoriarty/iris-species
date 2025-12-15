# Powershell
param (
    [string]$ExperimentConfig = ""
)

if ($ExperimentConfig -ne "") {
    python -m src.pipelines.training_pipeline `
        src/config/config.yaml `
        $ExperimentConfig
} else {
    python -m src.pipelines.training_pipeline `
        src/config/config.yaml
}