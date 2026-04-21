# Repository Guidelines

## Project Structure & Module Organization
`train.py`, `inference.py`, and `interactive_inference.py` are the main entrypoints. Core training logic lives in `trainer/`, generation models in `model/`, runtime pipelines in `pipeline/`, and shared helpers in `utils/`. The bundled Wan base-model code is under `wan/` with its own `configs/`, `modules/`, and `distributed/` packages. Runtime YAMLs live in `configs/`, prompt examples in `example/`, documentation in `docs/`, and images used by the README in `assets/`. Generated videos belong in `videos/`; large local weights are expected in ignored directories such as `longlive_models/` and `wan_models/`.

## Build, Test, and Development Commands
There is no separate build step; development is driven by Python entrypoints and shell wrappers.

```bash
pip install -r requirements.txt
bash train_init.sh
bash train_long.sh
bash inference.sh
bash interactive_inference.sh
```

`pip install -r requirements.txt` installs the training and inference dependencies. `train_init.sh` and `train_long.sh` launch distributed training with `torchrun` and the matching YAML in `configs/`. `inference.sh` runs offline generation from `configs/longlive_inference.yaml`; `interactive_inference.sh` runs the interactive pipeline with `configs/longlive_interactive_inference.yaml`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions, variables, files, and YAML keys, and `PascalCase` for classes such as `ScoreDistillationTrainer`. Keep modules focused and prefer adding configuration in `configs/longlive_*.yaml` instead of hardcoding paths or hyperparameters. This repo does not ship a Python formatter config, so preserve the surrounding style and keep imports, comments, and argument names consistent with nearby code.

## Testing Guidelines
No dedicated automated test suite is checked in yet. Validate changes by running the smallest relevant path you touched: a targeted inference command for pipeline/model edits, or the appropriate training launcher for trainer/config changes. When adding new configs or prompt formats, verify them against files in `example/` and confirm outputs land in `videos/` without breaking existing scripts.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Fix typo in README regarding training instructions`. Keep commits focused, reference the related issue, and sign off commits with `git commit -s`. Per `CONTRIBUTING.md`, open an issue before substantial work, keep PRs scoped to one concern, mark unfinished PRs with `[WIP]`, and include enough manual test detail for reviewers because the project does not currently rely on CI.
