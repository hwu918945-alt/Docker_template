#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template model wrapper (not used directly by run_infer.py).
Keeps a minimal interface in case the platform expects it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    weights_dir: Path


class Model:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def load(self) -> None:
        # Placeholder: actual loading is handled in run_infer.py
        pass

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Use run_infer.py for full pipeline inference.")


def build_model(weights_dir: str | Path) -> Model:
    return Model(ModelConfig(weights_dir=Path(weights_dir)))
