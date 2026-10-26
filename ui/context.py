"""Per-run context handed to every page."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnguard.config import Config
from churnguard.data import DataValidationError

from .workspace import Population, Workspace, population_for


@dataclass
class AppContext:
    ws: Workspace | None          # None only while no model is available (Data Explorer still works)
    config: Config
    threshold: float
    population_kind: str
    uploaded: pd.DataFrame | None

    def population(self, kind: str | None = None, *, show_errors: bool = True) -> Population | None:
        """Customers selected in the sidebar (or ``kind``), scored at the current threshold."""
        if self.ws is None:
            return None
        try:
            return population_for(self.ws, kind or self.population_kind, self.threshold, self.uploaded)
        except DataValidationError as exc:
            if show_errors:
                st.error(str(exc))
            return None
