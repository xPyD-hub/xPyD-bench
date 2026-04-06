"""Request body templating with Jinja2-style variable substitution (M37)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class TemplateError(Exception):
    """Raised when template rendering fails."""


_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def load_template_vars(path: str) -> dict[str, Any]:
    """Load template variables from a JSON or YAML file.

    Parameters
    ----------
    path:
        Path to a .json or .yaml/.yml file containing a flat dict of variables.

    Returns
    -------
    dict:
        Variable name -> value mapping.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Template vars file not found: {path}")

    text = p.read_text(encoding="utf-8")
    ext = p.suffix.lower()

    if ext == ".json":
        data = json.loads(text)
    elif ext in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        raise ValueError(
            f"Unsupported template vars format: {ext}. Use .json, .yaml, or .yml."
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"Template vars file must contain a dict/object, got {type(data).__name__}"
        )
    return data


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Render a template string by substituting ``{{ variable }}`` placeholders.

    Parameters
    ----------
    template:
        String potentially containing ``{{ var }}`` placeholders.
    variables:
        Variable name -> value mapping.

    Returns
    -------
    str:
        Rendered string with all placeholders replaced.

    Raises
    ------
    TemplateError:
        If a placeholder references an undefined variable.
    """

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in variables:
            raise TemplateError(
                f"Undefined template variable: '{name}'. "
                f"Available variables: {sorted(variables.keys())}"
            )
        return str(variables[name])

    return _VAR_PATTERN.sub(_replace, template)


def apply_templates(
    prompts: list[str], variables: dict[str, Any]
) -> list[str]:
    """Apply template variable substitution to a list of prompt strings.

    Parameters
    ----------
    prompts:
        List of prompt strings, potentially containing ``{{ var }}`` placeholders.
    variables:
        Variable name -> value mapping.

    Returns
    -------
    list[str]:
        Prompts with all placeholders replaced.
    """
    return [render_template(p, variables) for p in prompts]
