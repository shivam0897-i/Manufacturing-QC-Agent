"""Prompts package for Manufacturing QC Agent."""

from prompts.templates import (
    PROMPTS,
    RECOMMENDATION_SYSTEM_PROMPT,
    RECOMMENDATION_USER_PROMPT,
    RGB_RECOMMENDATION_PROMPT,
    format_defects_summary,
    format_rules_context
)

__all__ = [
    "PROMPTS",
    "RECOMMENDATION_SYSTEM_PROMPT",
    "RECOMMENDATION_USER_PROMPT",
    "RGB_RECOMMENDATION_PROMPT",
    "format_defects_summary",
    "format_rules_context"
]

