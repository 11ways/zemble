"""Grammar specs shared by the symbol graph and duplication detection."""

from zemble.languages.catalog import SPECS, complexity_for, family_of, spec_for
from zemble.languages.spec import CallRule, ComplexitySpec, LanguageSpec, Role, Rule
from zemble.languages.visibility import Visibility

__all__ = [
    "SPECS",
    "CallRule",
    "ComplexitySpec",
    "LanguageSpec",
    "Role",
    "Rule",
    "Visibility",
    "complexity_for",
    "family_of",
    "spec_for",
]
