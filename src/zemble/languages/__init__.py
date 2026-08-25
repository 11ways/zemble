"""Grammar specs shared by the symbol graph and duplication detection."""

from zemble.languages.catalog import SPECS, family_of, spec_for
from zemble.languages.spec import CallRule, LanguageSpec, Role, Rule
from zemble.languages.visibility import Visibility

__all__ = ["SPECS", "CallRule", "LanguageSpec", "Role", "Rule", "Visibility", "family_of", "spec_for"]
