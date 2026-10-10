"""Model backends, one module per kind of endpoint, and the registry of them.

Code outside this package should only need the protocols in
:mod:`.base` and the functions here: look a backend up by name with
:func:`load`, and it builds itself from the config with
``Backend.from_config``. Other packages can add backends by declaring an
entry point in the ``anki_lang_cards.backends`` group, named for the service
and pointing at the backend class; they cannot replace a built-in one.
"""

import importlib
import importlib.metadata
from collections.abc import Mapping, Sequence
from typing import Final

from . import base

ENTRY_POINT_GROUP: Final[str] = "anki_lang_cards.backends"

# The model specs that no backend claims by prefix go to this one.
FALLBACK: Final[str] = "llm"

# Imported on first use, so that e.g. the Anthropic SDK is only loaded when
# a run actually uses it.
_BUILT_IN: Final[Mapping[str, str]] = {
    "anthropic": "anki_lang_cards.backends.anthropic_sdk:AnthropicBackend",
    "tgw": "anki_lang_cards.backends.tgw:TGWBackend",
    "llm": "anki_lang_cards.backends.llm_fallback:LLMBackend",
}


class UnknownBackendError(LookupError):
    """No backend is registered under the requested name."""


def _registered() -> Mapping[str, str]:
    """Every backend name, mapped to the ``module:attribute`` it lives at."""
    found = {
        entry_point.name: entry_point.value
        for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    }
    return {**found, **_BUILT_IN}


def names() -> Sequence[str]:
    """The names of every available backend, built-in ones first."""
    registered = _registered()
    return (*_BUILT_IN, *sorted(registered.keys() - _BUILT_IN.keys()))


def load(name: str) -> type[base.Backend]:
    """The backend class registered under ``name``.

    :raises UnknownBackendError: If there is none.
    """
    try:
        target = _registered()[name]
    except KeyError:
        raise UnknownBackendError(
            f"Unknown model service {name!r}; choose from {', '.join(names())}"
        ) from None
    module_name, _, attribute = target.partition(":")
    backend: type[base.Backend] = getattr(
        importlib.import_module(module_name), attribute
    )
    return backend
