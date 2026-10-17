"""Model backends, one module per kind of endpoint, and the registry of them.

Code outside this package should only need the protocols in
:mod:`.base` and the functions here: look a backend up by name with
:func:`load`, and it builds itself from the config with
``Backend.from_config``. Other packages can add backends by declaring an
entry point in the ``anki_lang_cards.backends`` group, named for the service
and pointing at the backend class; they cannot replace a built-in one.
"""

import importlib.metadata
from collections.abc import Mapping, Sequence
from typing import Final

from . import base

ENTRY_POINT_GROUP: Final[str] = "anki_lang_cards.backends"

# The model specs that no backend claims by prefix go to this one.
FALLBACK: Final[str] = "llm"

# Entry points like the ones plugins declare, so that built-in and plugin
# backends load the same way. They are imported on first use, so that e.g.
# the Anthropic SDK is only loaded when a run actually uses it.
_BUILT_IN: Final[Mapping[str, importlib.metadata.EntryPoint]] = {
    name: importlib.metadata.EntryPoint(name, value, ENTRY_POINT_GROUP)
    for name, value in (
        ("anthropic", "anki_lang_cards.backends.anthropic_sdk:AnthropicBackend"),
        ("tgw", "anki_lang_cards.backends.tgw:TGWBackend"),
        ("llm", "anki_lang_cards.backends.llm_fallback:LLMBackend"),
    )
}


class UnknownBackendError(LookupError):
    """No backend is registered under the requested name."""


def _registered() -> Mapping[str, importlib.metadata.EntryPoint]:
    """Every backend name, mapped to the entry point that loads it.

    The built-ins come last so that a plugin cannot replace one of them.
    """
    found = {
        entry_point.name: entry_point
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
        entry_point = _registered()[name]
    except KeyError:
        raise UnknownBackendError(
            f"Unknown model service {name!r}; choose from {', '.join(names())}"
        ) from None
    backend: type[base.Backend] = entry_point.load()
    return backend
