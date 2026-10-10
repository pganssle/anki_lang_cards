"""Choosing a backend and its features from the config and command line."""

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import attrs

from .. import caches, config
from . import anthropic_sdk, base, llm_fallback, runner, tgw

DEDICATED_BACKENDS: Final[Mapping[str, type[base.Backend]]] = {
    "anthropic": anthropic_sdk.AnthropicBackend,
    "tgw": tgw.TGWBackend,
}
"""Backends that claim model specs starting with ``<service>/``."""

FALLBACK_BACKEND: Final[type[base.Backend]] = llm_fallback.LLMBackend

SERVICES: Final[Sequence[str]] = (*DEDICATED_BACKENDS, FALLBACK_BACKEND.service)

_FEATURE_DESCRIPTIONS: Final[Mapping[base.Feature, str]] = {
    base.Feature.BATCH: "Batching",
    base.Feature.PROMPT_CACHING: "Prompt caching",
}

_FEATURE_FLAGS: Final[Mapping[base.Feature, str]] = {
    base.Feature.BATCH: "batch",
    base.Feature.PROMPT_CACHING: "prompt-caching",
}


class SelectionError(Exception):
    """The configuration and command line do not settle on a usable model."""


class ModelNotConfiguredError(SelectionError):
    """No model was named on the command line or in the config."""


class FeatureUnavailableError(SelectionError):
    """A feature the user required is unsupported, and they declined to go on."""


@attrs.frozen
class ResolvedModel:
    spec: str
    """The model as the user wrote it, e.g. ``anthropic/claude-sonnet-5``."""

    service: str
    model_name: str
    """The model as its backend's API names it, e.g. ``claude-sonnet-5``."""

    @property
    def backend_class(self) -> type[base.Backend]:
        return DEDICATED_BACKENDS.get(self.service, FALLBACK_BACKEND)


def resolve(spec: str, service: str | None = None) -> ResolvedModel:
    """Work out which backend serves a model spec.

    :param service: Force a particular backend, e.g. ``"llm"`` to reach a
        Claude model through the ``llm`` library rather than the SDK.
    """
    if service is None:
        prefix, slash, _ = spec.partition("/")
        service = prefix if slash and prefix in DEDICATED_BACKENDS else "llm"
    elif service not in SERVICES:
        raise ValueError(f"Unknown model service {service!r}; choose from {SERVICES}")

    if service in DEDICATED_BACKENDS:
        model_name = spec.removeprefix(f"{service}/")
    else:
        model_name = spec
    return ResolvedModel(spec=spec, service=service, model_name=model_name)


@attrs.frozen
class FeatureRequest:
    feature: base.Feature
    requested: bool | None
    """True or False if the user said so, None to use it where available."""

    source: str = ""
    """Where the user said so, for messages; e.g. ``"--batch"``."""


def choose_features(
    requests: Sequence[FeatureRequest],
    available: frozenset[base.Feature],
    *,
    backend_description: str,
    confirm: Callable[[str], bool],
) -> frozenset[base.Feature]:
    """Decide which features to use.

    :param confirm: Asked whether to carry on without a required feature.
    :raises FeatureUnavailableError: If ``confirm`` says not to.
    """
    enabled: set[base.Feature] = set()
    for request in requests:
        if request.requested is False:
            continue
        if request.feature in available:
            enabled.add(request.feature)
            continue
        if request.requested is None:
            continue

        problem = (
            f"{_FEATURE_DESCRIPTIONS[request.feature]} was requested "
            f"({request.source}), but {backend_description} does not support it."
        )
        if not confirm(f"{problem} Continue without it?"):
            raise FeatureUnavailableError(
                f"{problem} Pass --no-{_FEATURE_FLAGS[request.feature]} "
                "to run without it."
            )
    return frozenset(enabled)


def confirm_on_terminal(question: str) -> bool:
    """Ask a yes/no question if there is someone at a terminal to answer it."""
    if not sys.stdin.isatty():
        return False
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the model-selection options shared by the pipeline CLIs."""
    group = parser.add_argument_group("model selection")
    group.add_argument(
        "--model-name",
        "--model",
        type=str,
        default=None,
        help="The model to use, e.g. 'anthropic/claude-sonnet-5'. Models "
        "prefixed with 'anthropic/' or 'tgw/' use a dedicated backend; "
        "anything else goes through the `llm` library and its plugins. "
        "Defaults to [models] default_model from the config; one of the two "
        "is required.",
    )
    group.add_argument(
        "--model-service",
        choices=SERVICES,
        default=None,
        help="Force a backend instead of choosing one from the model name.",
    )
    group.add_argument(
        "--batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send requests as a discounted batch job, which can take a while "
        "to come back. On by default where the backend supports it.",
    )
    group.add_argument(
        "--prompt-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache the shared system prompt between requests. On by default "
        "where the backend supports it.",
    )
    group.add_argument(
        "--max-simultaneous-queries",
        type=int,
        default=5,
        help="Maximum number of requests in flight at once when not batching.",
    )


def _feature_request(
    feature: base.Feature, cli_value: bool | None, config_value: bool | None
) -> FeatureRequest:
    flag = _FEATURE_FLAGS[feature]
    if cli_value is not None:
        return FeatureRequest(feature, cli_value, f"--{flag}")
    return FeatureRequest(
        feature, config_value, f"{flag.replace('-', '_')} = true in [models]"
    )


def model_spec(args: argparse.Namespace, conf: config.Config) -> str:
    """The model the command line or, failing that, the config names.

    :raises ModelNotConfiguredError: If neither names one.
    """
    if args.model_name is not None:
        return args.model_name
    if conf.models.default_model is not None:
        return conf.models.default_model
    if conf.anthropic is not None and conf.anthropic.default_model is not None:
        return conf.anthropic.default_model
    raise ModelNotConfiguredError(
        "No model is configured: pass --model, or set default_model under "
        "[models] in the config file."
    )


def make_runner(
    args: argparse.Namespace,
    conf: config.Config,
    cache_type: caches.CacheTypes,
    *,
    confirm: Callable[[str], bool] = confirm_on_terminal,
) -> tuple[ResolvedModel, runner.QueryRunner]:
    """Build the runner for the model the CLI arguments and config select.

    :raises ModelNotConfiguredError: See :func:`model_spec`.
    :raises FeatureUnavailableError: See :func:`choose_features`.
    """
    resolved = resolve(model_spec(args, conf), args.model_service)
    backend_class = resolved.backend_class
    features = choose_features(
        (
            _feature_request(base.Feature.BATCH, args.batch, conf.models.batch),
            _feature_request(
                base.Feature.PROMPT_CACHING,
                args.prompt_caching,
                conf.models.prompt_caching,
            ),
        ),
        backend_class.features,
        backend_description=(
            f"the {resolved.service} backend (model {resolved.model_name!r})"
        ),
        confirm=confirm,
    )
    backend = _make_backend(resolved, conf, features)
    query_cacher = caches.QueryCacher(
        cache_type=cache_type,
        model_name=resolved.spec,
        model_service=resolved.service,
    )
    return resolved, runner.QueryRunner(
        backend,
        query_cacher,
        batch=base.Feature.BATCH in features,
        max_concurrency=args.max_simultaneous_queries,
    )


def _make_backend(
    resolved: ResolvedModel,
    conf: config.Config,
    features: frozenset[base.Feature],
) -> base.Backend:
    match resolved.service:
        case "anthropic":
            anthropic_conf = conf.anthropic or config.AnthropicConfig()
            return anthropic_sdk.AnthropicBackend(
                resolved.model_name,
                api_key=anthropic_conf.api_key,
                base_url=anthropic_conf.base_url,
                max_tokens=anthropic_conf.max_tokens
                or anthropic_sdk.DEFAULT_MAX_TOKENS,
                effort=anthropic_conf.effort,
                prompt_caching=base.Feature.PROMPT_CACHING in features,
            )
        case "tgw":
            if conf.tgw is None:
                raise ValueError(
                    "A tgw model was selected, but there is no [tgw] config."
                )
            return tgw.TGWBackend(
                resolved.model_name,
                api_base_url=conf.tgw.api_base_url,
                max_connections=conf.tgw.max_connections,
                request_timeout=conf.tgw.request_timeout,
                max_tokens=conf.tgw.max_tokens,
            )
        case _:
            keys = {}
            if conf.anthropic is not None and conf.anthropic.api_key is not None:
                keys["anthropic"] = conf.anthropic.api_key
            return llm_fallback.LLMBackend(resolved.model_name, keys=keys)
