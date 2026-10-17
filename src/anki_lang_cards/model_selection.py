"""Model selection: which backend serves a run, and with which features.

The pipelines' command-line tools share the options defined here and call
:func:`make_runner` to get a query runner for whatever the arguments and
config select. Backends are found through the :mod:`.backends` registry, so
nothing here knows about any particular one.
"""

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import attrs

from . import backends, caches, config, runner
from .backends import base

_FEATURE_DESCRIPTIONS: Final[Mapping[base.Feature, str]] = {
    base.Feature.BATCH: "Batching",
    base.Feature.PROMPT_CACHING: "Prompt caching",
    base.Feature.STRUCTURED_OUTPUT: "Structured output",
}

_FEATURE_FLAGS: Final[Mapping[base.Feature, str]] = {
    base.Feature.BATCH: "batch",
    base.Feature.PROMPT_CACHING: "prompt-caching",
    base.Feature.STRUCTURED_OUTPUT: "structured-output",
}


class SelectionError(Exception):
    """The configuration and command line do not settle on a usable model."""


class ModelNotConfiguredError(SelectionError):
    """No model was named on the command line or in the config."""


class UnknownServiceError(SelectionError):
    """No backend is registered under the requested service name."""


class BackendConfigurationError(SelectionError):
    """The selected backend needs something the config file lacks."""


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
        return backends.load(self.service)


def _load(service: str) -> type[base.Backend]:
    try:
        return backends.load(service)
    except backends.UnknownBackendError as e:
        raise UnknownServiceError(str(e)) from e


def resolve(spec: str, service: str | None = None) -> ResolvedModel:
    """Work out which backend serves a model spec.

    A spec written ``<service>/<model>`` goes to that service's backend if
    the backend claims its prefix; anything else goes to the fallback.

    :param service: Force a particular backend, e.g. ``"llm"`` to reach a
        Claude model through the ``llm`` library rather than the SDK.
    :raises UnknownServiceError: If ``service`` names no backend.
    """
    if service is None:
        prefix, slash, _ = spec.partition("/")
        claimed = slash and prefix in backends.names() and _load(prefix).claims_prefix
        service = prefix if claimed else backends.FALLBACK

    if _load(service).claims_prefix:
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
        help="The model to use, e.g. 'anthropic/claude-sonnet-5'. A model "
        "prefixed with a backend's name goes to that backend if it takes "
        f"prefixes; anything else goes to the {backends.FALLBACK!r} backend. "
        "Defaults to [models] default_model from the config; one of the two "
        "is required.",
    )
    group.add_argument(
        "--model-service",
        choices=backends.names(),
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
        "--structured-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Have the model's API enforce the response format, so responses "
        "always parse. On by default where the model supports it.",
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

    :raises SelectionError: If the arguments and config do not settle on a
        usable model and backend.
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
            _feature_request(
                base.Feature.STRUCTURED_OUTPUT,
                args.structured_output,
                conf.models.structured_output,
            ),
        ),
        backend_class.supported_features(resolved.model_name),
        backend_description=(
            f"the {resolved.service} backend (model {resolved.model_name!r})"
        ),
        confirm=confirm,
    )
    try:
        backend = backend_class.from_config(resolved.model_name, conf, features)
    except base.ConfigurationError as e:
        raise BackendConfigurationError(str(e)) from e
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
