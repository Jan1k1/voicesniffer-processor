from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import FastAPI
from hypercorn.asyncio import serve
from hypercorn.config import Config
from voicesniffer_runtime.model_store import ModelStore, default_registry_path, load_registry

from voicesniffer_processor.app import Transcribe, create_app
from voicesniffer_processor.batching import MicroBatcher
from voicesniffer_processor.rules import RulePack
from voicesniffer_processor.settings import ProcessorSettings
from voicesniffer_processor.toxicity import ToxicityClassifier
from voicesniffer_processor.transcription import load_transcription_adapter

AdapterLoader = Callable[..., Transcribe]
LOGGER = logging.getLogger("voicesniffer.processor")


def build_application(
    environment: Mapping[str, str] | None = None,
    *,
    adapter_loader: AdapterLoader = load_transcription_adapter,
    toxicity_classifier: ToxicityClassifier | None = None,
) -> FastAPI:
    values = os.environ if environment is None else environment
    settings = ProcessorSettings.from_environment(values)
    model_id = _required_value(values, "VOICESNIFFER_MODEL_ID")
    models_dir = Path(_required_value(values, "VOICESNIFFER_MODELS_DIR")).resolve()
    registry_path = Path(
        values.get("VOICESNIFFER_MODEL_REGISTRY", str(default_registry_path()))
    ).resolve()
    rules_dir = Path(_required_value(values, "VOICESNIFFER_RULES_DIR")).resolve()
    model_threads = _bounded_integer(values, "VOICESNIFFER_MODEL_THREADS", 1, 64)

    registry = load_registry(registry_path)
    try:
        model = registry[model_id]
    except KeyError as exception:
        raise ValueError(f"unknown model: {model_id}") from exception
    model_dir = ModelStore(models_dir).fetch(model)
    adapter = adapter_loader(
        model_id,
        model_dir,
        threads=model_threads,
        registry=registry,
    )
    transcribe = _wrap_with_batching(adapter, values)
    rule_packs = _load_rule_packs(rules_dir)
    _report_unmoderatable_languages(model.languages, rule_packs)
    _report_untranscribable_languages(model.languages, rule_packs)
    application = create_app(
        settings,
        transcribe=transcribe,
        rule_packs=rule_packs,
        toxicity_classifier=toxicity_classifier,
    )
    if isinstance(transcribe, MicroBatcher):
        application.state.micro_batcher = transcribe
        # Starlette 1.x dropped `add_event_handler`; the router list is the
        # supported way to attach a shutdown hook to an already-built app.
        application.router.on_shutdown.append(transcribe.close)
    return application


def _wrap_with_batching(adapter: Transcribe, values: Mapping[str, str]) -> Transcribe:
    """Micro-batching is opt-in and off by default.

    It is a GPU lever, not a general one. Batching funnels every utterance
    through a single decode thread; on CPU that trades away the worker-level
    parallelism the processor relies on (measured 11.8 utt/s at four workers
    versus 8.4 when the same cores are spent on intra-op threading instead), so
    turning it on there would be a regression. Set
    ``VOICESNIFFER_MAX_BATCH_SIZE`` above 1 only where one device is doing the
    arithmetic and would otherwise idle between clips.
    """
    max_batch_size = _bounded_integer(values, "VOICESNIFFER_MAX_BATCH_SIZE", 1, 64)
    if max_batch_size == 1:
        return adapter
    batch = getattr(adapter, "batch", None)
    if batch is None:
        raise ValueError("VOICESNIFFER_MAX_BATCH_SIZE requires a batching transcriber")
    # Zero on measurement, not on principle: a 25 ms window cost 12% of
    # throughput and 100 ms of p95 against pure free-drain batching, because
    # requests already arrive faster than the model consumes them.
    window_ms = _bounded_integer(values, "VOICESNIFFER_BATCH_WINDOW_MS", 0, 500, default=0)
    LOGGER.info(
        "micro_batching enabled max_batch_size=%s window_ms=%s",
        max_batch_size,
        window_ms,
    )
    return MicroBatcher(
        batch,
        max_batch_size=max_batch_size,
        window_seconds=window_ms / 1_000,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    values = os.environ
    settings = ProcessorSettings.from_environment(values)
    application = build_application(values)
    configuration = Config.from_toml(values.get("VOICESNIFFER_HYPERCORN_CONFIG", "hypercorn.toml"))
    configuration.bind = [f"{settings.bind_host}:{settings.bind_port}"]
    configuration.workers = 1
    asyncio.run(serve(application, configuration))


def _report_unmoderatable_languages(
    model_languages: tuple[str, ...],
    rule_packs: Mapping[str, RulePack],
) -> tuple[str, ...]:
    """Name, at startup, every language this node can transcribe but not moderate.

    The two lists are set independently -- ``models.toml`` describes what the
    speech model was trained on, ``VOICESNIFFER_RULES_DIR`` describes what we
    have written rules for -- and nothing used to compare them. The shipped
    combination is a 25-language model against 23 packs, so ``fi`` and ``mt``
    are advertised in ``config.yml`` and the panel while no rule in the process
    can ever fire on them. A server pinned to one of those was answered 200 with
    ``severity: 0`` on every utterance and had no way to find out.

    Deliberately loud rather than fatal. Refusing to start would crash-loop the
    live cloud node the moment it restarts, and a self-hosted operator trimming
    the rules directory to the languages they actually run is a legitimate thing
    to do. What must not happen is a *request* being moderated by nothing, and
    that is refused outright in ``app.moderate`` rather than warned about.

    Returned as well as logged so a caller can assert on it.
    """
    missing = tuple(code for code in model_languages if code not in rule_packs)
    if missing:
        LOGGER.error(
            "languages_without_rules model_languages=%s installed_packs=%s missing=%s "
            "hint=requests pinned to these are refused with language_unsupported; "
            "either ship a rule pack or stop offering the language",
            len(model_languages),
            len(rule_packs),
            ",".join(missing),
        )
    return missing


def _report_untranscribable_languages(
    model_languages: tuple[str, ...],
    rule_packs: Mapping[str, RulePack],
) -> tuple[str, ...]:
    """Name, at startup, every language this node has rules for but cannot hear.

    The mirror of ``_report_unmoderatable_languages`` above, and the half that
    was missing. That one compares the model's list against the installed packs
    and finds the languages a server can be transcribed in and never moderated.
    This one goes the other way and finds the packs whose language the speech
    model was never trained on.

    The two failures are not symmetric. A language with no pack is refused
    cleanly with ``language_unsupported`` at the front of the handler. A pack
    with no model got all the way to the recognizer, which raised, and the
    request came back ``500 internal_error`` with nothing written down anywhere
    -- a configuration mistake reported as a crash. That request is now refused
    with the same ``language_unsupported`` the other direction gets, and this
    line exists so nobody has to reach a request to find out.

    The shipped combination has none of these: 23 packs against a 25-language
    model. It appears the moment a smaller model is deployed against the full
    rules directory -- ``moonshine-*-en`` is English only, so on that node every
    pack but ``en`` lands here.

    Loud rather than fatal, for the same reasons as its sibling: refusing to
    start would crash-loop a live node, and an operator who deliberately runs an
    English model with the packs left in place is doing something legitimate.

    Returned as well as logged so a caller can assert on it.
    """
    missing = tuple(code for code in sorted(rule_packs) if code not in model_languages)
    if missing:
        LOGGER.error(
            "rules_without_a_model model_languages=%s installed_packs=%s missing=%s "
            "hint=requests pinned to these are refused with language_unsupported; "
            "either deploy a model trained on them or remove the rule pack",
            len(model_languages),
            len(rule_packs),
            ",".join(missing),
        )
    return missing


def _load_rule_packs(rules_dir: Path) -> dict[str, RulePack]:
    packs: dict[str, RulePack] = {}
    for path in sorted(rules_dir.glob("*.yml")):
        pack = RulePack.load(path)
        if pack.language in packs:
            raise ValueError(f"duplicate rule language: {pack.language}")
        packs[pack.language] = pack
    if not packs:
        raise ValueError("VOICESNIFFER_RULES_DIR must contain language packs")
    return packs


def _required_value(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _bounded_integer(
    values: Mapping[str, str],
    name: str,
    minimum: int,
    maximum: int,
    default: int | None = None,
) -> int:
    raw_value = values.get(name, str(minimum if default is None else default))
    try:
        value = int(raw_value)
    except ValueError as exception:
        raise ValueError(f"{name} must be an integer") from exception
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


if __name__ == "__main__":
    main()
