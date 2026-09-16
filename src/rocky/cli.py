"""Small, explicit command line surface for the consolidated Rocky project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocky.analysis import dataset_analysis
from rocky.config import CONFIG
from rocky.data import export, validate, verify_export
from rocky.training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocky")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("check-data", validate), ("export-data", export), ("verify-data", verify_export)):
        command = commands.add_parser(name)
        command.set_defaults(action=function)

    analysis = commands.add_parser("analyze-data", help="measure corpus mix, lengths, facts and lexical overlap")
    analysis.add_argument("--output", type=Path)

    replay = commands.add_parser("replay", help="follow a supplied film scene exactly")
    replay.add_argument("--record", required=True)
    chat = commands.add_parser("chat", help="talk to a served model with explicit scene context")
    chat.add_argument("--scene", default="survivors")
    chat.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL ending in /v1")
    chat.add_argument("--model", required=True)
    evaluation = commands.add_parser("evaluate", help="record held-out inference results")
    evaluation.add_argument("--endpoint", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluation.add_argument("--mode", choices=("rollout", "continuation"), default="rollout")
    evaluation.add_argument("--suite", choices=("dialogues", "probes", "trajectories"), default="dialogues")
    for command in (chat, evaluation):
        command.add_argument("--context-tokens", type=int, default=4096)
        command.add_argument("--max-tokens", type=int, default=128)
        command.add_argument("--temperature", type=float, default=0.0)

    for name, objective in (("train-sft", "sft"), ("train-dpo", "dpo"), ("train-dpo-norm", "dpo-norm"), ("train-simpo", "simpo")):
        command = commands.add_parser(name)
        command.add_argument("--curriculum", choices=("core", "expanded"), default="core")
        command.add_argument("--config", type=Path, default=CONFIG)
        command.add_argument("--output", type=Path)
        command.add_argument("--sft-run", type=Path)
        command.add_argument("--resume", type=Path)
        command.add_argument("--hourly-usd", type=float, help="actual total rented GPU rate; required for execution")
        command.add_argument("--revision", help="40-character official Gemma 4 commit")
        command.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
        command.add_argument("--wandb-project", default="rocky-final")
        command.add_argument("--wandb-entity")
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--preflight", action="store_true")
        mode.add_argument("--execute", action="store_true")
        command.set_defaults(objective=objective)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze-data":
        result = dataset_analysis()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "chat":
        from rocky.inference import GenerationSettings, InferenceClient, load_tokenizer
        from rocky.session import SceneSession

        session = SceneSession(args.scene)
        client = InferenceClient(args.endpoint, args.model, load_tokenizer(),
                                 GenerationSettings(temperature=args.temperature, max_tokens=args.max_tokens,
                                                    context_tokens=args.context_tokens))
        print("Commands: /scene NAME resets history; /quit ends the session.")
        while True:
            try:
                grace = input("Grace: ").strip()
                if grace == "/quit":
                    return 0
                if grace.startswith("/scene "):
                    session.reset(grace.removeprefix("/scene ").strip())
                    print("Scene:", session.scene)
                    continue
                response = client.complete(session.prompt(grace))
                session.accept(grace, response)
                print("Rocky:", response)
            except (EOFError, KeyboardInterrupt):
                return 0
            except (ValueError, OSError) as error:
                print(error)
    if args.command == "replay":
        from rocky.session import FilmPlayback

        playback = FilmPlayback(args.record)
        if opening := playback.opening():
            print(opening)
        while playback.cursor < len(playback.messages):
            print("Grace's script:", playback.messages[playback.cursor]["content"])
            try:
                print(playback.reply(input("Grace: ")))
            except ValueError as error:
                print(error)
        return 0
    if args.command == "evaluate":
        from rocky.evaluation import evaluate
        from rocky.inference import GenerationSettings

        settings = GenerationSettings(temperature=args.temperature, max_tokens=args.max_tokens,
                                      context_tokens=args.context_tokens)
        print(json.dumps(evaluate(args.endpoint, args.model, args.output, args.split, args.mode,
                                  suite=args.suite, settings=settings), indent=2))
        return 0
    if hasattr(args, "action"):
        result = args.action()
    else:
        result = run_training(config_path=args.config, objective=args.objective, curriculum=args.curriculum,
                              output=args.output, preflight=args.preflight, execute=args.execute,
                              sft_run=args.sft_run, resume=args.resume, revision=args.revision,
                              hourly_usd=args.hourly_usd, wandb_mode=args.wandb_mode,
                              wandb_project=args.wandb_project, wandb_entity=args.wandb_entity)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
