#!/usr/bin/env python3
"""
fluxlate.main
CLI tool that translates .po files using local LLMs (via Ollama).
"""

from __future__ import annotations
import json
import pathlib
from typing import Any
import ollama
import polib
import typer
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL = "ministral-3:3b"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "llm_translator_agent" / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "llm_translator_agent" / "data" / "output"


def _structured_prompt(source: str, target_lang: str) -> str:
    """Build a prompt for structured JSON output."""
    return (
        f"Translate the following English string to {target_lang}. "
        "Preserve any placeholders such as %1, %2, %n, %{item}, %{fileName} exactly as they appear. "
        "Return ONLY a JSON object with a single key ``translation``.\n\n"
        f'{{"source": "{source}"}}'
    )


def translate_text(text: str, target_lang: str, verbose: bool = False) -> str:
    """Send text to Ollama and return the translation."""
    if not text.strip():
        return text

    prompt = _structured_prompt(text, target_lang)
    if verbose:
        typer.echo(f"  [DEBUG] Translating: {text!r}")

    try:
        result = ollama.generate(
            model=MODEL,
            prompt=prompt,
            format="json",
        )
        response_text = result.get("response", "")
        if verbose:
            typer.echo(f"  [DEBUG] Raw response: {response_text}")

        payload = json.loads(response_text)

        translation = None
        if isinstance(payload, dict):
            translation = payload.get("translation")
            if not isinstance(translation, str):

                def find_string(obj: Any) -> str | None:
                    if isinstance(obj, str):
                        return obj
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            res = find_string(v)
                            if res:
                                return res
                            if isinstance(k, str) and len(k) > 5:
                                return k
                    return None

                translation = find_string(payload)
        elif isinstance(payload, str):
            translation = payload

        if not isinstance(translation, str) or translation == response_text:
            translation = text

        if verbose:
            typer.echo(f"  [DEBUG] Translated: {translation!r}")

        return translation

    except Exception as exc:
        typer.secho(f"[WARN] Ollama failed for {text!r}: {exc}", fg=typer.colors.YELLOW)
        return text


def process_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    target_lang: str,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Read, translate, and write a .po file."""
    if not input_path.exists():
        typer.secho(f"[ERROR] Input file not found: {input_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    po = polib.pofile(str(input_path))
    typer.secho(
        f"🔎 Translating {input_path.name} to {target_lang} → {output_path.name}",
        fg=typer.colors.CYAN,
    )

    for entry in tqdm(po, unit="msg"):
        if entry.msgid_plural:
            for index in entry.msgstr_plural:
                if not force and entry.msgstr_plural[index].strip():
                    continue
                source = entry.msgid if index == 0 else entry.msgid_plural
                entry.msgstr_plural[index] = translate_text(
                    source, target_lang, verbose=verbose
                )
        else:
            if not force and entry.msgstr.strip():
                continue
            entry.msgstr = translate_text(entry.msgid, target_lang, verbose=verbose)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    po.save(str(output_path))
    typer.secho(f"✅ Saved translated file to {output_path}", fg=typer.colors.GREEN)


app = typer.Typer(help="Fluxlate: Translate .po files using local LLMs.")


@app.command()
def translate(
    path: str = typer.Argument(
        ..., help="Path to the .po file or filename in data/input."
    ),
    language: str = typer.Option("Hebrew", "--language", "-l", help="Target language."),
    output: str = typer.Option(None, "--output", "-o", help="Output filename."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing translations."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Translate a single .po file."""
    input_path = pathlib.Path(path).expanduser()
    if not input_path.exists():
        input_path = INPUT_DIR / path

    output_path = OUTPUT_DIR / (output or input_path.name)
    process_file(
        input_path, output_path, target_lang=language, force=force, verbose=verbose
    )


@app.command()
def list_inputs() -> None:
    """Show available .po files in the input folder."""
    files = sorted(p.name for p in INPUT_DIR.glob("*.po"))
    if not files:
        typer.secho("📂 No .po files found in data/input.", fg=typer.colors.YELLOW)
    else:
        typer.secho("📂 Available input files:", fg=typer.colors.CYAN)
        for f in files:
            typer.echo(f"  - {f}")


if __name__ == "__main__":
    app()
