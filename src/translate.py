#!/usr/bin/env python3
"""
translate.py
CLI tool that reads a .po file from data/input, translates each entry to Hebrew
using a local Ollama model (ministral‑3:3b), and writes the translated file
to data/output.
The Ollama call uses **structured output** (JSON) so we can reliably extract
the translation even if the model adds extra text.
"""

from __future__ import annotations
import json
import pathlib
import ollama
import polib
import typer
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL = "ministral-3:3b"  # change if you prefer another local model
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "llm_translator_agent" / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "llm_translator_agent" / "data" / "output"


def _structured_prompt(source: str, target_lang: str) -> str:
    """
    Build a prompt that asks the model to return a JSON object with a single
    field called ``translation``.
    """
    return (
        f"Translate the following English string to {target_lang}. "
        "Preserve any placeholders such as %1, %2, %n, %{item}, %{fileName} exactly as they appear. "
        "Return ONLY a JSON object with a single key ``translation``.\n\n"
        f'{{"source": "{source}"}}'
    )


def translate_text(text: str, target_lang: str, verbose: bool = False) -> str:
    """
    Send text to Ollama and return the translation.
    Ensures the result is a string.
    """
    if not text.strip():
        return text

    prompt = _structured_prompt(text, target_lang)
    if verbose:
        typer.echo(f"  [DEBUG] Translating: {text!r}")

    try:
        # Ollama structured output – we ask for JSON.
        result = ollama.generate(
            model=MODEL,
            prompt=prompt,
            format="json",
        )
        response_text = result.get("response", "")
        if verbose:
            typer.echo(f"  [DEBUG] Raw response: {response_text}")

        payload = json.loads(response_text)

        # Handle various LLM output quirks
        if isinstance(payload, dict):
            translation = payload.get("translation")
            if translation is None:
                # LLM might have used a different key or returned the whole object
                if "source" in payload and len(payload) == 1:
                    # Some models might swap keys if confused
                    translation = payload["source"]
                else:
                    # Fallback to choosing the first string value found
                    for val in payload.values():
                        if isinstance(val, str):
                            translation = val
                            break

            if isinstance(translation, dict):
                # If it's still a dict, try to get a string from it
                for val in translation.values():
                    if isinstance(val, str):
                        translation = val
                        break

        elif isinstance(payload, str):
            translation = payload
        else:
            translation = str(payload)

        # Final safety check: ensure we return a string
        if not isinstance(translation, str):
            translation = text

        if verbose:
            typer.echo(f"  [DEBUG] Translated: {translation!r}")

        return translation

    except Exception as exc:
        typer.secho(
            f"[WARN] Ollama failed for {text!r}: {exc}",
            fg=typer.colors.YELLOW,
        )
        return text


def process_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    target_lang: str,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Read, translate, and write a .po file."""
    po = polib.pofile(str(input_path))
    typer.secho(
        f"🔎 Translating {input_path.name} to {target_lang} → {output_path.name}",
        fg=typer.colors.CYAN,
    )

    for entry in tqdm(po, unit="msg"):
        # Handle Plural Entries
        if entry.msgid_plural:
            # entry.msgstr_plural is a dict: {0: 'trans0', 1: 'trans1', ...}
            # We need to translate both msgid and msgid_plural or just fill the slots
            for index in entry.msgstr_plural:
                if not force and entry.msgstr_plural[index].strip():
                    continue
                # For plural forms, we usually translate the plural form specifically
                # but models often need context. We'll use msgid for index 0 and msgid_plural for others
                source = entry.msgid if index == 0 else entry.msgid_plural
                entry.msgstr_plural[index] = translate_text(
                    source, target_lang, verbose=verbose
                )
        else:
            # Singular Entry
            if not force and entry.msgstr.strip():
                continue
            entry.msgstr = translate_text(entry.msgid, target_lang, verbose=verbose)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    po.save(str(output_path))
    typer.secho(f"✅ Saved translated file to {output_path}", fg=typer.colors.GREEN)


# ----------------------------------------------------------------------
# Typer CLI
# ----------------------------------------------------------------------
app = typer.Typer(help="Translate .po files using a local Ollama model.")


@app.command()
def translate(
    path: str = typer.Argument(
        ...,
        help="Path to the .po file to translate. Can be a filename in data/input or a direct path.",
    ),
    language: str = typer.Option(
        "Hebrew",
        "--language",
        "-l",
        help="The target language for translation.",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output filename (defaults to same name in data/output).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing translations in the .po file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output for debugging.",
    ),
) -> None:
    """
    Translate a single .po file.
    """
    # Resolve the input path:
    input_path = pathlib.Path(path).expanduser()
    if not input_path.exists():
        input_path = INPUT_DIR / path

    if not input_path.is_file():
        typer.secho(f"[ERROR] Input file not found: {input_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    output_path = OUTPUT_DIR / (output or input_path.name)
    process_file(
        input_path, output_path, target_lang=language, force=force, verbose=verbose
    )


@app.command()
def list_inputs() -> None:
    """Show all .po files available in the input folder."""
    files = sorted(p.name for p in INPUT_DIR.glob("*.po"))
    if not files:
        typer.secho("📂 No .po files found in data/input.", fg=typer.colors.YELLOW)
    else:
        typer.secho("📂 Available input files:", fg=typer.colors.CYAN)
        for f in files:
            typer.echo(f"  - {f}")


if __name__ == "__main__":
    app()
