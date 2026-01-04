# Fluxlate 🌀

**Fluxlate** is a powerful CLI tool designed to translate `.po` files using local Large Language Models (LLMs) via [Ollama](https://ollama.com/). It focuses on maintaining the structure of your localization files while providing high-quality translations for any target language.

## Features

- 🚀 **Local Translation**: No API keys required. Uses local models via Ollama.
- 📦 **Structured Output**: Forces JSON output from LLMs to ensure reliable extraction.
- 🔄 **Plural Support**: Correctly handles plural forms (`msgid_plural`).
- 🛠️ **Placeholder Preservation**: Keeps `%1`, `%{item}`, etc., intact.
- ⚡ **Differential Updates**: Skips already translated entries by default (use `--force` to overwrite).
- 🔍 **Verbose Debugging**: Step-by-step visibility into LLM responses.

## Installation

```bash
pip install fluxlate
```

Ensure you have [Ollama](https://ollama.com/) installed and the `ministral-3:3b` model (or your preferred model) pulled:

```bash
ollama pull ministral-3:3b
```

## Quick Start

### Translate a file

```bash
fluxlate translate path/to/messages.po --language Hebrew
```

### Options

- `-l, --language TEXT`: Specify the target language (default: Hebrew).
- `-o, --output TEXT`: Specify a custom output path.
- `-f, --force`: Overwrite existing translations.
- `-v, --verbose`: Show raw LLM input/output.

### List available inputs

If you have a project structure with `llm_translator_agent/data/input/`, you can list files:

```bash
fluxlate list-inputs
```

## Development

1. Clone the repository:
   ```bash
   git clone https://github.com/ereli/Fluxlate.git
   cd Fluxlate
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

3. Run tests:
   ```bash
   uv run pytest
   ```

## License

MIT
