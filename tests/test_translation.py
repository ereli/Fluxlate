import json
from fluxlate.main import translate_text, _structured_prompt


def test_structured_prompt():
    prompt = _structured_prompt("Hello", "Hebrew")
    assert "Translate the following English string to Hebrew" in prompt
    assert '"source": "Hello"' in prompt


def test_translate_text_simple(mocker):
    mock_generate = mocker.patch("ollama.generate")
    mock_generate.return_value = {"response": json.dumps({"translation": "שלום"})}

    result = translate_text("Hello", "Hebrew")
    assert result == "שלום"


def test_translate_text_nested_json(mocker):
    # Test our robust extraction logic with nested JSON
    mock_generate = mocker.patch("ollama.generate")
    mock_generate.return_value = {
        "response": json.dumps({"translation": {"source": "שלום world"}})
    }

    result = translate_text("Hello world", "Hebrew")
    assert result == "שלום world"


def test_translate_text_fallback_on_failure(mocker):
    mock_generate = mocker.patch("ollama.generate")
    mock_generate.side_effect = Exception("Ollama down")

    result = translate_text("Hello", "Hebrew")
    assert result == "Hello"


def test_translate_text_empty():
    result = translate_text("", "Hebrew")
    assert result == ""
