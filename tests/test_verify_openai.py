# tests/test_verify_openai.py
from types import SimpleNamespace

import numpy as np

from musiclang.verify.whisper_id import transcribe_language
from musiclang.verify.llm_judge import judge_transcript


class _FakeTranscriptions:
    def __init__(self, resp):
        self._resp = resp
    def create(self, **kwargs):
        return self._resp


class _FakeWhisperClient:
    def __init__(self, language, text):
        self.audio = SimpleNamespace(
            transcriptions=_FakeTranscriptions(SimpleNamespace(language=language, text=text))
        )


def test_transcribe_language_lowercases_and_strips(tmp_path):
    client = _FakeWhisperClient("English", "  hello world  ")
    lang, text = transcribe_language(np.zeros(16_000, dtype=np.float32), client=client)
    assert lang == "english" and text == "hello world"


def test_transcribe_language_failsoft_on_error():
    class Boom:
        audio = SimpleNamespace(transcriptions=SimpleNamespace(
            create=lambda **k: (_ for _ in ()).throw(RuntimeError("api down"))))
    lang, text = transcribe_language(np.zeros(16_000, dtype=np.float32), client=Boom())
    assert lang == "" and text == ""


class _FakeParsed:
    def __init__(self, value):
        self.choices = [SimpleNamespace(message=SimpleNamespace(
            parsed=SimpleNamespace(is_fluent_target_language=value, reason="x")))]


class _FakeChatClient:
    def __init__(self, value):
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            parse=lambda **k: _FakeParsed(value)))


def test_judge_true_for_fluent():
    assert judge_transcript("una larga frase en español clara", "spanish",
                            client=_FakeChatClient(True)) is True


def test_judge_false_for_short_transcript_without_calling_api():
    # empty/short -> False, must not need a client
    assert judge_transcript("", "english", client=None) is False
    assert judge_transcript("hi", "english", client=None) is False


def test_judge_failclosed_on_api_error():
    class Boom:
        chat = SimpleNamespace(completions=SimpleNamespace(
            parse=lambda **k: (_ for _ in ()).throw(RuntimeError("api down"))))
    assert judge_transcript("a long enough transcript here", "english", client=Boom()) is False
