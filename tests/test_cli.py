import io
import json
from types import SimpleNamespace

from lebrel_encrypted import cli


class Client:
    last_request = None
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def create(self, **request):
        Client.last_request = request
        return {"choices": [{"message": {"content": "terminal-answer"}, "finish_reason": "stop"}]}


def test_terminal_json_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr(cli, "Lebrel", Client)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'{"messages":[{"role":"user","content":"private-prompt"}]}')))
    assert cli.main(["infer"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["choices"][0]["message"]["content"] == "terminal-answer"
    assert output.err == ""
    assert Client.last_request["messages"][0]["content"] == "private-prompt"


def test_terminal_errors_do_not_echo_input(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'private-invalid-prompt')))
    assert cli.main(["infer"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "private-invalid-prompt" not in output.err
    assert "encrypted_inference_failed" in output.err
