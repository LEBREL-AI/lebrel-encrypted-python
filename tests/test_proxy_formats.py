"""The proxy speaks three request formats over one encrypted chat completion: OpenAI chat completions
(unchanged), the OpenAI Responses API (Codex) and the Anthropic Messages API (Claude Code). The
mock client below answers like the gateway, with tool calls and usage, so each translation is
checked both ways, streamed and not."""

import json
import threading

import httpx
import pytest

from lebrel_encrypted import Completion, MODEL_ID
from lebrel_encrypted.proxy import ProxyServer


class ToolStream:
    """Text, then one tool call spread over chunks, then usage: what the gateway streams for a tool turn."""

    def __init__(self, request):
        self.completed = False
        self.request_id = "req-stream-000002"
        self.closed = threading.Event()
        self.request = request

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        yield {"choices": [{"index": 0, "delta": {"content": "Let me "}, "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {"content": "look."}, "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_abc", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": "}}]}, "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"a.txt\"}"}}]}, "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}}
        self.completed = True

    def close(self):
        self.closed.set()


class ToolClient:
    instances = []

    def __init__(self, api_key):
        self.api_key = api_key
        self.request = None
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def create(self, **request):
        self.request = request
        if request.get("stream"):
            return ToolStream(request)
        completion = Completion({
            "id": "chatcmpl-1", "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Reading it.", "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49},
        })
        completion.request_id = "req-plain-0000002"
        completion.receipt_header = "eyJwYXlsb2FkIjp7fSwic2lnbmF0dXJlIjoiQUE9PSJ9"
        return completion

    def list_models(self):
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

    def close(self):
        pass


@pytest.fixture
def proxy():
    ToolClient.instances = []
    server = ProxyServer(0, client_factory=ToolClient, max_output_tokens=4096, catalog_url=None)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def events(text):
    """(event name, data) pairs of an SSE body."""
    out = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        out.append((name, json.loads(data) if data and data != "[DONE]" else data))
    return out


ANTHROPIC_REQUEST = {
    "model": "claude-sonnet-4-5", "max_tokens": 32000, "system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Read a.txt"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Sure."}, {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "hello"}]}, {"type": "text", "text": "Now summarize."}]},
    ],
    "tools": [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
    "tool_choice": {"type": "auto"}, "metadata": {"user_id": "x"}, "thinking": {"type": "adaptive"},
}


def test_messages_request_becomes_one_chat_completion_with_the_output_budget_capped(proxy):
    answer = httpx.post(proxy + "/v1/messages?beta=true", json=ANTHROPIC_REQUEST, headers={"x-api-key": "key-1", "anthropic-version": "2023-06-01"})
    assert answer.status_code == 200, answer.text
    sent = ToolClient.instances[-1].request
    assert ToolClient.instances[-1].api_key == "key-1", "Anthropic's x-api-key header carries the key"
    assert "model" not in sent, "the edition is the proxy's one model"
    assert sent["max_tokens"] == 4096, "Claude Code asks for 32k; the edition's budget caps it"
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Read a.txt"},
        {"role": "assistant", "content": "Sure.", "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "hello"},
        {"role": "user", "content": "Now summarize."},
    ]
    assert sent["tools"] == [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}, "description": "Read a file"}}]
    assert sent["tool_choice"] == "auto"
    body = answer.json()
    assert body["type"] == "message" and body["role"] == "assistant" and body["model"] == "claude-sonnet-4-5"
    assert body["id"] == "msg_reqplain0000002"
    assert body["content"] == [{"type": "text", "text": "Reading it."}, {"type": "tool_use", "id": "call_abc", "name": "read_file", "input": {"path": "a.txt"}}]
    assert body["stop_reason"] == "tool_use" and body["usage"]["input_tokens"] == 40 and body["usage"]["output_tokens"] == 9
    assert answer.headers["X-Lebrel-Request-Id"] == "req-plain-0000002" and answer.headers["Proof-Of-Edition-Receipt"]


def test_messages_stream_is_relayed_as_messages_events(proxy):
    with httpx.stream("POST", proxy + "/v1/messages", json=dict(ANTHROPIC_REQUEST, stream=True), headers={"Authorization": "Bearer key-2"}) as answer:
        assert answer.status_code == 200 and answer.headers["content-type"].startswith("text/event-stream")
        assert answer.headers["X-Lebrel-Request-Id"] == "req-stream-000002"
        got = events(answer.read().decode())
    names = [name for name, _ in got]
    assert names == ["message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert got[1][1]["content_block"] == {"type": "text", "text": ""}
    assert "".join(data["delta"]["text"] for name, data in got[2:4]) == "Let me look."
    assert got[5][1]["content_block"] == {"type": "tool_use", "id": "call_abc", "name": "read_file", "input": {}}
    assert "".join(data["delta"]["partial_json"] for name, data in got[6:8]) == "{\"path\": \"a.txt\"}"
    assert got[9][1]["delta"]["stop_reason"] == "tool_use" and got[9][1]["usage"] == {"output_tokens": 9, "input_tokens": 40}
    assert ToolClient.instances[-1].request["stream_options"] == {"include_usage": True}


def test_messages_errors_use_the_anthropic_shape_and_count_tokens_estimates(proxy):
    refused = httpx.post(proxy + "/v1/messages", json={"model": "x", "max_tokens": 10, "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}, headers={"x-api-key": "k"})
    assert refused.status_code == 400 and refused.json() == {"type": "error", "error": {"type": "invalid_request_error", "message": "invalid request"}}
    unauthorized = httpx.post(proxy + "/v1/messages", json=ANTHROPIC_REQUEST)
    assert unauthorized.status_code == 401 and unauthorized.json()["error"]["type"] == "authentication_error"
    counted = httpx.post(proxy + "/v1/messages/count_tokens", json=ANTHROPIC_REQUEST, headers={"x-api-key": "k"})
    assert counted.status_code == 200 and counted.json()["input_tokens"] > 20


RESPONSES_REQUEST = {
    "model": "gpt-5-codex", "instructions": "You are Codex.", "stream": False, "store": False, "max_output_tokens": 100000,
    "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Read a.txt"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Sure."}]},
        {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\": \"a.txt\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "hello"},
        {"type": "message", "role": "user", "content": "Now summarize."},
    ],
    "tools": [{"type": "function", "name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}, "strict": False}],
    "tool_choice": "auto", "parallel_tool_calls": True, "reasoning": {"effort": "medium"},
}


def test_responses_request_becomes_one_chat_completion_and_a_response_object(proxy):
    answer = httpx.post(proxy + "/v1/responses", json=RESPONSES_REQUEST, headers={"Authorization": "Bearer key-3"})
    assert answer.status_code == 200, answer.text
    sent = ToolClient.instances[-1].request
    assert sent["max_tokens"] == 4096 and sent["stream"] is False
    assert sent["messages"] == [
        {"role": "system", "content": "You are Codex."},
        {"role": "user", "content": "Read a.txt"},
        {"role": "assistant", "content": "Sure.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
        {"role": "user", "content": "Now summarize."},
    ]
    assert sent["tools"][0]["function"]["name"] == "read_file" and sent["tool_choice"] == "auto"
    body = answer.json()
    assert body["object"] == "response" and body["status"] == "completed" and body["model"] == "gpt-5-codex" and body["id"] == "resp_reqplain0000002"
    assert body["output"][0] == {"id": "msg_reqplain0000002", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "Reading it.", "annotations": []}]}
    assert body["output"][1] == {"id": "fc_reqplain0000002_0", "type": "function_call", "status": "completed", "call_id": "call_abc", "name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}
    assert body["usage"] == {"input_tokens": 40, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 9, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 49}


def test_responses_stream_ends_in_response_completed_with_the_whole_output(proxy):
    with httpx.stream("POST", proxy + "/v1/responses", json=dict(RESPONSES_REQUEST, stream=True), headers={"Authorization": "Bearer key-4"}) as answer:
        assert answer.status_code == 200
        got = events(answer.read().decode())
    names = [name for name, _ in got]
    assert names[:2] == ["response.created", "response.in_progress"]
    assert names[2:6] == ["response.output_item.added", "response.content_part.added", "response.output_text.delta", "response.output_text.delta"]
    assert "response.function_call_arguments.delta" in names and names[-1] == "response.completed"
    assert [data["sequence_number"] for _, data in got] == list(range(1, len(got) + 1))
    final = got[-1][1]["response"]
    assert final["status"] == "completed" and final["usage"]["total_tokens"] == 49
    assert [item["type"] for item in final["output"]] == ["message", "function_call"]
    assert final["output"][0]["content"][0]["text"] == "Let me look."
    assert final["output"][1]["arguments"] == "{\"path\": \"a.txt\"}" and final["output"][1]["call_id"] == "call_abc"


def test_responses_keeps_function_tools_and_leaves_hosted_and_namespaced_ones_out(proxy):
    # What Codex sends: its function tools, a web_search hosted tool and a namespace of sub-agent tools.
    tools = RESPONSES_REQUEST["tools"] + [{"type": "web_search", "external_web_access": False}, {"type": "namespace", "name": "multi_agent_v1", "description": "Sub-agents", "tools": [{"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}}]}]
    answer = httpx.post(proxy + "/v1/responses", json=dict(RESPONSES_REQUEST, tools=tools), headers={"Authorization": "Bearer k"})
    assert answer.status_code == 200, answer.text
    assert [tool["function"]["name"] for tool in ToolClient.instances[-1].request["tools"]] == ["read_file"]


def test_responses_refuses_what_it_cannot_express_in_its_own_error_shape(proxy):
    stateful = httpx.post(proxy + "/v1/responses", json=dict(RESPONSES_REQUEST, previous_response_id="resp_x"), headers={"Authorization": "Bearer k"})
    assert stateful.status_code == 400 and stateful.json()["error"]["type"] == "invalid_request_error"
    unknown = httpx.post(proxy + "/v1/unknown", json={}, headers={"Authorization": "Bearer k"})
    assert unknown.status_code == 404


def test_chat_completions_still_pass_through_untouched(proxy):
    answer = httpx.post(proxy + "/v1/chat/completions", json={"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50000}, headers={"Authorization": "Bearer k"})
    assert answer.status_code == 200 and answer.json()["choices"][0]["message"]["content"] == "Reading it."
    assert ToolClient.instances[-1].request["max_tokens"] == 50000, "the OpenAI route is the gateway's own contract; nothing is rewritten"


def test_output_limit_comes_from_the_public_catalog_once(monkeypatch):
    """The edition's output budget is read from the public catalog on the first request, kept for an hour, and
    a client that asks for more is capped rather than refused."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    reads = []

    class Catalog(BaseHTTPRequestHandler):
        def do_GET(self):
            reads.append(self.path)
            data = json.dumps({"editions": [{"modelId": MODEL_ID, "context": {"servedMaxTokens": 1048544, "servedOutputTokens": 8192}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    catalog = HTTPServer(("127.0.0.1", 0), Catalog)
    threading.Thread(target=catalog.serve_forever, daemon=True).start()
    server = ProxyServer(0, client_factory=ToolClient, catalog_url="http://127.0.0.1:%d/api/models" % catalog.server_port)
    try:
        assert server.output_limit() == 8192
        assert server.output_limit() == 8192
        assert reads == ["/api/models"], "one read serves every request of the hour"
        no_catalog = ProxyServer(0, client_factory=ToolClient, catalog_url=None)
        assert no_catalog.output_limit() is None
        no_catalog.server_close()
    finally:
        server.server_close()
        catalog.shutdown()
        catalog.server_close()
