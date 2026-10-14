"""Anthropic Messages API, spoken by the local proxy for clients such as Claude Code.

A Messages request becomes one OpenAI-format chat completion to the encrypted endpoint; the answer
comes back as a Messages response, streamed as Messages events when asked. Nothing here reaches
the network: the proxy runs these translations around the encrypted client.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "end_turn"}


class MessagesRequestError(ValueError):
    """A request this adapter cannot express as a chat completion; answered as invalid_request_error."""


def _text_of(content: Any, what: str) -> str:
    """Text from a string or a list of text blocks; anything else is refused by name."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                kind = block.get("type") if isinstance(block, dict) else type(block).__name__
                raise MessagesRequestError(f"{what} content of type {kind!r} is not supported")
        return "\n".join(parts)
    raise MessagesRequestError(f"{what} content must be text")


def _user_messages(content: Any) -> List[Dict[str, Any]]:
    """A user turn: tool results first (each its own tool message), then the user's own text."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        raise MessagesRequestError("message content must be a string or a list of blocks")
    tools: List[Dict[str, Any]] = []
    texts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise MessagesRequestError("message content blocks must be objects")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif kind == "tool_result" and isinstance(block.get("tool_use_id"), str):
            result = block.get("content", "")
            text = _text_of(result, "tool_result") if result is not None else ""
            if block.get("is_error") is True and text:
                text = "Error: " + text
            tools.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": text})
        else:
            raise MessagesRequestError(f"user content of type {kind!r} is not supported")
    messages = tools
    if texts:
        messages.append({"role": "user", "content": "\n".join(texts)})
    return messages


def _assistant_message(content: Any) -> Dict[str, Any]:
    """An assistant turn: its text and the tools it called; thinking blocks are the model's own and stay out."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise MessagesRequestError("message content must be a string or a list of blocks")
    texts: List[str] = []
    calls: List[Dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise MessagesRequestError("message content blocks must be objects")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif kind == "tool_use" and isinstance(block.get("id"), str) and isinstance(block.get("name"), str):
            arguments = block.get("input", {})
            calls.append({"id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": json.dumps(arguments if isinstance(arguments, dict) else {}, ensure_ascii=False)}})
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            raise MessagesRequestError(f"assistant content of type {kind!r} is not supported")
    message: Dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if calls:
        message["tool_calls"] = calls
    return message


def _tools(tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list):
        raise MessagesRequestError("tools must be a list")
    out: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise MessagesRequestError("each tool needs a name")
        if tool.get("type") not in (None, "custom"):
            raise MessagesRequestError(f"tool type {tool.get('type')!r} is not supported")
        function: Dict[str, Any] = {"name": tool["name"], "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}}}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        out.append({"type": "function", "function": function})
    return out


def _tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        raise MessagesRequestError("tool_choice must be an object")
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}
    raise MessagesRequestError("tool_choice is not supported")


def to_chat(request: Dict[str, Any], *, max_output_tokens: Optional[int] = None) -> Dict[str, Any]:
    """The chat completion that expresses a Messages request. `max_tokens` is capped at the edition's output budget."""
    if not isinstance(request, dict):
        raise MessagesRequestError("request must be an object")
    messages: List[Dict[str, Any]] = []
    system = request.get("system")
    if system is not None:
        text = _text_of(system, "system")
        if text:
            messages.append({"role": "system", "content": text})
    turns = request.get("messages")
    if not isinstance(turns, list) or not turns:
        raise MessagesRequestError("messages must be a non-empty list")
    for turn in turns:
        if not isinstance(turn, dict):
            raise MessagesRequestError("each message must be an object")
        role = turn.get("role")
        if role == "user":
            messages.extend(_user_messages(turn.get("content")))
        elif role == "assistant":
            messages.append(_assistant_message(turn.get("content")))
        elif role == "system":
            messages.append({"role": "system", "content": _text_of(turn.get("content"), "system")})
        else:
            raise MessagesRequestError(f"message role {role!r} is not supported")
    if not messages:
        raise MessagesRequestError("messages must be a non-empty list")
    chat: Dict[str, Any] = {"messages": messages, "stream": request.get("stream", False) is True}
    max_tokens = request.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise MessagesRequestError("max_tokens must be a positive integer")
    chat["max_tokens"] = min(max_tokens, max_output_tokens) if isinstance(max_output_tokens, int) and max_output_tokens > 0 else max_tokens
    for name in ("temperature", "top_p"):
        if name in request and request[name] is not None:
            chat[name] = request[name]
    stop = request.get("stop_sequences")
    if isinstance(stop, list) and stop:
        chat["stop"] = stop
    if request.get("tools"):
        chat["tools"] = _tools(request["tools"])
        choice = request.get("tool_choice")
        if choice is not None:
            chat["tool_choice"] = _tool_choice(choice)
            if choice.get("disable_parallel_tool_use") is True:
                chat["parallel_tool_calls"] = False
    if chat["stream"]:
        chat["stream_options"] = {"include_usage": True}
    return chat


def _usage(usage: Any) -> Dict[str, int]:
    usage = usage if isinstance(usage, dict) else {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _tool_input(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def message_id(request_id: Any) -> str:
    base = request_id if isinstance(request_id, str) and request_id else uuid.uuid4().hex
    return "msg_" + base.replace("-", "")


def from_chat(completion: Dict[str, Any], *, model: str, request_id: Any) -> Dict[str, Any]:
    """The Messages response for a finished chat completion."""
    choices = completion.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content: List[Dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            content.append({"type": "tool_use", "id": str(call.get("id") or "toolu_" + uuid.uuid4().hex[:24]), "name": str(function.get("name") or ""), "input": _tool_input(function.get("arguments"))})
    if not content:
        content.append({"type": "text", "text": ""})
    reason = choice.get("finish_reason")
    return {
        "id": message_id(request_id),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _STOP_REASONS.get(reason, "end_turn") if isinstance(reason, str) else "end_turn",
        "stop_sequence": None,
        "usage": _usage(completion.get("usage")),
    }


def stream_events(chunks: Iterator[Dict[str, Any]], *, model: str, request_id: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Messages events for a chat completion stream: text and tool calls become content blocks in order."""
    yield "message_start", {"type": "message_start", "message": {"id": message_id(request_id), "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}
    index = -1
    open_kind: Optional[str] = None  # "text" or "tool"
    tools: Dict[int, int] = {}  # tool call index in the chunk stream -> content block index
    pending_names: Dict[int, Dict[str, str]] = {}
    finish: Optional[str] = None
    usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def close_block() -> Iterator[Tuple[str, Dict[str, Any]]]:
        nonlocal open_kind
        if open_kind is not None:
            yield "content_block_stop", {"type": "content_block_stop", "index": index}
            open_kind = None

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = _usage(chunk["usage"])
        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        if choice is None:
            continue
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        text = delta.get("content")
        if isinstance(text, str) and text:
            if open_kind != "text":
                for event in close_block():
                    yield event
                index += 1
                open_kind = "text"
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}
            yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                position = call.get("index") if isinstance(call.get("index"), int) else len(tools)
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                if position not in tools:
                    for event in close_block():
                        yield event
                    index += 1
                    tools[position] = index
                    open_kind = "tool"
                    pending_names[position] = {"id": str(call.get("id") or "toolu_" + uuid.uuid4().hex[:24]), "name": str(function.get("name") or "")}
                    yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": pending_names[position]["id"], "name": pending_names[position]["name"], "input": {}}}
                elif tools[position] != index:
                    # Arguments for an earlier call after another block opened: reopen that call's block in place.
                    for event in close_block():
                        yield event
                    index = tools[position]
                    open_kind = "tool"
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    yield "content_block_delta", {"type": "content_block_delta", "index": tools[position], "delta": {"type": "input_json_delta", "partial_json": arguments}}
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            finish = reason
    for event in close_block():
        yield event
    if index < 0:
        # Nothing was said: one empty text block keeps the message well-formed.
        yield "content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": _STOP_REASONS.get(finish or "stop", "end_turn"), "stop_sequence": None}, "usage": {"output_tokens": usage.get("output_tokens", 0), "input_tokens": usage.get("input_tokens", 0)}}
    yield "message_stop", {"type": "message_stop"}


def count_tokens_estimate(request: Dict[str, Any]) -> int:
    """A size-based estimate of the prompt (about four characters a token): the edition's own count is in every answer's usage."""
    chat = to_chat(dict(request, max_tokens=1), max_output_tokens=None)
    size = len(json.dumps(chat.get("messages", []), ensure_ascii=False)) + len(json.dumps(chat.get("tools", []), ensure_ascii=False))
    return max(1, size // 4)


def error_body(status: int, code: str) -> Dict[str, Any]:
    kinds = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error", 411: "invalid_request_error", 413: "invalid_request_error", 415: "invalid_request_error", 429: "rate_limit_error", 503: "overloaded_error"}
    return {"type": "error", "error": {"type": kinds.get(status, "api_error"), "message": code.replace("_", " ")}}
