"""OpenAI Responses API, spoken by the local proxy for clients such as Codex.

A Responses request becomes one chat completion to the encrypted endpoint; the answer comes back
as a Response object, streamed as Responses events when asked. Conversation state is the
client's: every request carries its full input, as Codex sends it, and nothing is stored here.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple


class ResponsesRequestError(ValueError):
    """A request this adapter cannot express as a chat completion; answered as invalid_request_error."""


def _text_parts(content: Any, what: str) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in ("input_text", "output_text", "text") and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif kind == "refusal" and isinstance(part.get("refusal"), str):
                parts.append(part["refusal"])
            else:
                raise ResponsesRequestError(f"{what} content of type {kind!r} is not supported")
        return "\n".join(parts)
    raise ResponsesRequestError(f"{what} content must be text")


def _append_assistant(messages: List[Dict[str, Any]], *, text: Optional[str] = None, call: Optional[Dict[str, Any]] = None) -> None:
    """Assistant text and the calls that followed it form one assistant message, as chat completions expect."""
    last = messages[-1] if messages else None
    if last is None or last.get("role") != "assistant" or last.get("_closed"):
        last = {"role": "assistant", "content": ""}
        messages.append(last)
    if text:
        last["content"] = (last["content"] + "\n" + text) if last["content"] else text
    if call is not None:
        last.setdefault("tool_calls", []).append(call)


def to_chat(request: Dict[str, Any], *, max_output_tokens: Optional[int] = None) -> Dict[str, Any]:
    """The chat completion that expresses a Responses request. `max_output_tokens` is capped at the edition's output budget."""
    if not isinstance(request, dict):
        raise ResponsesRequestError("request must be an object")
    if request.get("previous_response_id"):
        raise ResponsesRequestError("previous_response_id is not supported: send the full conversation as input")
    messages: List[Dict[str, Any]] = []
    instructions = request.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    items = request.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise ResponsesRequestError("input must be a non-empty string or list")
    for item in items:
        if not isinstance(item, dict):
            raise ResponsesRequestError("each input item must be an object")
        kind = item.get("type") or "message"
        if kind == "message":
            role = item.get("role")
            if role in ("user", "system", "developer"):
                messages.append({"role": "system" if role == "developer" else role, "content": _text_parts(item.get("content"), role)})
            elif role == "assistant":
                _append_assistant(messages, text=_text_parts(item.get("content"), "assistant"))
                messages[-1]["_closed"] = False
            else:
                raise ResponsesRequestError(f"message role {role!r} is not supported")
        elif kind == "function_call":
            if not isinstance(item.get("call_id"), str) or not isinstance(item.get("name"), str):
                raise ResponsesRequestError("function_call items need call_id and name")
            arguments = item.get("arguments")
            _append_assistant(messages, call={"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}})
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str):
                raise ResponsesRequestError("function_call_output items need call_id")
            output = item.get("output")
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": output if isinstance(output, str) else _text_parts(output, "function_call_output")})
            if messages and len(messages) > 1 and messages[-2].get("role") == "assistant":
                messages[-2]["_closed"] = True
        elif kind == "reasoning":
            continue
        else:
            raise ResponsesRequestError(f"input item type {kind!r} is not supported")
    for message in messages:
        message.pop("_closed", None)
    chat: Dict[str, Any] = {"messages": messages, "stream": request.get("stream", False) is True}
    limit = request.get("max_output_tokens")
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ResponsesRequestError("max_output_tokens must be a positive integer")
        chat["max_tokens"] = min(limit, max_output_tokens) if isinstance(max_output_tokens, int) and max_output_tokens > 0 else limit
    elif isinstance(max_output_tokens, int) and max_output_tokens > 0:
        chat["max_tokens"] = max_output_tokens
    for name in ("temperature", "top_p"):
        if name in request and request[name] is not None:
            chat[name] = request[name]
    tools = request.get("tools")
    if isinstance(tools, list) and tools:
        functions: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise ResponsesRequestError("each tool must be an object")
            if tool.get("type") != "function":
                # Hosted tools (web_search, the computer tools) and tool namespaces are the platform's own;
                # the model keeps every function tool and simply does not have the others.
                continue
            if not isinstance(tool.get("name"), str):
                raise ResponsesRequestError("each function tool needs a name")
            function: Dict[str, Any] = {"name": tool["name"], "parameters": tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {"type": "object", "properties": {}}}
            if isinstance(tool.get("description"), str):
                function["description"] = tool["description"]
            functions.append({"type": "function", "function": function})
        if functions:
            chat["tools"] = functions
        choice = request.get("tool_choice")
        if not functions:
            pass
        elif isinstance(choice, str) and choice in ("auto", "none", "required"):
            chat["tool_choice"] = choice
        elif isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("name"), str):
            chat["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif choice is not None:
            raise ResponsesRequestError("tool_choice is not supported")
        if request.get("parallel_tool_calls") is False:
            chat["parallel_tool_calls"] = False
    if chat["stream"]:
        chat["stream_options"] = {"include_usage": True}
    return chat


def _usage(usage: Any) -> Dict[str, Any]:
    usage = usage if isinstance(usage, dict) else {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {"input_tokens": prompt, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": completion, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": prompt + completion}


def response_id(request_id: Any) -> str:
    base = request_id if isinstance(request_id, str) and request_id else uuid.uuid4().hex
    return "resp_" + base.replace("-", "")


def _skeleton(request: Dict[str, Any], *, model: str, request_id: Any, created_at: int) -> Dict[str, Any]:
    return {
        "id": response_id(request_id),
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "error": None,
        "incomplete_details": None,
        "instructions": request.get("instructions") if isinstance(request.get("instructions"), str) else None,
        "max_output_tokens": request.get("max_output_tokens") if isinstance(request.get("max_output_tokens"), int) else None,
        "model": model,
        "output": [],
        "parallel_tool_calls": request.get("parallel_tool_calls", True) is not False,
        "previous_response_id": None,
        "reasoning": None,
        "store": False,
        "temperature": request.get("temperature"),
        "text": {"format": {"type": "text"}},
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools") if isinstance(request.get("tools"), list) else [],
        "top_p": request.get("top_p"),
        "truncation": "disabled",
        "usage": None,
        "user": None,
        "metadata": request.get("metadata") if isinstance(request.get("metadata"), dict) else {},
    }


def _message_item(item_id: str, text: str, status: str = "completed") -> Dict[str, Any]:
    return {"id": item_id, "type": "message", "status": status, "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _call_item(item_id: str, call: Dict[str, Any], status: str = "completed") -> Dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments")
    return {"id": item_id, "type": "function_call", "status": status, "call_id": str(call.get("id") or "call_" + uuid.uuid4().hex[:24]), "name": str(function.get("name") or ""), "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}


def from_chat(completion: Dict[str, Any], *, request: Dict[str, Any], model: str, request_id: Any, created_at: Optional[int] = None) -> Dict[str, Any]:
    """The Response object for a finished chat completion."""
    response = _skeleton(request, model=model, request_id=request_id, created_at=created_at or int(time.time()))
    base = response["id"][5:]
    choices = completion.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    text = message.get("content")
    if isinstance(text, str) and text:
        response["output"].append(_message_item("msg_" + base, text))
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for position, call in enumerate(calls):
            if isinstance(call, dict):
                response["output"].append(_call_item(f"fc_{base}_{position}", call))
    if not response["output"]:
        response["output"].append(_message_item("msg_" + base, ""))
    reason = choice.get("finish_reason")
    if reason == "length":
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    else:
        response["status"] = "completed"
    response["usage"] = _usage(completion.get("usage"))
    return response


def stream_events(chunks: Iterator[Dict[str, Any]], *, request: Dict[str, Any], model: str, request_id: Any, created_at: Optional[int] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Responses events for a chat completion stream, ending in response.completed with the whole response."""
    response = _skeleton(request, model=model, request_id=request_id, created_at=created_at or int(time.time()))
    base = response["id"][5:]
    sequence = 0

    def event(name: str, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        nonlocal sequence
        sequence += 1
        return name, dict(payload, type=name, sequence_number=sequence)

    yield event("response.created", {"response": dict(response)})
    yield event("response.in_progress", {"response": dict(response)})
    output: List[Dict[str, Any]] = []
    text_index: Optional[int] = None  # output index of the open message item
    text_parts: List[str] = []
    calls: Dict[int, int] = {}  # tool call position in the chunk stream -> output index
    arguments: Dict[int, List[str]] = {}
    finish: Optional[str] = None
    usage = _usage(None)

    def close_text() -> Iterator[Tuple[str, Dict[str, Any]]]:
        nonlocal text_index
        if text_index is None:
            return
        text = "".join(text_parts)
        item = output[text_index]
        item["content"][0]["text"] = text
        item["status"] = "completed"
        yield event("response.output_text.done", {"item_id": item["id"], "output_index": text_index, "content_index": 0, "text": text, "logprobs": []})
        yield event("response.content_part.done", {"item_id": item["id"], "output_index": text_index, "content_index": 0, "part": item["content"][0]})
        yield event("response.output_item.done", {"output_index": text_index, "item": item})
        text_index = None

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
            if text_index is None:
                text_index = len(output)
                item = _message_item(f"msg_{base}_{text_index}", "", status="in_progress")
                output.append(item)
                text_parts.clear()
                yield event("response.output_item.added", {"output_index": text_index, "item": dict(item, content=[])})
                yield event("response.content_part.added", {"item_id": item["id"], "output_index": text_index, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
            text_parts.append(text)
            yield event("response.output_text.delta", {"item_id": output[text_index]["id"], "output_index": text_index, "content_index": 0, "delta": text, "logprobs": []})
        tool_deltas = delta.get("tool_calls")
        if isinstance(tool_deltas, list):
            for call in tool_deltas:
                if not isinstance(call, dict):
                    continue
                position = call.get("index") if isinstance(call.get("index"), int) else len(calls)
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                if position not in calls:
                    for pending in close_text():
                        yield pending
                    calls[position] = len(output)
                    arguments[position] = []
                    item = _call_item(f"fc_{base}_{position}", {"id": call.get("id"), "function": {"name": function.get("name"), "arguments": ""}}, status="in_progress")
                    output.append(item)
                    yield event("response.output_item.added", {"output_index": calls[position], "item": dict(item)})
                piece = function.get("arguments")
                if isinstance(piece, str) and piece:
                    arguments[position].append(piece)
                    yield event("response.function_call_arguments.delta", {"item_id": output[calls[position]]["id"], "output_index": calls[position], "delta": piece})
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            finish = reason
    for pending in close_text():
        yield pending
    for position, index in calls.items():
        item = output[index]
        item["arguments"] = "".join(arguments[position])
        item["status"] = "completed"
        yield event("response.function_call_arguments.done", {"item_id": item["id"], "output_index": index, "arguments": item["arguments"]})
        yield event("response.output_item.done", {"output_index": index, "item": item})
    if not output:
        item = _message_item(f"msg_{base}_0", "")
        output.append(item)
        yield event("response.output_item.added", {"output_index": 0, "item": dict(item, content=[])})
        yield event("response.output_item.done", {"output_index": 0, "item": item})
    response["output"] = output
    response["usage"] = usage
    if finish == "length":
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "max_output_tokens"}
        yield event("response.incomplete", {"response": response})
    else:
        response["status"] = "completed"
        yield event("response.completed", {"response": response})


def error_body(status: int, code: str) -> Dict[str, Any]:
    kinds = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error", 411: "invalid_request_error", 413: "invalid_request_error", 415: "invalid_request_error", 429: "rate_limit_error"}
    return {"error": {"type": kinds.get(status, "server_error"), "code": code, "message": code.replace("_", " "), "param": None}}
