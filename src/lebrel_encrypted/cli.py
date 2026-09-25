"""Terminal JSON/SSE interface and local proxy command."""

import argparse
import json
import sys

from .client import Lebrel, LebrelError, MAX_REQUEST_BYTES, _json


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("proxy", "--proxy"):
        from .proxy import main as proxy_main
        return proxy_main(args[1:])
    parser = argparse.ArgumentParser(description="Lebrel encrypted inference: JSON from stdin, API key from LEBREL_API_KEY")
    parser.add_argument("command", choices=["infer"])
    parser.parse_args(args)
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Request exceeds its size limit")
        request = _json(raw)
        if not isinstance(request, dict):
            raise ValueError("A JSON completion request is required")
        with Lebrel() as client:
            result = client.create(**request)
            if request.get("stream", False):
                with result:
                    for event in result:
                        print("data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", flush=True)
                    if not result.completed:
                        raise LebrelError("Stream did not complete")
                    print("data: [DONE]\n", flush=True)
            else:
                print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        return 0
    except (LebrelError, ValueError, TypeError, UnicodeError):
        print('{"error":{"code":"encrypted_inference_failed","message":"The encrypted request did not complete"}}', file=sys.stderr)
        return 1
    except (BrokenPipeError, KeyboardInterrupt):
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
