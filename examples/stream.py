from lebrel_encrypted import Lebrel

with Lebrel() as client:
    with client.chat.completions.create(
        messages=[{"role": "user", "content": "Explain how HPKE works."}],
        max_tokens=512,
        stream=True,
    ) as stream:
        for chunk in stream:
            for choice in chunk.get("choices", []):
                print(choice.get("delta", {}).get("content") or "", end="", flush=True)
        assert stream.completed
        print()
