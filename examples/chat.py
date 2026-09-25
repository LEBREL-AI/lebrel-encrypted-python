from lebrel_encrypted import Lebrel

with Lebrel() as client:
    answer = client.chat.completions.create(
        messages=[{"role": "user", "content": "Write the opening of a story."}],
        max_tokens=256,
    )
    print(answer["choices"][0]["message"]["content"])
    # The signed receipt of this answer: which weights answered, bound to this request and this text.
    receipt = client.receipt(answer)
    print("receipt", receipt.request_id, "verified" if receipt.verified else "NOT VERIFIED")
