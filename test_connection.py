"""
test_connection.py — Quick smoke test for Azure OpenAI connectivity.

Run BEFORE writing build_index.py to verify:
  1. .env credentials are loaded correctly
  2. Embedding deployment responds with a 1536-dim vector
  3. Chat deployment generates a response

Fail-fast: catches config or network issues in 5 seconds, not 5 minutes.
"""
from openai import AzureOpenAI
import config


def main():
    # Show what we're about to test (key is partially redacted for safety)
    print("=" * 60)
    print("Azure OpenAI connection test")
    print("=" * 60)
    print(f"Endpoint:   {config.AZURE_ENDPOINT}")
    print(f"API key:    {config.AZURE_API_KEY[:6]}...{config.AZURE_API_KEY[-4:]}")
    print(f"API ver:    {config.AZURE_API_VERSION}")
    print(f"Embedding:  {config.EMBEDDING_DEPLOYMENT}")
    print(f"Chat:       {config.CHAT_DEPLOYMENT}")
    print()

    # Single client instance is reused for both calls
    client = AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )

    # --- Test 1: embedding endpoint ---
    print("[1/2] Testing embedding deployment...")
    try:
        response = client.embeddings.create(
            model=config.EMBEDDING_DEPLOYMENT,
            input="A heartwarming romantic comedy about second chances",
        )
        vector = response.data[0].embedding
        print(f"      OK - got {len(vector)}-dim vector")
        print(f"      First 4 values: {vector[:4]}")
        if len(vector) != config.EMBEDDING_DIM:
            print(f"      WARN: expected {config.EMBEDDING_DIM} dims, "
                  f"got {len(vector)}")
    except Exception as e:
        print(f"      FAIL - {type(e).__name__}: {e}")
        return

    print()

    # --- Test 2: chat endpoint ---
    print("[2/2] Testing chat deployment...")
    try:
        response = client.chat.completions.create(
            model=config.CHAT_DEPLOYMENT,
            messages=[
                {"role": "user", "content": "Reply with exactly: 'CineMatch online.'"},
            ],
        )
        reply = response.choices[0].message.content
        print(f"      OK - response: {reply!r}")
        usage = response.usage
        print(f"      Tokens: prompt={usage.prompt_tokens}, "
              f"completion={usage.completion_tokens}")
    except Exception as e:
        print(f"      FAIL - {type(e).__name__}: {e}")
        return

    print()
    print("All checks passed. Ready to build the index.")


if __name__ == "__main__":
    main()