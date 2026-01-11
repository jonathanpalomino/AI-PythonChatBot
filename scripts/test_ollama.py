"""Test Ollama connection directly"""
from ollama import Client as OllamaClient

try:
    client = OllamaClient(host="http://localhost:11434")
    print("✅ Client created")

    models = client.list()
    print(f"✅ Models retrieved: {models}")
    print(f"Number of models: {len(models.get('models', []))}")

    for model in models.get("models", []):
        print(f"  - {model['name']}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
