import json
import time
import urllib.error
import urllib.request

BASE_URL = "http://localhost:8000/api/v1"


def make_request(url, method="GET", data=None):
    headers = {'Content-Type': 'application/json'}
    if data:
        data = json.dumps(data).encode('utf-8')

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.getcode(), json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))
    except Exception as e:
        return 500, str(e)


def create_conversation():
    print("Creating conversation...")
    status, data = make_request(f"{BASE_URL}/conversations", "POST", {
        "title": "Perf Test",
        "settings": {
            "provider": "local",
            "model": "qwen3:1.7b",
            "temperature": 0.7,
            "tool_mode": "manual",
            "enabled_tools": [],
            "max_tokens": 2000,
            "memory_enabled": False
        }
    })

    if status != 201:
        print(f"Error creating conversation: {data}")
        return None
    return data["id"]


def send_message(conversation_id, message):
    print(f"\nSending message: '{message}'")
    start_time = time.time()
    status, data = make_request(f"{BASE_URL}/conversations/{conversation_id}/chat", "POST", {
        "message": message
    })
    end_time = time.time()
    duration = end_time - start_time

    if status == 200:
        print(f"Response: {data['message']['content'][:50]}...")
        print(f"Time: {duration:.2f} seconds")
        print(f"Tools executed: {data.get('tools_executed', [])}")
    else:
        print(f"Error: {data}")
        print(f"Time: {duration:.2f} seconds")


def main():
    try:
        conv_id = create_conversation()
        if not conv_id:
            return

        # Test 1: First message
        send_message(conv_id, "hola")

        # Test 2: Second message
        send_message(conv_id, "como estas")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()
