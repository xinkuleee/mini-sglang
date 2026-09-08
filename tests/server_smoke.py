#!/usr/bin/env python3
"""真实 GGUF HTTP 回归；只用标准库，可同样验证 Rust 服务。

python tests/server_smoke.py --binary build/mini-sglang --model model.gguf
python tests/server_smoke.py --url http://127.0.0.1:1919 --chat
"""
import argparse
import concurrent.futures
import http.client
import json
import pathlib
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1919")
    parser.add_argument("--binary")
    parser.add_argument("--model")
    parser.add_argument("--chat", action="store_true", help="require a working GGUF chat template")
    parser.add_argument("--expect-no-chat", action="store_true")
    parser.add_argument("--require-cancel", action="store_true", help="require observed HTTP cancellation (use a nontrivial model)")
    args = parser.parse_args()
    process = None
    log = tempfile.TemporaryFile(mode="w+b")
    if args.binary:
        if not args.model:
            parser.error("--binary requires --model")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        args.url = f"http://127.0.0.1:{port}"
        process = subprocess.Popen([str(pathlib.Path(args.binary).resolve()), "--model",
            str(pathlib.Path(args.model).resolve()), "--serve", "--port", str(port),
            "--context-size", "4096", "--batch-tokens", "64",
            "--prefill-chunk", "16", "--max-sequences", "4", "--cache-sequences", "4"],
            stdout=log, stderr=log)

    def call(body=None, endpoint="/v1/completions"):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(args.url + endpoint, data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def stream(body, chat=False):
        body = dict(body, stream=True)
        request = urllib.request.Request(args.url + ("/v1/chat/completions" if chat else "/v1/completions"),
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        text, endings, done, usage = "", 0, 0, None
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.headers.get_content_type() == "text/event-stream"
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    done += 1
                    continue
                event = json.loads(payload)
                assert "error" not in event, event
                choice = event["choices"][0]
                text += choice.get("delta", {}).get("content", "") if chat else choice["text"]
                if choice.get("finish_reason") is not None:
                    endings += 1
                    assert choice["finish_reason"] in ("stop", "length")
                    usage = event["usage"]
        assert (done, endings) == (1, 1), (done, endings)
        return text, usage

    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                status, health = call(endpoint="/health")
                if status == 200:
                    break
            except (OSError, urllib.error.URLError):
                pass
            if process and process.poll() is not None:
                raise AssertionError("server exited during startup")
            if time.monotonic() > deadline:
                raise AssertionError("server did not become healthy")
            time.sleep(0.1)
        assert health["status"] == "ok", health

        nonce = time.time_ns()
        body = {"prompt": f"A small story about a friendly cloud {nonce}:",
                "max_tokens": 8, "temperature": 0, "seed": 123}
        status, cold = call(body)
        assert status == 200, cold
        assert cold["choices"][0]["finish_reason"] in ("stop", "length"), cold
        assert 0 < cold["usage"]["completion_tokens"] <= 8, cold
        assert cold["usage"]["prefill_tokens"] > 1, cold
        status, warm = call(body)
        assert status == 200, warm
        assert warm["choices"][0]["text"] == cold["choices"][0]["text"], (cold, warm)
        assert warm["usage"]["cached_tokens"] > 0, warm
        assert warm["usage"]["prefill_tokens"] < cold["usage"]["prefill_tokens"], (cold, warm)
        text, usage = stream(body)
        assert text == cold["choices"][0]["text"], (text, cold)
        assert usage["completion_tokens"] == cold["usage"]["completion_tokens"]

        before = call(endpoint="/health")[1]["stats"]["forward_calls"]
        status, zero = call({"prompt": "Zero token request.", "max_tokens": 0})
        assert status == 200 and zero["usage"]["completion_tokens"] == 0, zero
        assert zero["usage"]["prefill_tokens"] == 0 and zero["choices"][0]["text"] == "", zero
        after = call(endpoint="/health")[1]["stats"]["forward_calls"]
        assert before == after, (before, after)

        for invalid in ({"prompt": []}, {"prompt": ""}, {"prompt": "a", "max_tokens": -1},
                        {"prompt": "a", "max_tokens": 1.2}, {"prompt": "a", "temperature": -1},
                        {"prompt": "a", "top_p": 0}, {"prompt": "a", "stream": "yes"},
                        {"prompt": "a", "seed": -1}, {"prompt": "a", "unknown": True},
                        {"prompt": "a", "model": "not-the-loaded-model"},
                        {"prompt": "a", "max_tokens": 1000000}):
            status, error = call(invalid)
            assert status == 400 and "error" in error, (invalid, status, error)

        def parallel(index):
            status, result = call({"prompt": f"A cloud says hello {index}:", "max_tokens": 4})
            assert status == 200, result
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(parallel, range(6)))
        assert len({item["id"] for item in results}) == 6

        chat_body = {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8}
        if args.chat:
            status, chat = call(chat_body, "/v1/chat/completions")
            assert status == 200, chat
            text, _ = stream(chat_body, chat=True)
            assert text == chat["choices"][0]["message"]["content"], chat
        if args.expect_no_chat:
            status, error = call(chat_body, "/v1/chat/completions")
            assert status == 400 and "template" in error["error"]["message"].lower(), (status, error)

        # close an actual streaming socket before consuming its body; owner must release active KV.
        before_disconnect = call(endpoint="/health")[1]["stats"]
        url = urllib.parse.urlparse(args.url)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=60)
        connection.request("POST", "/v1/completions", json.dumps({
            "prompt": "A long story about a cloud travelling around the world:",
            "max_tokens": 1024, "stream": True}), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200, response.read()
        response.close()
        connection.close()
        deadline = time.monotonic() + 30
        while True:
            status, health = call(endpoint="/health")
            stats = health["stats"]
            if (stats["total_submitted"] >= before_disconnect["total_submitted"] + 1
                    and stats["active_requests"] == stats["queued_requests"] == stats["reserved_tokens"] == 0):
                break
            assert time.monotonic() < deadline, health
            time.sleep(0.05)
        assert status == 200, health
        observed_cancel = stats["total_cancelled"] > before_disconnect["total_cancelled"]
        if args.require_cancel:
            assert observed_cancel, ("request finished before cancellation could be observed", before_disconnect, stats)
        print(json.dumps({"status": "passed", "observed_cancel": observed_cancel, "cold_prefill_tokens": cold["usage"]["prefill_tokens"],
            "warm_prefill_tokens": warm["usage"]["prefill_tokens"],
            "cached_tokens": warm["usage"]["cached_tokens"], "final_stats": stats}))
    except Exception:
        if process:
            log.flush()
            log.seek(0)
            print(log.read().decode("utf-8", errors="replace"))
        raise
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log.close()


if __name__ == "__main__":
    main()
