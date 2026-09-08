#!/usr/bin/env python3
"""Black-box conformance against actual GGUF weights; no mock model is used.

Run one implementation: --binary /path/to/mini-sglang --model tiny.gguf
Compare both: --cpp /path/to/cpp --rust /path/to/rust --model tiny.gguf
"""
import argparse
import json
import subprocess
from pathlib import Path


def run(binary, model, prompts, *, chunk=8, active=4, cache=4, tokens=12):
    command = [str(binary), "--model", str(model), "--json", "--temperature", "0",
               "--max-tokens", str(tokens), "--context-size", "2048",
               "--batch-tokens", "64", "--prefill-chunk", str(chunk),
               "--max-sequences", str(active), "--cache-sequences", str(cache),
               "--threads", "2", "--gpu-layers", "0"]
    for prompt in prompts:
        command += ["--prompt", prompt]
    completed = subprocess.run(command, capture_output=True, encoding="utf-8", timeout=120)
    if completed.returncode:
        raise AssertionError(f"command failed ({completed.returncode}): {command}\n{completed.stderr[-6000:]}")
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(f"stdout must be only JSON: {completed.stdout[:1000]}") from error
    results = document["results"]
    assert len(results) == len(prompts), document
    for prompt, result in zip(prompts, results):
        assert result["prompt"] == prompt, "results must retain submission order"
        assert result["finish_reason"] in ("length", "stop", "eos"), result
        assert len(result["token_ids"]) <= tokens, result
        assert isinstance(result["text"], str), result
        assert result["usage"]["completion_tokens"] <= tokens, result
    return document


def sequences(document):
    return [item["token_ids"] for item in document["results"]]


def check(binary, model):
    prompts = ["Once upon a time, there was a little", "The small dog went to the",
               "Once upon a time, there was a brave"]
    standalone = [sequences(run(binary, model, [prompt], cache=0))[0] for prompt in prompts]
    batched = run(binary, model, prompts, cache=0)
    assert sequences(batched) == standalone, "batch composition changed greedy output"
    for chunk in (1, 7, 64):
        assert sequences(run(binary, model, prompts, chunk=chunk, cache=0)) == standalone, \
            f"chunk size {chunk} changed greedy output"
    repeated = run(binary, model, [prompts[0], prompts[0], prompts[2]], active=1)
    assert sequences(repeated) == [standalone[0], standalone[0], standalone[2]], \
        "prefix reuse changed greedy output"
    cold, hot, partial = repeated["results"]
    assert cold["usage"]["cached_tokens"] == 0, cold
    assert hot["usage"]["cached_tokens"] == hot["usage"]["prompt_tokens"] - 1, hot
    assert hot["usage"]["prefill_tokens"] == 1, hot
    assert 0 < partial["usage"]["cached_tokens"] < partial["usage"]["prompt_tokens"], partial
    # One retained entry forces eviction; coming back to a different prompt must recompute.
    evicted = run(binary, model, [prompts[0], prompts[1], prompts[0]], active=1, cache=1)
    assert sequences(evicted) == [standalone[0], standalone[1], standalone[0]], \
        "eviction corrupted live or reallocated KV"
    zero = run(binary, model, [prompts[0]], tokens=0)
    assert sequences(zero) == [[]], zero
    assert zero["results"][0]["usage"]["prefill_tokens"] == 0, zero
    invalid = subprocess.run([str(binary), "--model", str(model), "--prompt", "Hi",
                              "--temperature", "-1"], capture_output=True, timeout=30)
    assert invalid.returncode != 0, "negative temperature silently accepted"
    return {"binary": str(binary), "greedy_token_ids": standalone,
            "cached_prefill": [row["usage"] for row in repeated["results"]],
            "checks": ["standalone=batch", "chunks=1,7,64", "full prefix", "partial prefix",
                       "LRU eviction", "zero-token no forward", "invalid sampling rejected"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--rust", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binaries = [binary for binary in (args.binary, args.cpp, args.rust) if binary]
    if not binaries:
        parser.error("supply --binary or --cpp and --rust")
    reports = [check(binary.resolve(), args.model.resolve()) for binary in binaries]
    if len(reports) > 1:
        assert all(report["greedy_token_ids"] == reports[0]["greedy_token_ids"] for report in reports), \
            "C++ and Rust greedy sequences differ"
    summary = json.dumps({"status": "passed", "model": str(args.model.resolve()),
                          "implementations": reports}, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
