# Benchmarking online serving throughput for LLM API endpoints.
import argparse
import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()


DEFAULT_PROMPT = "You are a helpful assistant that respeonds with the answer in the most concise possible way."
DEFAULT_NUM_REQUESTS = 5

REQUEST_LATENCY: list[tuple[int, int, float]] = []  # (prompt len, output len, latency)


# @dataclasses.dataclass
# class ApiContext:
#     session: aiohttp.ClientSession
#     index: int
#     model: str
#     prompt: str


# @dataclasses.dataclass
# class ApiResult:
#     def __init__(self, index, start_time, response, chunk_gen):
#         self.index = index
#         self.start_time = start_time
#         self.latency = time.time() - start_time
#         self.response = response
#         self.chunk_gen = chunk_gen

#     index: int
#     start_time: int
#     latency: float  # HTTP response time
#     response: aiohttp.ClientResponse
#     chunk_gen: Generator[str, None, None]


# async def post(
#     context: ApiContext,
#     url: str,
#     headers: dict,
#     data: dict,
#     make_chunk_gen: callable(aiohttp.ClientResponse) = None,
# ):
#     start_time = time.time()
#     response = await context.session.post(url, headers=headers, data=json.dumps(data))
#     chunk_gen = make_chunk_gen(response) if make_chunk_gen else None
#     return ApiResult(context.index, start_time, response, chunk_gen)


def get_headers(*, auth_token: str | None = None, x_api_key: str | None = None) -> dict:
    headers = {
        "Content-Type": "application/json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if x_api_key:
        headers["x-api-key"] = x_api_key
    return headers


async def get_request(
    input_requests: list[tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[tuple[str, int, int], None]:
    """Forms requests into a Poisson process using the provided request_rate by
    adding an async sleep timer. The request_rate is measured as requests per
    second.

    Args:
        input_requests (list[tuple[str, int, int]]): _description_
        request_rate (float): intended request rate, measured in seconds.

    Returns:
        AsyncGenerator[tuple[str, int, int], None]: _description_

    Yields:
        Iterator[AsyncGenerator[tuple[str, int, int], None]]: _description_
    """
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    api_url: str,
    headers: dict,
    prompt: str,
    prompt_len: int,
    output_len: int,
) -> None:
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": f"{prompt}"}],
        "temperature": 0.0,
        "stream": True,
    }
    timeout = httpx.Timeout(3 * 3600)
    request_start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            response = await client.post(api_url, headers=headers, json=payload)

            chunks = []
            output_len = 0
            time_to_first_token = None

            # Stream the response
            async for line in response.aiter_lines():
                if not line and not time_to_first_token:
                    time_to_first_token = time.perf_counter() - request_start_time

                line = line.strip()
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    data = json.loads(data)
                    if data["choices"]:
                        output_len += 1
                        content = data["choices"][0]["delta"].get("content")
                        if content:
                            chunks.append(content)

            output = "".join(chunks)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    # print(REQUEST_LATENCY, time_to_first_token, output_len)
    print(f"time to first token: {time_to_first_token * 1000:.2f} ms")


async def benchmark(
    api_url: str,
    model: str,
    headers: dict,
    input_requests: list[tuple[str, int, int]],
    request_rate: float,
) -> None:
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(api_url, headers, prompt, prompt_len, output_len)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    headers = get_headers(auth_token=os.getenv("OPENAI_API_KEY"))

    # tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    # input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    num_prompts = args.num_prompts
    # requests: prompt, input_len, max_len
    input_requests = [(DEFAULT_PROMPT, 3, 4000)] * num_prompts

    request_rate = args.request_rate

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(args.api_url, args.model, headers, input_requests, request_rate)
    )
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {num_prompts / benchmark_time:.2f} requests/s")
    print(f"Throughput: {num_prompts / benchmark_time * 60:.2f} RPM")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency*1000:.2f} ms")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print(
        "Average latency per output token: "
        f"{avg_per_output_token_latency*1000:.2f} ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    # For local testing, localhost:8000
    parser.add_argument(
        "--api-url",
        "-a",
        type=str,
        default="https://api.openai.com/v1/chat/completions",
        help="Base URL for the LLM API endpoint",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gpt-3.5-turbo", help="Model to benchmark"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=5, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=20 / 60,  # float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are"
        " sent at time 0. Otherwise, we use Poisson process to synthesize the request "
        "arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-requests",
        "-n",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help="Number of requests to make",
    )
    # parser.add_argument(
    #     "--max-tokens",
    #     type=int,
    #     default=DEFAULT_MAX_TOKENS,
    #     help="Max tokens for the response",
    # )
    # # parser.add_argument(
    # #     "--dataset", type=str, required=True, help="Path to the dataset."
    # # )
    # parser.add_argument(
    #     "--trust-remote-code",
    #     action="store_true",
    #     help="trust remote code from huggingface",
    # )
    # parser.add_argument(
    #     "prompt",
    #     type=str,
    #     nargs="?",
    #     default=DEFAULT_PROMPT,
    #     help="Prompt to send to the API",
    # )
    # parser.add_argument(
    #     "--no-warmup",
    #     action="store_false",
    #     dest="warmup",
    #     help="Don't do a warmup call to the API",
    # )

    # parser.add_argument(
    #     "--print",
    #     "-p",
    #     action="store_true",
    #     dest="print",
    #     help="Print the response",
    # )
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument(
    #     "--verbose",
    #     "-v",
    #     action="store_true",
    #     dest="verbose",
    #     help="Print verbose output",
    # )
    # group.add_argument(
    #     "--minimal",
    #     action="store_true",
    #     dest="minimal",
    #     help="Print minimal output",
    # )
    args = parser.parse_args()
    main(args)
