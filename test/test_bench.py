import pytest
from frag.bench import get_headers, get_request


def test_get_headers():
    headers_1 = get_headers(auth_token="1234")

    assert headers_1["Content-Type"] == "application/json"
    assert headers_1["Authorization"] == "Bearer 1234"

    headers_2 = get_headers(x_api_key="3456")
    assert headers_2["Content-Type"] == "application/json"
    with pytest.raises(KeyError):
        _ = headers_2[
            "Authorization"
        ]  # Raises KeyError because no headers are provided
    assert headers_2["x-api-key"] == "3456"


@pytest.mark.asyncio
async def test_get_request():
    input_requests = [("GET", 1, 2), ("POST", 3, 4), ("PUT", 5, 6)]
    request_rate = 2.0

    # TODO: This only tests the number of counts is the same. This does not
    # test the request rates...
    async def validate_requests():
        count = 0
        async for request in get_request(input_requests, request_rate):
            assert request in input_requests
            count += 1
        assert count == len(input_requests)

    await validate_requests()
