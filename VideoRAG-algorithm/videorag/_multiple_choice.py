import json


async def request_multiple_choice_response(
    query,
    system_prompt,
    use_model_func,
    max_attempts=3,
    on_invalid_response=None,
):
    """Return the expected JSON response without retrying indefinitely."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        response = await use_model_func(
            query, system_prompt=system_prompt, use_cache=False
        )
        try:
            parsed_response = json.loads(response)
            if not isinstance(parsed_response, dict):
                raise ValueError("Response must be a JSON object")
            if (
                "Answer" not in parsed_response
                or "Explanation" not in parsed_response
            ):
                raise ValueError("Response must contain Answer and Explanation")
            return parsed_response
        except (TypeError, ValueError) as error:
            last_error = error
            if on_invalid_response is not None:
                on_invalid_response(error, attempt, max_attempts)

    raise ValueError(
        f"Model did not return valid multiple-choice JSON after {max_attempts} attempts"
    ) from last_error
