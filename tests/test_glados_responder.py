from src.agents.glados_responder_agent import GladosResponderOutputSchema, get_final_glados_response


def echo_tool_result(user_input):
    return GladosResponderOutputSchema(final_response=f"reply to {user_input.tool_result}")


async def test_same_query_and_tool_result_hits_cache(fake_llm):
    fake_llm.on(GladosResponderOutputSchema, echo_tool_result)

    first = await get_final_glados_response("Turn on the lights", "Turned on")
    second = await get_final_glados_response("Turn on the lights", "Turned on")

    assert first == second == "reply to Turned on"
    assert len(fake_llm.calls) == 1


async def test_new_tool_result_is_not_served_from_cache(fake_llm):
    fake_llm.on(GladosResponderOutputSchema, echo_tool_result)

    first = await get_final_glados_response("What is the temperature?", "It is 20 degrees")
    second = await get_final_glados_response("What is the temperature?", "It is 25 degrees")

    assert first == "reply to It is 20 degrees"
    assert second == "reply to It is 25 degrees"
    assert len(fake_llm.calls) == 2


async def test_without_tool_result(fake_llm):
    fake_llm.on(GladosResponderOutputSchema, echo_tool_result)

    assert await get_final_glados_response("Hello, GLaDOS.") == "reply to None"
    assert fake_llm.calls[0].user_input.tool_result is None


async def test_every_call_starts_with_empty_history(fake_llm):
    fake_llm.on(GladosResponderOutputSchema, echo_tool_result)

    await get_final_glados_response("one", "a")
    await get_final_glados_response("two", "b")

    assert [call.history_length for call in fake_llm.calls] == [0, 0]


async def test_history_is_passed_and_keys_the_cache(fake_llm):
    fake_llm.on(GladosResponderOutputSchema, echo_tool_result)

    await get_final_glados_response("And now?", "ok", "User: Hi\nGLaDOS: Oh. It's you.")
    await get_final_glados_response("And now?", "ok")

    assert fake_llm.calls[0].user_input.conversation_history == "User: Hi\nGLaDOS: Oh. It's you."
    assert len(fake_llm.calls) == 2
