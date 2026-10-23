import httpx

from src.agents.orchestrator_agent import OrchestratorOutputSchema
from src.services.conversation_store import conversations
from tests.test_api import api, pipeline, route_by_keyword  # noqa: F401 (fixtures)
from tests.test_speech_to_text import fake_openrouter_stt, make_wav
from tests.test_text_to_speech import AUDIO_URL, TTS_PATH, fake_speaker


#######################
# CONVERSATION MEMORY #
#######################
async def test_without_conversation_id_nothing_is_remembered(api, pipeline):
    fake_llm, _ = pipeline

    await api.post("/api/start", json={"message": "Hello GLaDOS"})
    await api.post("/api/start", json={"message": "Hello again"})

    assert [call.user_input.conversation_history for call in fake_llm.calls] == [None] * 4


async def test_conversation_history_reaches_orchestrator_and_responder(api, pipeline):
    fake_llm, _ = pipeline

    await api.post("/api/start", json={"message": "Hello GLaDOS", "conversation_id": "telegram:1"})
    await api.post("/api/start", json={"message": "Hello again", "conversation_id": "telegram:1"})
    await api.post("/api/start", json={"message": "Hello there", "conversation_id": "telegram:2"})

    second_turn = fake_llm.calls[2:4]
    expected_history = "User: Hello GLaDOS\nGLaDOS: GLaDOS: None"
    assert [call.user_input.conversation_history for call in second_turn] == [expected_history] * 2
    assert fake_llm.calls[4].user_input.conversation_history is None
    assert conversations.recent_turns("telegram:1") == [
        ("Hello GLaDOS", "GLaDOS: None"),
        ("Hello again", "GLaDOS: None"),
    ]


async def test_follow_up_reaches_the_tool_rewritten(api, pipeline, fake_mcp):
    fake_llm, _ = pipeline
    conversations.add_turn("telegram:1", "Turn on the kitchen lights", "Done.")

    def follow_up(user_input):
        if user_input.conversation_history:
            return OrchestratorOutputSchema(tool_name="Home Assistant Tool",
                                            standalone_query="Turn off the kitchen lights")
        return route_by_keyword(user_input)

    fake_llm.on(OrchestratorOutputSchema, follow_up)

    await api.post("/api/start", json={"message": "Turn it off", "conversation_id": "telegram:1"})

    assert fake_mcp.llm_requests[0]["messages"][1]["content"] == "Turn off the kitchen lights"
    # GLaDOS still answers the message as it was written, with the history for context.
    assert fake_llm.calls[1].user_input.chat_message == "Turn it off"


async def test_failed_pipeline_is_not_remembered(api, pipeline):
    fake_llm, _ = pipeline

    def rate_limited(_):
        raise RuntimeError("429")

    fake_llm.on(OrchestratorOutputSchema, rate_limited)

    await api.post("/api/start", json={"message": "Hello GLaDOS", "conversation_id": "telegram:1"})

    assert conversations.recent_turns("telegram:1") == []


async def test_audio_transcript_joins_the_conversation(api, pipeline):
    _, fake_http = pipeline
    fake_openrouter_stt(fake_http, "Hello")
    fake_speaker(fake_http)

    await api.post("/api/start_audio", params={"conversation_id": "telegram:1"}, content=make_wav())

    assert conversations.recent_turns("telegram:1") == [("Hello", "GLaDOS: None")]


async def test_clear_conversation(api):
    conversations.add_turn("telegram:1", "Hello", "Oh. It's you.")

    response = await api.delete("/api/conversations/telegram:1")

    assert response.status_code == 200
    assert conversations.recent_turns("telegram:1") == []


###########
# /speech #
###########
async def test_speech_returns_ogg_opus_voice_note(api, fake_http):
    fake_http.on("POST", TTS_PATH, httpx.Response(200, json={"url": AUDIO_URL}))
    fake_http.on("GET", "/api/tts_proxy/reply.mp3", httpx.Response(200, content=make_wav()))

    response = await api.post("/api/speech", json={"text": "Hello, test subject."})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/ogg"
    assert response.content.startswith(b"OggS")


async def test_speech_render_failure_is_502(api, fake_http):
    fake_http.on("POST", TTS_PATH, httpx.Response(500, text="piper down"))

    response = await api.post("/api/speech", json={"text": "Hello"})

    assert response.status_code == 502
    assert response.json() == {"message": "Failed to render speech."}


async def test_speech_with_undecodable_audio_is_502(api, fake_http):
    fake_http.on("POST", TTS_PATH, httpx.Response(200, json={"url": AUDIO_URL}))
    fake_http.on("GET", "/api/tts_proxy/reply.mp3", httpx.Response(200, content=b"not audio"))

    response = await api.post("/api/speech", json={"text": "Hello"})

    assert response.status_code == 502
