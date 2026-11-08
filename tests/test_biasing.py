"""End-to-end test: Home Assistant's names really change what Whisper hears.

The unit tests prove that the right names are collected and turned into a
prompt. This one proves the prompt reaches whisper.cpp and does its job: each
recording names a device Whisper has never heard of, and it must come back
wrong without Home Assistant and right with it.

Home Assistant is played by a fake websocket server in this process that
answers the read-only commands in hass_api.py with a fixed home. The server
under test is the real one, started with --hass-token and --hass-api pointed at
the fake, running whisper.cpp on the CPU like test_server.py.

The recordings in tests/biasing/ were synthesized with Piper (en_US-lessac-
medium) and are committed, so the test is deterministic. Each was kept only
because the plain model gets its name wrong: a recording that comes back right
without the names proves nothing about biasing.
"""

import asyncio
import re
import sys
import tempfile
import wave
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from aiohttp import WSMsgType, web
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

from . import _DIR, _LOCAL_DIR, _SAMPLES_PER_CHUNK, _START_TIMEOUT, _TRANSCRIBE_TIMEOUT

_BIASING_DIR = _DIR / "biasing"
_TOKEN = "test-token"

# (recording, name that must be heard). What each one says is in
# tests/biasing/sentences.txt.
_CASES = [
    ("ecobee", "ecobee"),
    ("nanoleaf", "nanoleaf"),
    ("govee", "govee"),
    ("tado", "tado"),
    ("yeelight", "yeelight"),
    ("zibbet", "zibbet"),
]

# The home the fake Home Assistant serves: exposed entities with their friendly
# names, all in the living room.
_ENTITIES = {
    "climate.ecobee": "Ecobee",
    "light.nanoleaf": "Nanoleaf",
    "light.govee_strip": "Govee Strip",
    "climate.tado": "Tado",
    "light.yeelight": "Yeelight",
    "light.zibbet_lamp": "Zibbet Lamp",
}


def _results() -> Dict[str, Any]:
    """The reply to each command hass_api.py sends, keyed by type."""
    return {
        "homeassistant/expose_entity/list": {
            "exposed_entities": {
                entity_id: {"conversation": True} for entity_id in _ENTITIES
            }
        },
        "get_states": [
            {"entity_id": entity_id, "attributes": {"friendly_name": name}}
            for entity_id, name in _ENTITIES.items()
        ],
        "config/entity_registry/get_entries": {
            entity_id: {"area_id": "living_room", "aliases": []}
            for entity_id in _ENTITIES
        },
        "config/device_registry/list": [],
        "config/area_registry/list": [
            {"area_id": "living_room", "name": "Living Room", "aliases": []}
        ],
        "config/floor_registry/list": [],
    }


class FakeHomeAssistant:
    """Just enough of Home Assistant's websocket API for hass_api.py."""

    def __init__(self) -> None:
        self.commands: List[str] = []
        self._runner: Optional[web.AppRunner] = None
        self.port = 0

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/api/websocket", self._websocket)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)

        await websocket.send_json({"type": "auth_required"})
        auth = await websocket.receive_json()
        if auth.get("access_token") != _TOKEN:
            await websocket.send_json({"type": "auth_invalid"})
            await websocket.close()
            return websocket

        await websocket.send_json({"type": "auth_ok"})

        results = _results()
        async for msg in websocket:
            if msg.type != WSMsgType.TEXT:
                break

            command = msg.json()
            self.commands.append(command["type"])
            await websocket.send_json(
                {
                    "id": command["id"],
                    "type": "result",
                    "success": True,
                    "result": results[command["type"]],
                }
            )

        return websocket


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())


async def _transcribe(wav_path: Path, extra_args: List[str]) -> Tuple[str, str]:
    """Run the server, transcribe one recording through it, and return the
    transcript and the server's log.

    One server per recording: the server ends a connection once it has sent the
    transcript, and over stdio that ends the server.
    """
    # A file, not a pipe: whisper.cpp logs a lot, and a full pipe would stall
    # the server.
    log_file = tempfile.TemporaryFile()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_vulkan_whisper",
        "--uri",
        "stdio://",
        "--model",
        "base",
        "--device",
        "cpu",
        "--data-dir",
        str(_LOCAL_DIR),
        "--language",
        "en",
        "--debug",
        *extra_args,
        stdin=PIPE,
        stdout=PIPE,
        stderr=log_file,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    def log() -> str:
        log_file.seek(0)
        return log_file.read().decode(errors="replace")

    text: Optional[str] = None
    try:
        await async_write_event(Describe().event(), proc.stdin)
        while True:
            event = await asyncio.wait_for(
                async_read_event(proc.stdout), timeout=_START_TIMEOUT
            )
            assert event is not None, log()[-5000:]
            if Info.is_type(event.type):
                break

        await async_write_event(Transcribe().event(), proc.stdin)
        with wave.open(str(wav_path), "rb") as wav_file:
            await async_write_event(
                AudioStart(
                    rate=wav_file.getframerate(),
                    width=wav_file.getsampwidth(),
                    channels=wav_file.getnchannels(),
                ).event(),
                proc.stdin,
            )
            for chunk in wav_to_chunks(wav_file, _SAMPLES_PER_CHUNK):
                await async_write_event(chunk.event(), proc.stdin)

        await async_write_event(AudioStop().event(), proc.stdin)

        while text is None:
            event = await asyncio.wait_for(
                async_read_event(proc.stdout), timeout=_TRANSCRIBE_TIMEOUT
            )
            assert event is not None, log()[-5000:]
            if Transcript.is_type(event.type):
                text = Transcript.from_event(event).text
    finally:
        proc.stdin.close()
        await proc.wait()

    assert proc.returncode == 0, log()[-5000:]
    return text, log()


@pytest.mark.asyncio
async def test_home_assistant_names_fix_misheard_devices() -> None:
    hass = FakeHomeAssistant()
    await hass.start()
    hass_args = [
        "--hass-token",
        _TOKEN,
        "--hass-api",
        f"http://127.0.0.1:{hass.port}/api",
        # Generous: CI runners are slow, and a fetch that misses the deadline
        # would leave the utterance unbiased.
        "--hass-prompt-timeout",
        "30",
    ]

    failures = []
    try:
        for recording, name in _CASES:
            wav_path = _BIASING_DIR / f"{recording}.wav"
            plain, _ = await _transcribe(wav_path, [])
            biased, log = await _transcribe(wav_path, hass_args)

            # Shown on failure, and with -s.
            print(f"{recording}: {plain!r} -> {biased!r}")

            # The names were fetched from Home Assistant and put in the prompt.
            prompts = re.findall(r"Initial prompt: (.*)", log)
            assert prompts and ("Ecobee" in prompts[-1]), log[-5000:]

            if name in _normalize(plain):
                failures.append(
                    f"{recording}: heard {plain!r} without the names already; "
                    "this recording does not test biasing"
                )
            if name not in _normalize(biased):
                failures.append(f"{recording}: still heard {biased!r} with the names")
    finally:
        await hass.stop()

    assert "get_states" in hass.commands
    assert not failures, "\n".join(failures)
