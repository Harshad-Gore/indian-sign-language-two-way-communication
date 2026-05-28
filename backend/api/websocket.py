"""
WebSocket handler for real-time translation streaming.

WS /realtime/translate

Message protocol (client → server):
  { "type": "translate", "text": "...", "isl_grammar": true }
  { "type": "ping" }

Message protocol (server → client):
  { "event": "log",       "data": { "level": "info", "message": "..." } }
  { "event": "nlp",       "data": { ...NLPBreakdown... } }
  { "event": "animation", "data": { ...AnimationData... } }
  { "event": "done",      "data": { "processing_time_ms": 123 } }
  { "event": "error",     "data": { "message": "..." } }
"""

from __future__ import annotations
import json
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from services import nlp_engine, animation_engine
from config import settings


async def _send(ws: WebSocket, event: str, data: Any) -> None:
    try:
        await ws.send_text(json.dumps({"event": event, "data": data}))
    except Exception:
        pass


async def realtime_translate_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    client = websocket.client
    logger.info(f"WS connected: {client}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, "error", {"message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await _send(websocket, "pong", {})
                continue

            if msg_type == "translate":
                text = msg.get("text", "").strip()
                isl_grammar = msg.get("isl_grammar", True)

                if not text:
                    await _send(websocket, "error", {"message": "Empty text"})
                    continue

                t_start = time.perf_counter()

                # Step 1: log
                await _send(websocket, "log", {"level": "info", "message": f"Processing: \"{text[:80]}\""})

                # Step 2: NLP
                await _send(websocket, "log", {"level": "info", "message": "Running NLP pipeline..."})
                try:
                    nlp_result = nlp_engine.process_text(text, apply_isl_grammar=isl_grammar)
                    await _send(websocket, "nlp", {
                        "original": nlp_result["original"],
                        "simplified": nlp_result["simplified"],
                        "gloss_sequence": nlp_result["gloss_sequence"],
                        "tokens": nlp_result["tokens"],
                        "sentence_structure": nlp_result["sentence_structure"],
                    })
                    await _send(websocket, "log", {
                        "level": "success",
                        "message": f"Gloss: {' → '.join(nlp_result['gloss_sequence'])}",
                    })
                except Exception as e:
                    await _send(websocket, "error", {"message": f"NLP error: {e}"})
                    continue

                # Step 3: Animation
                await _send(websocket, "log", {"level": "info", "message": "Generating animation frames..."})
                try:
                    anim = animation_engine.generate_animation(
                        gloss_sequence=nlp_result["gloss_sequence"],
                        fps=settings.animation_fps,
                        hold_frames=20,
                        interp_frames=settings.interpolation_frames,
                        source_text=text,
                    )
                    await _send(websocket, "animation", anim)
                    await _send(websocket, "log", {
                        "level": "success",
                        "message": f"Animation ready: {anim['total_frames']} frames @ {anim['fps']}fps",
                    })
                except Exception as e:
                    await _send(websocket, "error", {"message": f"Animation error: {e}"})
                    continue

                processing_ms = round((time.perf_counter() - t_start) * 1000, 1)
                await _send(websocket, "done", {"processing_time_ms": processing_ms})
                logger.info(f"WS translation done in {processing_ms}ms")

            else:
                await _send(websocket, "error", {"message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {client}")
    except Exception as e:
        logger.error(f"WS error: {e}")
        try:
            await _send(websocket, "error", {"message": str(e)})
        except Exception:
            pass
