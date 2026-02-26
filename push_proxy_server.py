# SPDX-License-Identifier: Apache-2.0
# Push-based disaggregated prefill proxy server.
#
# 2-request concurrent flow:
#   Proxy sends Prefill + Decode requests simultaneously.
#   Decode allocates blocks → sends block info to Prefill's ZMQ.
#   Prefill computes KV and NIXL WRITEs per-layer to Decode.
#   Decode waits for push notification, then generates tokens.

import argparse
import asyncio
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logging.getLogger().handlers:
    log_level = os.environ.get("PUSH_PROXY_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append({
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=prefiller_base_url,
                limits=httpx.Limits(
                    max_connections=None,
                    max_keepalive_connections=None,
                ),
            ),
            "host": host,
            "port": port,
            "id": i,
        })

    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append({
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=decoder_base_url,
                limits=httpx.Limits(
                    max_connections=None,
                    max_keepalive_connections=None,
                ),
            ),
            "host": host,
            "port": port,
            "id": i,
        })

    app.state.prefill_iterator = itertools.cycle(
        range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(
        range(len(app.state.decode_clients)))

    print(
        f"[PushProxy] Initialized {len(app.state.prefill_clients)} prefill "
        f"and {len(app.state.decode_clients)} decode clients.")

    yield

    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()
    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push-based disaggregated prefill proxy server")

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")

    parser.add_argument(
        "--prefiller-hosts", "--prefiller-host",
        type=str, nargs="+", default=["localhost"])
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port",
        type=int, nargs="+", default=[8100])
    parser.add_argument(
        "--prefiller-zmq-port",
        type=int, default=14580,
        help="ZMQ side channel port on prefill instances")

    parser.add_argument(
        "--decoder-hosts", "--decoder-host",
        type=str, nargs="+", default=["localhost"])
    parser.add_argument(
        "--decoder-ports", "--decoder-port",
        type=int, nargs="+", default=[8200])

    args = parser.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(
        zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """Get the next client in round-robin fashion."""
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def _handle_completions(api: str, request: Request):
    """
    2-request concurrent push-based disaggregated prefill flow:
      - Prefill: compute KV + NIXL WRITE per-layer to Decode
      - Decode:  allocate blocks → send block info to Prefill ZMQ →
                 wait for push notification → generate tokens
    Both requests are sent concurrently.
    """
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        decode_client = get_next_client(request.app, "decode")
        prefill_client = get_next_client(request.app, "prefill")

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }

        logger.info(
            "[PushProxy] req=%s prefill=%s decode=%s",
            request_id, prefill_client["id"], decode_client["id"])

        # Prefill request data (sent AFTER decode to let block alloc happen first)
        prefill_data = req_data.copy()
        prefill_data["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_push_kv": True,
            "decode_request_id": request_id,
        }
        logger.info(
            "[PushProxy] prefill req=%s kv_transfer_params=%s",
            request_id, prefill_data["kv_transfer_params"])
        prefill_data["stream"] = False
        prefill_data["max_tokens"] = 1
        prefill_data.pop("min_tokens", None)
        if "max_completion_tokens" in prefill_data:
            prefill_data["max_completion_tokens"] = 1
        if "stream_options" in prefill_data:
            del prefill_data["stream_options"]

        # Decode request: allocate → push block info to Prefill ZMQ →
        # wait for push notification → generate (streaming)
        decode_data = req_data.copy()
        decode_data["kv_transfer_params"] = {
            "do_remote_prefill": True,
            "push_mode": True,
            "prefill_zmq_host": prefill_client["host"],
            "prefill_zmq_port": global_args.prefiller_zmq_port,
            "proxy_request_id": request_id,
        }
        logger.info(
            "[PushProxy] decode req=%s kv_transfer_params=%s",
            request_id, decode_data["kv_transfer_params"])

        # Delay (ms) before sending prefill request.
        # Gives decode time to alloc blocks + send ZMQ block_info,
        # so prefill's save_kv_layer() finds block_info ready (no pending).
        prefill_delay_s = float(os.environ.get(
            "PUSH_PREFILL_DELAY_MS", "20")) / 1000.0

        async def _delayed_prefill():
            """Send prefill request after a short delay."""
            await asyncio.sleep(prefill_delay_s)
            return await prefill_client["client"].post(
                api, json=prefill_data, headers=headers)

        async def generate():
            # Fire prefill AFTER decode stream starts (decode allocs first)
            prefill_task = asyncio.create_task(_delayed_prefill())
            log_chunks = os.environ.get("PUSH_PROXY_CHUNK_LOG", "0") == "1"
            chunk_idx = 0
            try:
                async with decode_client["client"].stream(
                    "POST", api, json=decode_data, headers=headers
                ) as response:
                    response.raise_for_status()
                    logger.info(
                        "[PushProxy] decode req=%s status=%s",
                        request_id, response.status_code)
                    async for line in response.aiter_lines():
                        if line == "":
                            continue
                        chunk = (line + "\n").encode("utf-8")
                        if log_chunks:
                            chunk_idx += 1
                            logger.info(
                                "[PushProxy] decode req=%s chunk=%d bytes=%d",
                                request_id, chunk_idx, len(chunk))
                        yield chunk
            finally:
                # Ensure prefill task completes (even if decode finishes first)
                try:
                    resp = await prefill_task
                    resp.raise_for_status()
                    logger.info(
                        "[PushProxy] prefill req=%s status=%s",
                        request_id, resp.status_code)
                except Exception as e:
                    logger.error("Prefill request failed: %s", e)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(
            f"Error in push proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "mode": "push",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
