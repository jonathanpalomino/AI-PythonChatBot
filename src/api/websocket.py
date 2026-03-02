# =============================================================================
# src/api/websocket.py
# WebSocket para notificaciones de procesamiento de archivos
# =============================================================================
"""
WebSocket para notificaciones de procesamiento de archivos

Usa canal con prefijo 'pythonchatbot:' para evitar conflictos con DB 0.
"""
from fastapi import WebSocket, WebSocketDisconnect
import json
import redis.asyncio as aioredis
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREFIX = "pythonchatbot:"


async def websocket_endpoint(websocket: WebSocket, file_id: str = None):
    """
    WebSocket para recibir notificaciones de procesamiento.
    
    El cliente puede conectarse a:
    - /ws/file/{file_id} - Notificaciones de un archivo específico
    
    Args:
        websocket: WebSocket connection
        file_id: UUID del archivo a monitorear
    """
    await websocket.accept()
    
    redis_client = aioredis.from_url(settings.REDIS_URL)
    pubsub = redis_client.pubsub()
    
    # Canal con prefijo único
    channel = f"{PREFIX}file_status:{file_id}"
    await pubsub.subscribe(channel)
    
    logger.info(f"WebSocket connected for file: {file_id}, channel: {channel}")
    
    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=1.0
            )
            
            if message and message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    await websocket.send_json(data)
                    logger.debug(f"Sent notification to client for file {file_id}: {data.get('status')}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse notification: {e}")
                except Exception as e:
                    logger.error(f"Failed to send notification: {e}")
            
            # Check client still connected
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for file: {file_id}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        await pubsub.unsubscribe(channel) if channel else None
        await pubsub.close()
        await redis_client.close()
        await websocket.close()
        logger.info(f"WebSocket cleanup completed for file: {file_id}")
