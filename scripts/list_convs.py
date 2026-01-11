import asyncio
from sqlalchemy import select
from src.database.connection import get_async_db
from src.models.models import Conversation

async def list_conversations():
    async for db in get_async_db():
        result = await db.execute(select(Conversation).limit(5))
        conversations = result.scalars().all()
        print("Recent Conversations:")
        for conv in conversations:
            print(f"ID: {conv.id} - Title: {conv.title}")
        break

if __name__ == "__main__":
    asyncio.run(list_conversations())
