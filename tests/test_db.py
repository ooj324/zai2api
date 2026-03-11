import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app.core._db import async_session_maker
from app.models.db_models import RequestLog
from sqlalchemy import select

async def main():
    async with async_session_maker() as session:
        stmt = select(RequestLog).order_by(RequestLog.id.desc()).limit(1)
        res = await session.execute(stmt)
        row = res.first()
        print(dict(row._mapping))

asyncio.run(main())
