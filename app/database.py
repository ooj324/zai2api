import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

def get_db_url(db_url: str = None) -> str:
    """获取数据库连接 URL"""
    url = db_url or settings.DATABASE_URL
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        # 如果已经是 postgresql+asyncpg 或者是 sqlite+aiosqlite，则直接使用
        elif url.startswith("sqlite"):
            if not url.startswith("sqlite+aiosqlite"):
                url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return url
    
    # 默认使用 SQLite
    db_path = settings.DB_PATH
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(db_path)
    return f"sqlite+aiosqlite:///{db_path}"

# 全局引擎和会话工厂
engine = create_async_engine(get_db_url(), echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    """初始化数据库表"""
    from app.models.db_models import Base
    import app.models.db_models  # 确保模型被注册
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """关闭数据库连接"""
    await engine.dispose()
