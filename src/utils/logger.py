"""
日志配置
使用loguru替代标准logging，更简洁好用
"""
import sys
from loguru import logger

# 移除默认handler
logger.remove()

# 控制台输出：INFO级别，带颜色
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# 文件输出：DEBUG级别，自动轮转
logger.add(
    "logs/rag_system.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

# 请求日志：单独记录每次RAG请求
logger.add(
    "logs/requests.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    level="INFO",
    rotation="50 MB",
    retention="30 days",
    encoding="utf-8",
    filter=lambda record: "request" in record["extra"].get("tags", []),
)
