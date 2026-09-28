"""
纯文本文档加载器
支持 .txt, .md, .csv 等纯文本格式
自动检测文件编码（兼容GBK/UTF-8）
"""
import chardet
from loguru import logger
from .base_loader import BaseLoader, Document


class TextLoader(BaseLoader):

    SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".rst", ".json", ".yaml", ".yml"]

    def load(self, file_path: str) -> list[Document]:
        logger.info(f"加载文本文件: {file_path}")

        # 自动检测编码
        with open(file_path, "rb") as f:
            raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding", "utf-8") or "utf-8"

        logger.debug(f"检测到编码: {encoding} (置信度: {detected.get('confidence', 0):.2f})")

        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # 降级尝试常见编码
            for fallback in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    text = raw.decode(fallback)
                    logger.warning(f"使用降级编码 {fallback} 读取: {file_path}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"无法解码文件: {file_path}")

        # 按段落分块（每2000字符约一页）
        documents = []
        chars_per_page = 2000
        paragraphs = text.split("\n\n")

        current_content = []
        current_chars = 0
        page_num = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            current_content.append(para)
            current_chars += len(para)

            if current_chars >= chars_per_page:
                documents.append(Document(
                    content="\n\n".join(current_content),
                    source=file_path,
                    page=page_num,
                    total_pages=0,
                    metadata={"format": "text", "encoding": encoding}
                ))
                page_num += 1
                current_content = []
                current_chars = 0

        # 最后一页
        if current_content:
            documents.append(Document(
                content="\n\n".join(current_content),
                source=file_path,
                page=page_num,
                total_pages=page_num,
                metadata={"format": "text", "encoding": encoding}
            ))

        # 回填总页数
        for doc in documents:
            doc.total_pages = page_num

        logger.info(f"文本解析完成: {file_path}, 共{len(documents)}页")
        return documents
