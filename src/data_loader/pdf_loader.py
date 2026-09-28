"""
PDF文档加载器
使用pdfplumber解析PDF，对中文学术论文效果较好，支持表格提取
"""
import pdfplumber
from loguru import logger
from .base_loader import BaseLoader, Document


class PDFLoader(BaseLoader):

    SUPPORTED_EXTENSIONS = [".pdf"]

    def load(self, file_path: str) -> list[Document]:
        logger.info(f"加载PDF文件: {file_path}")
        documents = []

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                empty_pages = 0

                for i, page in enumerate(pdf.pages):
                    # 提取文本
                    text = page.extract_text()

                    # 如果页面文本为空，尝试提取表格内容
                    if not text or not text.strip():
                        tables = page.extract_tables()
                        if tables:
                            text = self._tables_to_text(tables)

                    if text and text.strip():
                        # 清理文本：去除多余空白、修复换行
                        text = self._clean_text(text)
                        documents.append(Document(
                            content=text,
                            source=file_path,
                            page=i + 1,
                            total_pages=total_pages,
                            metadata={
                                "format": "pdf",
                                "page_width": float(page.width),
                                "page_height": float(page.height),
                            }
                        ))
                    else:
                        empty_pages += 1

                logger.info(
                    f"PDF解析完成: {file_path}, "
                    f"共{total_pages}页, 有效{len(documents)}页, "
                    f"空白{empty_pages}页"
                )

        except Exception as e:
            logger.error(f"PDF解析失败: {file_path}, 错误: {e}")
            raise

        return documents

    def _clean_text(self, text: str) -> str:
        """清理PDF提取的文本"""
        import re
        # 合并连续空白
        text = re.sub(r'[ \t]+', ' ', text)
        # 合并连续换行（保留段落间一个空行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _tables_to_text(self, tables: list) -> str:
        """将表格转为文本格式"""
        lines = []
        for table_idx, table in enumerate(tables):
            lines.append(f"[表格 {table_idx + 1}]")
            for row in table:
                cells = [str(cell) if cell else "" for cell in row]
                lines.append(" | ".join(cells))
            lines.append("")
        return "\n".join(lines)
