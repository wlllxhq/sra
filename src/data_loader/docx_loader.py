"""
Word文档加载器
使用python-docx解析.docx文件
"""
from docx import Document as DocxDocument
from loguru import logger
from .base_loader import BaseLoader, Document


class DocxLoader(BaseLoader):

    SUPPORTED_EXTENSIONS = [".docx", ".doc"]

    def load(self, file_path: str) -> list[Document]:
        logger.info(f"加载Word文件: {file_path}")
        documents = []

        try:
            doc = DocxDocument(file_path)

            # 收集所有段落和表格，按页组织
            # python-docx不直接支持按页拆分，这里按段落分块
            all_content = []
            current_page_content = []
            char_count = 0
            page_num = 1
            chars_per_page = 2000  # 模拟每页约2000字符

            for element in doc.element.body:
                # 处理段落
                if element.tag.endswith('}p'):
                    for para in doc.paragraphs:
                        if para._element is element and para.text.strip():
                            text = para.text.strip()
                            current_page_content.append(text)
                            char_count += len(text)

                            # 模拟分页
                            if char_count >= chars_per_page:
                                documents.append(Document(
                                    content="\n".join(current_page_content),
                                    source=file_path,
                                    page=page_num,
                                    total_pages=0,  # 后面更新
                                    metadata={"format": "docx"}
                                ))
                                page_num += 1
                                current_page_content = []
                                char_count = 0

                # 处理表格
                elif element.tag.endswith('}tbl'):
                    for table in doc.tables:
                        if table._element is element:
                            table_text = self._table_to_text(table)
                            if table_text:
                                current_page_content.append(table_text)
                                char_count += len(table_text)

            # 处理最后一页
            if current_page_content:
                documents.append(Document(
                    content="\n".join(current_page_content),
                    source=file_path,
                    page=page_num,
                    total_pages=page_num,
                    metadata={"format": "docx"}
                ))

            # 回填总页数
            for doc_item in documents:
                doc_item.total_pages = page_num

            logger.info(
                f"Word解析完成: {file_path}, "
                f"共{len(documents)}页"
            )

        except Exception as e:
            logger.error(f"Word解析失败: {file_path}, 错误: {e}")
            raise

        return documents

    def _table_to_text(self, table) -> str:
        """将Word表格转为文本"""
        lines = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            lines.append(" | ".join(cells))
        return "\n".join(lines) if lines else ""
