"""
工具: keyword_extract（关键词提取）
=====================================
从文本中提取关键词和主题词。
"""
import re
from collections import Counter

from .base_tool import ToolBase, ToolResult


class KeywordExtractTool(ToolBase):
    """关键词提取工具"""

    # 停用词（中英文）
    STOP_WORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        '自己', '这', '他', '她', '它', '们', '那', '些', '什么', '怎么', '为什么',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'of', 'in', 'on', 'at',
        'to', 'for', 'with', 'by', 'from', 'as', 'into', 'through',
    }

    @property
    def name(self) -> str:
        return "keyword_extract"

    @property
    def description(self) -> str:
        return "从文本中提取关键词和主题词，按词频排序"

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "text": "需要提取关键词的文本",
            "top_k": "返回前k个关键词，默认10",
        }

    def execute(self, text: str = "", top_k: int = 10, **kwargs) -> ToolResult:
        if not text:
            return ToolResult(success=False, error="缺少text参数")

        # 提取英文单词（长度≥3）和中文词汇（长度≥2）
        words = []

        # 英文单词
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words.extend([w for w in english_words if w not in self.STOP_WORDS])

        # 中文词汇（简单分词：连续中文字符，长度≥2）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        words.extend([w for w in chinese_words if w not in self.STOP_WORDS])

        if not words:
            return ToolResult(
                success=True,
                content="未提取到有效关键词",
                detail="文本中未找到符合要求的词汇",
            )

        # 统计词频
        word_counts = Counter(words)
        top_keywords = word_counts.most_common(top_k)

        # 构建结果
        keyword_list = [f"{word}({count})" for word, count in top_keywords]
        keywords_only = [word for word, _ in top_keywords]

        return ToolResult(
            success=True,
            content=f"提取到{len(top_keywords)}个关键词: {', '.join(keyword_list[:5])}",
            detail="关键词列表（词频）:\n" + "\n".join(
                f"{i+1}. {word}: {count}次" for i, (word, count) in enumerate(top_keywords)
            ),
            metadata={
                "keywords": keywords_only,
                "word_counts": dict(top_keywords),
                "total_unique": len(word_counts),
            },
        )
