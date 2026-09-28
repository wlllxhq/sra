"""
RAG Prompt模板
定义系统提示词和用户提示词的构建逻辑
"""

# RAG系统提示词
RAG_SYSTEM_PROMPT = """你是一个专业的学术科研助理。你的核心职责是基于提供的参考文献内容，准确、有条理地回答用户的问题。

## 工作规则

1. **严格基于参考文献回答**：只使用提供的参考文献中的信息来回答，不要使用你自己的知识或编造内容。
2. **引用标注**：回答中引用参考文献时，必须使用 [来源: 文档名, 第X页] 的格式标注出处。
3. **信息不足时如实说明**：如果参考文献中没有足够信息回答问题，请明确说"根据当前知识库中的文献，未找到与您问题直接相关的信息"，并建议用户换一种方式提问或上传更多相关文献。
4. **回答格式**：
   - 使用清晰的分点或分段组织答案
   - 关键术语可以加粗标注
   - 涉及数据对比时，优先使用表格展示
   - 保持学术严谨性，避免口语化表达

## 回答示例

用户问："这篇论文用了什么方法？"

好的回答：
该论文提出了基于**注意力机制**的特征融合方法[来源: example.pdf, 第3页]。主要包含以下创新点：
1. **多尺度特征提取**：使用三种不同大小的卷积核...[来源: example.pdf, 第4页]
2. **自适应权重分配**：通过注意力机制动态调整...[来源: example.pdf, 第5页]
"""

# 降级提示词（检索无结果时使用）
FALLBACK_PROMPT = """当前知识库中未找到与您问题相关的文档内容。

可能的原因：
1. 尚未上传相关论文或文档
2. 已上传的文档中不包含相关内容
3. 提问方式可能需要调整

建议您：
- 确认相关论文已上传并成功索引
- 尝试用更具体的关键词提问
- 尝试拆分复杂问题为多个简单问题"""


def build_rag_prompt(question: str, context_list: list[dict]) -> str:
    """
    构建RAG的用户提示词

    将检索到的参考文献段落和用户问题组装成结构化的prompt

    Args:
        question: 用户问题
        context_list: 检索到的上下文列表，每个包含 content, source, page

    Returns:
        str: 组装好的user prompt
    """
    if not context_list:
        return f"用户问题：{question}\n\n（当前知识库中无相关文档，请按照降级策略回答）"

    # 构建参考文献部分
    context_parts = []
    for i, ctx in enumerate(context_list):
        source = ctx.get("source", "未知文档")
        page = ctx.get("page", "?")

        # 提取文件名（去掉路径）
        filename = source.split("/")[-1] if "/" in source else source
        filename = filename.split("\\")[-1] if "\\" in filename else filename

        context_parts.append(
            f"【参考文献 {i+1}】\n"
            f"文档: {filename}\n"
            f"页码: 第{page}页\n"
            f"内容:\n{ctx['content']}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = (
        f"以下是从知识库中检索到的参考文献内容：\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"用户问题：{question}\n\n"
        f"请基于以上参考文献回答用户的问题，并在回答中标注引用来源。"
    )

    return prompt


def build_comparison_prompt(question: str,
                            context_a: dict,
                            context_b: dict) -> str:
    """
    构建对比分析的提示词（Agent调用时使用）

    Args:
        question: 对比问题
        context_a: 论文A的上下文
        context_b: 论文B的上下文
    """
    return (
        f"请对比分析以下两篇论文的内容，回答用户的问题。\n\n"
        f"【论文A】{context_a.get('source', '未知')} 第{context_a.get('page', '?')}页：\n"
        f"{context_a['content']}\n\n"
        f"【论文B】{context_b.get('source', '未知')} 第{context_b.get('page', '?')}页：\n"
        f"{context_b['content']}\n\n"
        f"用户问题：{question}"
    )
