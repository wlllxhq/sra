"""
工具基类
========
定义所有Agent工具的抽象接口和返回值结构。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """
    工具执行结果

    Attributes:
        success: 是否执行成功
        content: 结果内容（文本摘要，≤300字供observation事件使用）
        detail: 详细结果（供Agent内部推理使用，不直接展示给前端）
        error: 错误信息（success=False时）
        metadata: 附加元数据
    """
    success: bool = True
    content: str = ""
    detail: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def to_observation_text(self) -> str:
        """转为observation事件的content（截断到300字）"""
        text = self.content if self.success else f"工具执行失败: {self.error}"
        if len(text) > 300:
            text = text[:297] + "..."
        return text


class ToolBase(ABC):
    """
    工具抽象基类

    子类必须实现:
        - name: 工具名称（snake_case英文）
        - description: 工具功能描述
        - execute(): 执行逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（snake_case英文，前端展示用）"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能描述（中文，供ReAct系统提示词和前端展示）"""
        ...

    @property
    def parameters(self) -> dict[str, str]:
        """参数说明 {参数名: 类型描述}，默认无参数"""
        return {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        ...

    def safe_execute(self, **kwargs) -> ToolResult:
        """安全执行：捕获所有异常，返回error结果而非抛出"""
        try:
            return self.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"工具 {self.name} 执行异常: {str(e)}",
            )
