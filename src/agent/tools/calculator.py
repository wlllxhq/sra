"""
工具: calculator（计算器）
===========================
执行数学计算，支持基础运算、统计计算和数学函数。
"""
import ast
import operator
import statistics
from typing import Any

from .base_tool import ToolBase, ToolResult


class CalculatorTool(ToolBase):
    """计算器工具"""

    # 支持的运算符
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # 支持的数学函数
    FUNCTIONS = {
        'sqrt': lambda x: x ** 0.5,
        'abs': abs,
        'round': round,
        'sum': sum,
        'max': max,
        'min': min,
        'mean': statistics.mean,
        'median': statistics.median,
        'stdev': statistics.stdev,
        'variance': statistics.variance,
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "执行数学计算，支持基础运算(+,-,*,/,**)、统计函数(mean,median,stdev)和数学函数(sqrt,abs,round)"

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "expression": "数学表达式，如 '2+3*4' 或 'mean([1,2,3,4])'",
            "numbers": "数字列表（可选，用于统计计算）",
        }

    def execute(self, expression: str = "", numbers: list = None, **kwargs) -> ToolResult:
        if not expression and not numbers:
            return ToolResult(success=False, error="缺少expression或numbers参数")

        try:
            result = None

            # 如果有表达式，解析并计算
            if expression:
                result = self._safe_eval(expression)

            # 如果有数字列表且表达式中包含统计函数
            if numbers and isinstance(numbers, list):
                # 如果表达式是函数调用，传入numbers
                if result is None and expression:
                    result = self._safe_eval(expression, numbers)

            if result is None and numbers:
                # 默认对numbers求和
                result = sum(numbers)

            return ToolResult(
                success=True,
                content=f"计算结果: {result}",
                detail=f"表达式: {expression}\n结果: {result}",
                metadata={"result": result, "expression": expression},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"计算失败: {str(e)}",
            )

    def _safe_eval(self, expr: str, data: list = None) -> Any:
        """
        安全计算表达式（使用AST解析，不使用eval）

        Args:
            expr: 数学表达式字符串
            data: 可选的数据列表（用于统计函数）

        Returns:
            计算结果
        """
        # 解析AST
        tree = ast.parse(expr, mode='eval')

        # 递归求值
        def _eval(node):
            if isinstance(node, ast.Constant):  # 数字常量
                return node.value

            elif isinstance(node, ast.BinOp):  # 二元运算
                left = _eval(node.left)
                right = _eval(node.right)
                op_func = self.OPERATORS.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
                return op_func(left, right)

            elif isinstance(node, ast.UnaryOp):  # 一元运算
                operand = _eval(node.operand)
                op_func = self.OPERATORS.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
                return op_func(operand)

            elif isinstance(node, ast.Call):  # 函数调用
                if not isinstance(node.func, ast.Name):
                    raise ValueError("只支持简单函数名")

                func_name = node.func.id
                func = self.FUNCTIONS.get(func_name)
                if func is None:
                    raise ValueError(f"不支持的函数: {func_name}")

                # 获取参数
                args = [_eval(arg) for arg in node.args]

                # 如果提供了data且函数需要列表，使用data
                if data and not args and func_name in ['mean', 'median', 'stdev', 'variance', 'sum', 'max', 'min']:
                    return func(data)

                return func(*args)

            elif isinstance(node, ast.List):  # 列表
                return [_eval(elt) for elt in node.elts]

            else:
                raise ValueError(f"不支持的AST节点类型: {type(node).__name__}")

        return _eval(tree.body)
