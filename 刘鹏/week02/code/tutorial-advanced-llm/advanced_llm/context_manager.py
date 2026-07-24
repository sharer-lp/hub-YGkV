"""
上下文管理器 — 多轮对话历史管理策略

==================== 设计说明 ====================

【本文件职责】
    解决多轮对话中"历史消息越来越长"的问题。

【核心背景知识：为什么需要上下文管理？】
    大模型 API 是无状态的（Stateless）：
        - 模型不会"记住"上一次请求的内容
        - 每次请求必须携带完整对话历史
        - 模型看到的"记忆"其实就是你传过去的 messages 列表

    随着对话轮数增加，历史消息会：
        1. 消耗更多 token（成本线性增长：10轮对话 ≈ 1轮的 10 倍费用）
        2. 逼近上下文窗口限制（如 8K/32K/128K，超出会报错）
        3. 降低响应速度（输入越长，处理越慢）

    所以必须"裁剪"历史，在"保留信息"和"控制成本"之间取得平衡。

【核心设计模式：策略模式（Strategy Pattern）】
    将"如何裁剪历史"抽象为可替换的策略对象：
        - ContextManager 不关心具体用哪种策略
        - 只调用 strategy.manage(messages)
        - 切换策略只需换一个对象，无需改 ContextManager 代码

    类比：导航软件
        - ContextManager = 导航软件（只负责"规划路线"）
        - Strategy = 路线策略（最短距离 / 最少时间 / 避免高速）
        - 切换策略不影响导航软件的界面和操作流程

【4 种策略对比】
    ┌─────────────────────┬──────────┬──────────┬──────────────────────────┐
    │ 策略                │ 成本     │ 信息保留 │ 适用场景                 │
    ├─────────────────────┼──────────┼──────────┼──────────────────────────┤
    │ 滑动窗口截断        │ 低       │ 低       │ 闲聊、简单问答           │
    │ Token 预算截断      │ 低       │ 中       │ 精确控制成本             │
    │ 摘要压缩            │ 中       │ 高       │ 长对话保持连贯性         │
    │ 混合策略（推荐）    │ 中       │ 高       │ 大多数生产环境           │
    └─────────────────────┴──────────┴──────────┴──────────────────────────┘

【关键规则（DeepSeek 官方文档）】
    - 无工具调用时：assistant 的 reasoning_content 无需回传（节省 token）
    - 有工具调用时：assistant 的 reasoning_content 必须回传（API 强制要求）
    本模块在 add_assistant_message() 中自动处理这个规则。

==================================================
"""

# ==================== 导入区 ====================
import logging
from abc import ABC, abstractmethod  # 策略模式需要抽象基类
from typing import Optional

logger = logging.getLogger(__name__)


# ==================== Token 估算工具 ====================
# 【知识点：什么是 Token？】
#   Token 是大模型处理文本的最小单位（不是字，也不是词）：
#     - 英文："hello" = 1 token，"unbelievable" = 3 tokens (un + believ + able)
#     - 中文："你好" = 2~3 tokens（每个汉字约 1.5 token）
#     - 标点、空格也算 token
#   模型按 token 计费：输入 token + 输出 token = 总费用
#   模型有上下文窗口限制：如 8K = 约 6000 汉字 / 约 5000 英文单词
#
# 【为什么用估算而非精确计算？】
#   精确计算需要 tiktoken 库（OpenAI 的 tokenizer），但：
#     1. 不同模型的 tokenizer 不同（DeepSeek/Qwen/Claude 各有各的）
#     2. 引入额外依赖增加复杂度
#     3. 上下文管理只需"大概"知道是否超预算，不需要精确到个位数
#   所以轻量级估算足够用了。

def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数（轻量级，无需额外依赖）

    规则：
        - 中文：约 1 字 = 1.5 token
        - 英文：约 1 词 = 1.3 token
        - 混合：取加权平均

    生产环境建议使用 tiktoken 精确计算：
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = len(enc.encode(text))
    """
    if not text:
        return 0

    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    # 中文约 1.5 token/字，英文约 0.3 token/字符（含空格标点）
    return int(chinese_chars * 1.5 + other_chars * 0.35)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算 messages 列表的总 token 数"""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += estimate_tokens(content)
        # 每条消息的格式开销（role、分隔符等）
        total += 4
        # reasoning_content 也计入
        reasoning = msg.get("reasoning_content", "")
        if reasoning:
            total += estimate_tokens(reasoning)
        # tool_calls 也计入
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                total += estimate_tokens(func.get("name", ""))
                total += estimate_tokens(func.get("arguments", ""))
    return total


# ==================== 策略抽象基类 ====================
# 【知识点：策略模式（Strategy Pattern）】
#   属于 GoF 23 种设计模式中的"行为型模式"。
#   核心思想：将"算法"封装为独立的对象，使它们可以互相替换。
#
#   不用策略模式（if-else 硬编码）：
#       if strategy_name == "sliding":
#           messages = messages[-20:]
#       elif strategy_name == "budget":
#           ...
#       elif strategy_name == "summary":
#           ...
#       # 每新增一种策略，这里就要加一个 elif！
#
#   用策略模式：
#       result = strategy.manage(messages)  # 一行搞定，新增策略无需改这里
#
#   符合"开闭原则"：对扩展开放（新增策略类），对修改关闭（不改 ContextManager）。

class ContextStrategy(ABC):
    """上下文管理策略抽象基类

    所有具体策略必须继承此类并实现：
        - manage(): 对 messages 进行裁剪/压缩
        - name:     策略名称（用于日志和调试）
    """

    @abstractmethod
    def manage(self, messages: list[dict]) -> list[dict]:
        """
        对 messages 列表进行裁剪/压缩

        Args:
            messages: 完整对话历史（含 system）

        Returns:
            处理后的 messages（可直接传给 API）
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称（用于日志输出和调试信息）"""
        ...


# ==================== 策略 1：滑动窗口截断 ====================
# 【算法思想】
#   类似操作系统中的"滑动窗口"协议：
#   维护一个固定大小的窗口，窗口随新消息向前滑动，
#   窗口外的旧消息被丢弃。
#
#   示例（max_rounds=3，每轮 2 条）：
#   消息: [S, U1, A1, U2, A2, U3, A3, U4, A4, U5, A5]
#   截断后: [S, U3, A3, U4, A4, U5, A5]  ← 只保留最近 3 轮
#   （S = system，始终保留）

class SlidingWindowStrategy(ContextStrategy):
    """
    滑动窗口截断策略

    原理：只保留最近 N 轮对话，丢弃更早的消息。
    优点：实现简单，零额外开销
    缺点：会丢失早期重要信息（如用户姓名、偏好设定）

    适用场景：
        - 闲聊、问答等不依赖长期记忆的场景
        - 对成本敏感的简单应用

    注意：system 消息始终保留（不受窗口限制）
    """

    def __init__(self, max_rounds: int = 10):
        """
        Args:
            max_rounds: 最大保留轮数（1轮 = 1个user + 1个assistant）
        """
        self.max_rounds = max_rounds

    @property
    def name(self) -> str:
        return f"滑动窗口(max_rounds={self.max_rounds})"

    def manage(self, messages: list[dict]) -> list[dict]:
        # 分离 system 消息和非 system 消息
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        # 计算最大消息数（每轮约 2 条：user + assistant，工具调用可能更多）
        max_messages = self.max_rounds * 4  # 预留工具调用空间

        if len(non_system) > max_messages:
            # 保留最近的 N 条
            non_system = non_system[-max_messages:]
            logger.info(
                f"[{self.name}] 截断历史: {len(messages)} → {len(system_msgs) + len(non_system)} 条"
            )

        return system_msgs + non_system


# ==================== 策略 2：Token 预算截断 ====================

class TokenBudgetStrategy(ContextStrategy):
    """
    Token 预算截断策略

    原理：按 token 数而非消息数截断，更精确地控制上下文大小。
    优点：精确控制成本，不会超出模型上下文窗口
    缺点：可能在不完整的位置截断（如截断到一半的工具调用）

    适用场景：
        - 需要精确控制 token 消耗的生产环境
        - 上下文窗口较小的模型

    截断规则：
        1. 始终保留 system 消息
        2. 始终保留最后一条 user 消息（当前提问）
        3. 从最早的消息开始删除，直到满足预算
        4. 不截断工具调用对（assistant+tool 必须成对保留）
    """

    def __init__(self, max_tokens: int = 8000, reserve_tokens: int = 2000):
        """
        Args:
            max_tokens: 上下文最大 token 数
            reserve_tokens: 为模型输出预留的 token 数
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens

    @property
    def name(self) -> str:
        return f"Token预算(max={self.max_tokens}, reserve={self.reserve_tokens})"

    def manage(self, messages: list[dict]) -> list[dict]:
        budget = self.max_tokens - self.reserve_tokens

        # 分离 system 和非 system
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        # 计算 system 占用
        system_tokens = estimate_messages_tokens(system_msgs)
        available = budget - system_tokens

        if available <= 0:
            logger.warning(f"[{self.name}] system 消息已超出预算!")
            return system_msgs + non_system[-1:]  # 只保留最后一条

        # 从后往前累加，找到满足预算的截断点
        kept = []
        used_tokens = 0

        for msg in reversed(non_system):
            msg_tokens = estimate_messages_tokens([msg])
            if used_tokens + msg_tokens > available:
                break
            kept.insert(0, msg)
            used_tokens += msg_tokens

        if len(kept) < len(non_system):
            logger.info(
                f"[{self.name}] 截断: {len(non_system)} → {len(kept)} 条, "
                f"tokens: {used_tokens}/{available}"
            )

        return system_msgs + kept


# ==================== 策略 3：摘要压缩 ====================

class SummaryStrategy(ContextStrategy):
    """
    摘要压缩策略

    原理：当历史超过阈值时，用模型将早期对话总结为一段摘要，
         替换原始历史，大幅压缩 token 消耗。

    优点：信息保留好，能记住早期关键信息
    缺点：需要额外 API 调用（增加延迟和成本）

    适用场景：
        - 长对话需要保持上下文连贯性
        - 客服、教育辅导等需要记住用户信息的场景

    工作流程：
        1. 检测历史是否超过阈值
        2. 将早期消息发给模型生成摘要
        3. 用摘要替换早期消息（作为 system 补充）
        4. 保留近期消息原文
    """

    def __init__(
        self,
        max_rounds: int = 10,
        summary_trigger: int = 8,
        client=None,
    ):
        """
        Args:
            max_rounds: 保留最近几轮原文
            summary_trigger: 超过几轮时触发摘要
            client: LLM 客户端（用于生成摘要）
        """
        self.max_rounds = max_rounds
        self.summary_trigger = summary_trigger
        self.client = client
        self._summary: Optional[str] = None

    @property
    def name(self) -> str:
        return f"摘要压缩(trigger={self.summary_trigger}, keep={self.max_rounds})"

    def _generate_summary(self, messages: list[dict]) -> str:
        """调用模型生成对话摘要"""
        if not self.client:
            return ""

        # 构建摘要请求
        conversation_text = "\n".join(
            f"{m['role']}: {m.get('content', '')}"
            for m in messages
            if m.get("content")
        )

        summary_messages = [
            {
                "role": "system",
                "content": (
                    "你是一个对话摘要助手。请将以下对话历史压缩为一段简洁的摘要，"
                    "保留关键信息（用户姓名、偏好、重要决定、未解决的问题）。"
                    "摘要应控制在 200 字以内。只输出摘要内容，不要其他解释。"
                ),
            },
            {"role": "user", "content": f"请总结以下对话：\n\n{conversation_text}"},
        ]

        try:
            result = self.client.chat(summary_messages, enable_thinking=False, max_tokens=300)
            return result.content or ""
        except Exception as e:
            logger.error(f"[{self.name}] 生成摘要失败: {e}")
            return ""

    def manage(self, messages: list[dict]) -> list[dict]:
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        # 估算轮数（简单按消息数 / 2）
        approx_rounds = len(non_system) // 2

        if approx_rounds <= self.summary_trigger:
            return messages  # 未触发，原样返回

        # 分割：早期（需要摘要）+ 近期（保留原文）
        keep_count = self.max_rounds * 2  # 每轮约 2 条
        early_msgs = non_system[:-keep_count] if keep_count < len(non_system) else []
        recent_msgs = non_system[-keep_count:]

        # 生成或更新摘要
        if early_msgs:
            self._summary = self._generate_summary(early_msgs)
            logger.info(f"[{self.name}] 已压缩 {len(early_msgs)} 条早期消息为摘要")

        # 重组 messages
        result = list(system_msgs)

        # 将摘要作为 system 补充
        if self._summary:
            result.append({
                "role": "system",
                "content": f"[对话历史摘要]\n{self._summary}",
            })

        result.extend(recent_msgs)
        return result


# ==================== 策略 4：混合策略（推荐） ====================

class HybridStrategy(ContextStrategy):
    """
    混合策略（生产环境推荐）

    原理：结合滑动窗口 + Token 预算 + 可选摘要
        1. 先按轮数粗截断（滑动窗口）
        2. 再按 token 预算精确截断
        3. 可选：对截断部分生成摘要

    优点：兼顾成本控制和信息保留
    缺点：实现稍复杂

    适用场景：
        - 大多数生产环境的首选
        - 需要平衡成本和体验的长对话应用
    """

    def __init__(
        self,
        max_rounds: int = 15,
        max_tokens: int = 8000,
        reserve_tokens: int = 2000,
        enable_summary: bool = False,
        client=None,
    ):
        self._sliding = SlidingWindowStrategy(max_rounds=max_rounds)
        self._budget = TokenBudgetStrategy(max_tokens=max_tokens, reserve_tokens=reserve_tokens)
        self._summary = SummaryStrategy(client=client) if enable_summary else None

    @property
    def name(self) -> str:
        return "混合策略(滑动窗口+Token预算)"

    def manage(self, messages: list[dict]) -> list[dict]:
        # 第一步：滑动窗口粗截断
        messages = self._sliding.manage(messages)

        # 第二步：Token 预算精确截断
        messages = self._budget.manage(messages)

        return messages


# ==================== 上下文管理器（统一入口） ====================

class ContextManager:
    """
    上下文管理器 — 多轮对话历史管理统一入口

    用法：
        from advanced_llm.context_manager import ContextManager, HybridStrategy

        # 创建管理器
        cm = ContextManager(strategy=HybridStrategy(max_rounds=10, max_tokens=8000))

        # 添加消息
        cm.add_user_message("你好")
        cm.add_assistant_message("你好！有什么可以帮你的？")

        # 获取处理后的 messages（自动应用策略）
        messages = cm.get_messages()

        # 重置对话
        cm.reset()
    """

    def __init__(
        self,
        system_prompt: str = "你是一个有帮助的AI助手。",
        strategy: ContextStrategy = None,
    ):
        """
        Args:
            system_prompt: 系统提示词
            strategy: 上下文管理策略（默认使用混合策略）
        """
        from advanced_llm.config import cfg

        self.system_prompt = system_prompt
        self.strategy = strategy or HybridStrategy(
            max_rounds=cfg.MAX_HISTORY_ROUNDS,
            max_tokens=cfg.MAX_CONTEXT_TOKENS,
        )
        self._messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self._turn_count = 0

    @property
    def turn_count(self) -> int:
        """当前对话轮数"""
        return self._turn_count

    @property
    def message_count(self) -> int:
        """当前消息总数"""
        return len(self._messages)

    def add_user_message(self, content: str):
        """添加用户消息"""
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self,
        content: str,
        reasoning: str = None,
        tool_calls: list = None,
        include_reasoning: bool = False,
    ):
        """
        添加助手消息

        关键规则：
        - 无工具调用时：默认不包含 reasoning（节省 token）
        - 有工具调用时：必须包含 reasoning（API 要求）

        Args:
            content: 回答内容
            reasoning: 思考过程
            tool_calls: 工具调用列表
            include_reasoning: 是否强制包含 reasoning
        """
        msg = {"role": "assistant", "content": content}

        # 有工具调用时必须保留 reasoning
        should_include = include_reasoning or bool(tool_calls)

        if reasoning and should_include:
            msg["reasoning_content"] = reasoning

        if tool_calls:
            msg["tool_calls"] = tool_calls

        self._messages.append(msg)

        if content:  # 有最终回答才算一轮
            self._turn_count += 1

    def add_tool_result(self, tool_call_id: str, content: str):
        """添加工具执行结果"""
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def get_messages(self) -> list[dict]:
        """
        获取处理后的 messages（应用上下文管理策略）

        Returns:
            可直接传给 API 的 messages 列表
        """
        return self.strategy.manage(self._messages)

    def get_raw_messages(self) -> list[dict]:
        """获取原始未处理的 messages（调试用）"""
        return list(self._messages)

    def get_token_estimate(self) -> int:
        """估算当前历史的 token 数"""
        return estimate_messages_tokens(self._messages)

    def reset(self, system_prompt: str = None):
        """重置对话历史"""
        if system_prompt:
            self.system_prompt = system_prompt
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self._turn_count = 0

    def __repr__(self):
        return (
            f"ContextManager(strategy={self.strategy.name}, "
            f"turns={self._turn_count}, messages={len(self._messages)}, "
            f"est_tokens={self.get_token_estimate()})"
        )
