"""
LLM 客户端抽象基类 — 整个客户端架构的"契约"

==================== 设计说明 ====================

【本文件职责】
    定义所有 LLM 客户端必须遵守的"接口契约"。
    所有格式的客户端（OpenAI / Anthropic）必须继承 BaseLLMClient，
    对外暴露统一的调用接口，上层业务代码无需关心底层协议差异。

【核心设计模式：模板方法模式（Template Method Pattern）】
    父类定义"算法骨架"，子类实现"具体步骤"：
        - 抽象方法（子类必须实现）：chat(), chat_stream()
        - 通用方法（子类可复用）：chat_stream_print(), build_assistant_message()

    类比：做菜的食谱
        父类（食谱）：备菜 → 炒菜 → 调味 → 装盘
        子类A（川菜）：实现"炒菜"步骤 → 用辣椒
        子类B（粤菜）：实现"炒菜"步骤 → 用蚝油
        "装盘"步骤通用，子类不需要重写。

【核心知识点：ABC 抽象基类（Abstract Base Class）】
    继承 ABC 意味着这个类被定义为"抽象基类"，有两大作用：

    1. 禁止实例化：
       client = BaseLLMClient()  # ❌ TypeError: Can't instantiate abstract class
       抽象类只能被继承，不能直接创建对象。
       因为它定义的是"接口"而非"实现"，实例化一个没有具体实现的类没有意义。

    2. 强制子类实现抽象方法：
       配合 @abstractmethod 装饰器使用。
       如果子类没有实现所有抽象方法，子类也无法实例化：
           class BadClient(BaseLLMClient):
               pass  # 没有实现 chat() 和 chat_stream()
           BadClient()  # ❌ TypeError!

    3. 在 LLM 客户端设计中的价值：
       - 定义统一接口：所有客户端必须实现 chat() 和 chat_stream()
       - 保证一致性：不管底层是 OpenAI 还是 Anthropic，调用方代码写法一样
       - 方便扩展：新增模型提供商，只需继承基类并实现抽象方法即可
       - 编译期检查：忘记实现某个方法时，实例化就会报错（而非运行时才发现）

【核心知识点：面向接口编程（Program to Interface）】
    上层代码只依赖 BaseLLMClient（接口），不依赖具体实现：
        def do_something(client: BaseLLMClient):  # 参数类型是基类
            result = client.chat(messages)         # 不关心是 OpenAI 还是 Anthropic
    这就是"依赖倒置原则"（DIP）：高层模块不依赖低层模块，都依赖抽象。

【统一返回格式】
    非流式 → ChatResult（一次性返回完整结果）
    流式   → yield StreamChunk（逐块返回，打字机效果）

==================================================
"""

# ==================== 导入区 ====================
# 【知识点：abc 模块】
#   abc = Abstract Base Classes，Python 标准库中专门用于定义抽象基类的模块。
#   - ABC: 抽象基类的基类（继承它表示"我是一个抽象类"）
#   - abstractmethod: 装饰器，标记"子类必须实现这个方法"
#
# 【知识点：Generator 类型注解】
#   Generator[YieldType, SendType, ReturnType]
#   - YieldType: yield 出去的类型（StreamChunk）
#   - SendType:  外部 send() 进来的类型（None = 不接受）
#   - ReturnType: 生成器结束时的返回值（None = 无返回值）
#   流式调用就是一个生成器：每次 yield 一个数据块，而非一次性返回全部。

from abc import ABC, abstractmethod
from typing import Optional, Generator

from advanced_llm.models import ChatResult, StreamChunk


class BaseLLMClient(ABC):
    """
    LLM 客户端统一接口（抽象基类）

    【继承关系】
        BaseLLMClient (ABC)
        ├── OpenAIClient      → DeepSeek / Qwen / Kimi（OpenAI 兼容格式）
        └── AnthropicClient   → Claude（Anthropic 原生格式）

    子类必须实现（@abstractmethod 强制）：
        - chat()          非流式调用（发请求 → 等完整响应 → 返回）
        - chat_stream()   流式调用（发请求 → 逐块接收 → yield 出去）

    子类可选覆盖（已有通用实现）：
        - chat_stream_print()        流式打印（默认实现已通用）
        - build_assistant_message()  多轮对话消息构建

    【知识点：为什么 __init__ 不是抽象方法？】
    因为所有客户端的初始化逻辑是相似的（读配置、设置模型参数），
    放在基类统一处理，子类只需 super().__init__() 后补充自己的初始化。
    这也是"模板方法模式"的体现：公共逻辑上提，差异逻辑下放。
    """

    def __init__(self, model_name: str = None):
        """
        初始化客户端（从全局配置中读取模型信息）

        Args:
            model_name: 模型别名（如 "deepseek", "qwen", "kimi", "claude"）
                        为 None 时使用 ACTIVE_MODEL 指定的默认模型

        【知识点：延迟导入（Lazy Import）】
        这里用 `from advanced_llm.config import cfg` 而非文件顶部导入，
        是为了避免循环导入：
            config.py → factory.py → clients/base.py → config.py（循环！）
        放在函数内部，只有调用 __init__ 时才真正导入，此时模块已加载完毕。
        """
        from advanced_llm.config import cfg
        model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model
        self.model = model_cfg.model              # 模型标识（如 "deepseek-chat"）
        self.model_name = model_cfg.name          # 别名（如 "deepseek"）
        self.provider = model_cfg.provider        # API 格式（"openai" / "anthropic"）
        self.enable_thinking = cfg.ENABLE_THINKING          # 全局思考开关
        self.supports_thinking = model_cfg.supports_thinking  # 该模型是否支持思考
        self.thinking_param_type = model_cfg.thinking_param_type  # 思考参数类型

    # ==================== 必须实现（抽象方法） ====================
    # 【知识点：@abstractmethod 装饰器】
    #   被 @abstractmethod 标记的方法：
    #     1. 基类中不需要有具体实现（用 ... 或 pass 占位）
    #     2. 子类必须重写（override），否则子类也是抽象类，无法实例化
    #     3. 相当于 Java/TypeScript 中的 interface 方法声明
    #
    #   ... 是 Python 的 Ellipsis 对象，在抽象方法中用作"空实现"占位符，
    #   比 pass 更明确地表达"这里故意留空，由子类实现"的意图。

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> ChatResult:
        """
        非流式调用（发送请求 → 等待完整响应 → 一次性返回）

        Args:
            messages: 对话历史列表，格式：
                      [{"role": "system"|"user"|"assistant", "content": "..."}]
                      多轮对话时包含完整历史（由上下文管理器裁剪）
            **kwargs: 额外参数（tools, tool_choice, temperature, enable_thinking 等）
                      使用 **kwargs 是为了让不同子类可以接受各自特有的参数

        Returns:
            ChatResult 统一结构（屏蔽了 OpenAI/Anthropic 的响应格式差异）

        【知识点：**kwargs 的作用】
        **kwargs 允许函数接受任意数量的关键字参数：
            client.chat(msgs, tools=TOOLS, temperature=0.5)
            client.chat(msgs, enable_thinking=True)
        在基类中不限制具体参数，让子类自由扩展。
        """
        ...  # 抽象方法，由子类实现

    @abstractmethod
    def chat_stream(self, messages: list[dict], **kwargs) -> Generator[StreamChunk, None, None]:
        """
        流式调用（发送请求 → 逐块接收 → 逐个 yield 出去）

        【知识点：生成器（Generator）vs 普通函数】
        普通函数：return 一次，调用结束
        生成器函数：yield 多次，每次暂停等下次调用继续

        流式输出为什么用生成器？
            - 模型生成 1000 字，每生成 5 个字就推送一次
            - 如果用 return，必须等 1000 字全部生成完才能返回（用户等很久）
            - 用 yield，每 5 个字就"吐"一次给调用方（打字机效果）

        调用方式：
            for chunk in client.chat_stream(messages):
                print(chunk.data, end="")  # 逐字打印

        Args:
            messages: 同上
            **kwargs: 额外参数

        Yields:
            StreamChunk 统一结构（type="reasoning"/"content"/"done"）
        """
        ...  # 抽象方法，由子类实现

    # ==================== 通用实现（模板方法，子类可覆盖但通常不需要） ====================
    # 【知识点：模板方法模式的体现】
    #   chat_stream_print() 定义了"流式打印"的算法骨架：
    #     1. 遍历 chunk → 2. 分类处理 → 3. 拼接结果 → 4. 返回 ChatResult
    #   其中"遍历 chunk"调用的是抽象方法 chat_stream()（由子类实现），
    #   而"分类处理、拼接结果"是通用逻辑（所有客户端一样）。
    #   这就是模板方法：父类控制流程，子类填充细节。

    def chat_stream_print(self, messages: list[dict], **kwargs) -> ChatResult:
        """
        流式调用并实时打印到终端，最终返回完整结果

        通用实现，子类一般无需覆盖。
        内部调用 self.chat_stream()（多态：实际执行的是子类的实现）。

        【知识点：多态（Polymorphism）】
        self.chat_stream() 在运行时根据 self 的实际类型决定调用哪个版本：
            - self 是 OpenAIClient    → 调用 OpenAIClient.chat_stream()
            - self 是 AnthropicClient → 调用 AnthropicClient.chat_stream()
        这就是"运行时多态"，面向对象三大特性之一。

        Returns:
            ChatResult: 包含完整的 content、reasoning、usage
        """
        reasoning_parts = []   # 收集思考过程的碎片
        content_parts = []     # 收集回答内容的碎片
        usage = None           # token 统计（最后一个 chunk 才有）
        is_answering = False   # 是否已进入回答阶段（用于格式控制）

        for chunk in self.chat_stream(messages, **kwargs):
            if chunk.type == "reasoning":
                # 思考阶段：打印思考过程
                if not reasoning_parts:
                    print("💭 [思考中] ", end="", flush=True)
                print(chunk.data, end="", flush=True)
                reasoning_parts.append(chunk.data)

            elif chunk.type == "content":
                # 回答阶段：打印最终回答
                if not is_answering:
                    if reasoning_parts:
                        print("\n")  # 思考与回答之间换行分隔
                    is_answering = True
                print(chunk.data, end="", flush=True)
                content_parts.append(chunk.data)

            elif chunk.type == "done":
                # 结束：获取 token 统计
                usage = chunk.usage

        print()  # 结尾换行

        # 将碎片拼接为完整结果
        return ChatResult(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            usage=usage,
        )

    @staticmethod
    def build_assistant_message(result: ChatResult, include_reasoning: bool = False) -> dict:
        """
        构建用于多轮对话拼接的 assistant 消息

        【核心规则（DeepSeek 官方文档明确要求）】
        ┌─────────────────────────────────────────────────────────────┐
        │ 无工具调用时：剥离 reasoning，只保留 content（节省 token） │
        │ 有工具调用时：必须保留 reasoning（API 要求回传，否则报错） │
        └─────────────────────────────────────────────────────────────┘

        【为什么有这个规则？】
        思考内容（reasoning）可能很长（几千 token），每轮都回传很浪费钱。
        但有工具调用时，模型的思考过程包含了"为什么要调这个工具"的推理，
        如果丢弃，模型在后续轮次中会"忘记"自己为什么调了工具，导致逻辑断裂。

        Args:
            result: ChatResult 对象（来自 chat() 的返回值）
            include_reasoning: 是否强制包含 reasoning（覆盖自动判断）

        Returns:
            assistant 消息字典，可直接 append 到 messages 列表中
        """
        msg = {
            "role": "assistant",
            "content": result.content,
        }

        # 有工具调用时强制保留 reasoning（这是 API 硬性要求，不可跳过）
        if result.has_tool_calls:
            include_reasoning = True

        if include_reasoning and result.reasoning:
            msg["reasoning_content"] = result.reasoning

        return msg

    def __repr__(self):
        """调试时的字符串表示

        【知识点：self.__class__.__name__】
        获取当前实例的实际类名（而非写死 "BaseLLMClient"）：
            - OpenAIClient 实例 → 显示 "OpenAIClient(...)"
            - AnthropicClient 实例 → 显示 "AnthropicClient(...)"
        这样基类的 __repr__ 可以被子类直接复用，无需每个子类重写。
        """
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"provider={self.provider!r}, "
            f"thinking={self.enable_thinking})"
        )
