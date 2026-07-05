import ssl_fix  # noqa: F401 — 必须在任何会触发 SSL 的 import 之前
import re
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt


# ==================== 查询意图识别 ====================

# 列举型查询关键词（作为 LLM 意图识别失败时的兜底策略）
# 当大模型调用异常时，回退到关键词匹配保证服务可用
LIST_QUERY_KEYWORDS = [
    "有哪些用例", "列出用例", "列出所有", "显示所有", "查看所有",
    "所有用例", "全部用例", "用例列表", "用例清单", "包含哪些",
    "有多少用例", "用例数", "统计", "都有什么", "有哪些测试",
    "列出来", "全部列出", "所有测试用例",
]


def _is_list_query_fallback(user_input: str) -> bool:
    """关键词兜底判定：当 LLM 意图识别不可用时使用。

    特征：包含"有哪些用例""列出""全部""所有"等关键词。
    返回 True 表示列举型查询，应走元数据全量查询。
    """
    if not user_input:
        return False
    text = user_input.strip()
    for kw in LIST_QUERY_KEYWORDS:
        if kw in text:
            return True
    return False


def extract_dataset_from_query(user_input: str):
    """尝试从用户提问中提取数据集名称。

    例如"数据集3333有什么用例" → 提取 "3333"
    "数据集 3333" → "3333"
    "3333数据集" → "3333"
    "数据集V1.0回归测试的用例" → "V1.0回归测试"
    "给我所有数据集的支付模块的测试用例" → None（用户想看所有数据集，不限定）
    匹配不到返回 None。
    """
    if not user_input:
        return None

    # 用户明确想看"所有/全部/各"数据集时，不限定数据集，直接返回 None
    all_dataset_keywords = ["所有数据集", "全部数据集", "各数据集", "每一个数据集", "所有数据集合", "全部的数据集"]
    for kw in all_dataset_keywords:
        if kw in user_input:
            return None

    # 匹配 "数据集xxx" 或 "xxx数据集"
    # 数据集名可能含中文、字母、数字、点、下划线
    patterns = [
        r"数据集[:：\s]*([A-Za-z0-9\u4e00-\u9fa5_.\-]+)",
        r"([A-Za-z0-9\u4e00-\u9fa5_.\-]+)数据集",
    ]
    # 提问中数据集名之后常跟的分隔词，按最长优先匹配截断
    separators = [
        "有什么用例", "有哪些用例", "有什么测试", "有哪些测试",
        "有什么", "有哪些", "包含什么", "包含哪些",
        "的用例", "的测试", "的case", "的用例数",
        "都有", "里有", "中有", "里面",
        "是什么", "有多少",
        "的", "里", "中", "有",
    ]
    for pat in patterns:
        m = re.search(pat, user_input)
        if m:
            name = m.group(1).strip()
            # 提取结果以"的"开头或结尾视为无效（如"的支付模块"是错误提取）
            if name.startswith("的") or name.endswith("的"):
                continue
            # 按分隔词截断，取数据集名部分
            for sep in separators:
                idx = name.find(sep)
                if idx > 0:
                    name = name[:idx].strip()
                    break
            # 截断后再次校验，避免残留"的"
            if name and not name.startswith("的") and not name.endswith("的"):
                return name
    return None


class RagService(object):
    def __init__(self, metadata_filter=None):
        """初始化 RAG 服务。

        :param metadata_filter: 元数据过滤条件，如 {"module": "登录"}，
                                用于限定检索范围。为 None 时检索全部用例。
        """
        self.metadata_filter = metadata_filter

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 持有 Chroma collection 引用，供列举型查询做全量元数据查询
        self.collection = self.vector_service.vector_store._collection

        # 测试用例场景的 prompt：强调以已有用例为依据，回答用例相关问题
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个专业的测试用例助手。"
                 "请根据以下检索到的测试用例参考资料，回答用户关于测试用例的问题。"
                 "你可以帮助用户查询用例、分析用例覆盖度、基于已有用例风格生成新用例建议。\n\n"
                 "输出格式要求：\n"
                 "1. 当回答涉及展示测试用例时，必须使用 Markdown 表格格式输出，表头为："
                 "| 用例ID | 数据集 | 模块 | 优先级 | 标题 | 前置条件 | 测试步骤 | 预期结果 |\n"
                 "2. 如果用例较多，全部展示在表格中，不要省略或截断。\n"
                 "3. 非用例展示的内容（如分析、建议）用普通文本或列表输出即可。\n"
                 "4. 如果参考资料中已列出用例，请完整呈现，不要遗漏。\n\n"
                 "参考资料中的测试用例如下:\n{context}"),
                ("system", "用户的对话历史记录如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        # 意图识别专用模型：用轻量模型 + temperature=0，兼顾速度与稳定性
        self.intent_model = ChatTongyi(
            model=config.intent_model_name,
            temperature=0.0,
        )
        # 意图识别 prompt：强制模型只返回 LIST 或 SEMANTIC，便于程序解析
        # 注意：字符串内避免使用中文全角引号，否则 Python 会误判字符串边界
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个查询意图分类器。请判断用户的提问属于以下哪种类型：\n"
             "- LIST：用户想查看、列举、统计一批用例。特征是关注 有多少、有哪些、全部、所有、列出、给我 等词，"
             "期望得到完整列表而非个别用例的细节。\n"
             "- SEMANTIC：用户想了解某个具体问题、查找特定用例的步骤或预期结果、寻求测试建议等，"
             "期望得到针对性的回答。\n\n"
             "判断要点：\n"
             "1. 用户想看 全部、所有、有哪些 用例 → LIST\n"
             "2. 用户想统计用例数量、查看用例清单 → LIST\n"
             "3. 用户问某个功能怎么测、某条用例的预期结果、某模块的测试要点 → SEMANTIC\n"
             "4. 模棱两可时，若用户明确要求 全部 或 所有 则判 LIST，否则判 SEMANTIC\n"
             "5. 关键判据：用户是否期望看到一批用例的完整列表。只要提问中有 所有、全部、各 等词修饰用例范围，就判 LIST\n\n"
             "示例：\n"
             "- 给我所有数据集的支付模块的测试用例 → LIST（用户要全部支付用例）\n"
             "- 列出登录模块的用例 → LIST\n"
             "- 支付功能怎么测 → SEMANTIC\n"
             "- TC_001的预期结果是什么 → SEMANTIC\n\n"
             "只返回 LIST 或 SEMANTIC 这一个词，不要输出任何其他内容。"),
            ("user", "{question}")
        ])

        self.chain = self.__get_chain()

    def _get_all_cases_as_docs(self, dataset_override=None, module_override=None):
        """按当前过滤条件全量查询用例（用于列举型查询），返回 Document 列表。

        :param dataset_override: 若指定则覆盖 dataset 过滤条件
                                （用于从用户提问中提取数据集名的情况）
        :param module_override: 若指定则按 module 过滤
                                （用于从用户提问中提取模块名的情况）
        """
        where = {}
        if dataset_override:
            where["dataset"] = dataset_override
        if module_override:
            where["module"] = module_override
        # 若未显式指定 dataset/module，则回退到构造时的 metadata_filter
        if not where and self.metadata_filter:
            where = dict(self.metadata_filter)

        try:
            result = self.collection.get(
                where=where if where else None,
                include=["metadatas", "documents"]
            )
            docs = []
            ids = result.get("ids", [])
            metadatas = result.get("metadatas", [])
            documents = result.get("documents", [])
            for cid, meta, doc in zip(ids, metadatas, documents):
                docs.append(Document(page_content=doc, metadata=meta))
            return docs
        except Exception as e:
            print(f"全量查询用例失败: {e}")
            return []

    def _extract_module_from_query(self, user_input: str):
        """从用户提问中提取模块名：匹配知识库中已存在的模块名。

        例如"支付模块的测试用例" → "支付"
        "登录模块有哪些用例" → "登录"
        匹配不到返回 None。
        """
        if not user_input:
            return None
        try:
            # 取知识库中所有模块名
            result = self.collection.get(include=["metadatas"])
            modules = set()
            for meta in result.get("metadatas", []):
                m = meta.get("module", "")
                if m:
                    modules.add(m)
            # 按模块名长度降序匹配，避免短模块名被长模块名包含导致误匹配
            for mod in sorted(modules, key=len, reverse=True):
                if mod in user_input:
                    return mod
        except Exception as e:
            print(f"提取模块名失败: {e}")
        return None

    def _detect_query_intent(self, user_input: str) -> bool:
        """用大模型判断用户提问是否为列举型查询。

        调用轻量模型（intent_model）做意图分类，返回 True 表示列举型。
        若 LLM 调用失败，回退到关键词匹配（_is_list_query_fallback）保证服务可用。
        """
        if not user_input or not user_input.strip():
            return False

        try:
            # 构建意图识别链：prompt → 模型 → 纯文本解析
            intent_chain = self.intent_prompt | self.intent_model | StrOutputParser()
            raw_result = intent_chain.invoke({"question": user_input})
            # 清理返回内容：去空白、转大写，便于匹配
            result = raw_result.strip().upper()

            # 容错解析：只要返回内容包含 LIST 即判定为列举型
            if "LIST" in result:
                print(f"[意图识别] LLM判定: 列举型 (返回: {raw_result.strip()})")
                return True
            else:
                print(f"[意图识别] LLM判定: 语义型 (返回: {raw_result.strip()})")
                return False
        except Exception as e:
            # LLM 调用异常时回退到关键词匹配，保证服务不中断
            print(f"[意图识别] LLM调用失败({e})，回退到关键词匹配")
            return _is_list_query_fallback(user_input)

    def __get_chain(self):
        """获取最终的执行链"""
        # 向量检索器（用于语义相似度查询）
        retriever = self.vector_service.get_retriever(metadata_filter=self.metadata_filter)

        def format_document(docs: list[Document]):
            """将检索到的用例文档格式化为参考文本"""
            if not docs:
                return "无相关测试用例"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"测试用例：{doc.page_content}\n用例元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def get_context(value: dict) -> str:
            """分流获取参考资料：
            - 列举型查询（"有哪些用例""列出全部"等）：全量元数据查询，避免 top-k 漏掉用例
            - 其他查询：走向量相似度检索，返回最相关的 k 条
            """
            user_input = value.get("input", "")

            if self._detect_query_intent(user_input):
                # 列举型查询：尝试从提问中提取数据集名和模块名，缩小全量查询范围
                ds_in_query = extract_dataset_from_query(user_input)
                mod_in_query = self._extract_module_from_query(user_input)
                docs = self._get_all_cases_as_docs(
                    dataset_override=ds_in_query,
                    module_override=mod_in_query,
                )
                scope = []
                if ds_in_query:
                    scope.append(f"数据集: {ds_in_query}")
                if mod_in_query:
                    scope.append(f"模块: {mod_in_query}")
                scope_str = f" ({', '.join(scope)})" if scope else ""
                print(f"[路由] 识别为列举型查询，全量返回 {len(docs)} 条用例{scope_str}")
            else:
                docs = retriever.invoke(user_input)
                print(f"[路由] 语义检索，返回 {len(docs)} 条用例")

            return format_document(docs)

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(get_context)
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    # 测试列举型查询
    print("=== 测试1: 列举型查询 ===")
    res1 = RagService().chain.invoke({"input": "数据集3333有哪些用例？"}, session_config)
    print(res1)
