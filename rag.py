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

# 列举型查询关键词：用户想"列出/查看全部用例"而非"找最相关的几条"
LIST_QUERY_KEYWORDS = [
    "有哪些用例", "列出用例", "列出所有", "显示所有", "查看所有",
    "所有用例", "全部用例", "用例列表", "用例清单", "包含哪些",
    "有多少用例", "用例数", "统计", "都有什么", "有哪些测试",
    "列出来", "全部列出", "所有测试用例",
]


def is_list_query(user_input: str) -> bool:
    """判断用户提问是否为列举型查询（想看全部用例而非找最相关的几条）。

    特征：包含"有哪些用例""列出""全部""所有"等关键词。
    这类查询应走元数据全量查询，而非向量相似度 top-k 检索，避免漏掉用例。
    """
    if not user_input:
        return False
    text = user_input.strip()
    # 关键词命中即为列举型查询
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
    匹配不到返回 None。
    """
    if not user_input:
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
            # 按分隔词截断，取数据集名部分
            for sep in separators:
                idx = name.find(sep)
                if idx > 0:
                    name = name[:idx].strip()
                    break
            if name:
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

        self.chain = self.__get_chain()

    def _get_all_cases_as_docs(self, dataset_override=None):
        """按当前过滤条件全量查询用例（用于列举型查询），返回 Document 列表。

        :param dataset_override: 若指定则覆盖 dataset 过滤条件
                                （用于从用户提问中提取数据集名的情况）
        """
        where = {}
        if dataset_override:
            where["dataset"] = dataset_override
        elif self.metadata_filter:
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

            if is_list_query(user_input):
                # 尝试从提问中提取数据集名，进一步缩小范围
                ds_in_query = extract_dataset_from_query(user_input)
                docs = self._get_all_cases_as_docs(dataset_override=ds_in_query)
                print(f"[路由] 识别为列举型查询，全量返回 {len(docs)} 条用例"
                      f"{' (数据集: ' + ds_in_query + ')' if ds_in_query else ''}")
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
