
md5_path = "./md5.text"


# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


# spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 1000        # 文本分割的阈值


# 检索返回匹配的文档数量
# 说明：列举型查询（"有哪些用例"）已走全量元数据查询不受此限制；
#       此值仅影响语义相似度查询（如"登录功能怎么测"）的返回条数，适当调大避免漏检
similarity_threshold = 10

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# 意图识别专用模型：用轻量模型降低延迟和成本（仅判断查询类型，不需要强推理能力）
# temperature 设为 0 保证判定结果稳定
intent_model_name = "qwen-turbo"

session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }


# ==================== 测试用例相关配置 ====================

# 用例的标准字段名（内部统一使用这些 key）
# dataset 用于按"任务/批次/项目"维度隔离用例，避免不同任务用例混在一起
case_fields = ["case_id", "dataset", "module", "priority", "title", "precondition", "steps", "expected"]

# 字段别名映射表：将各种常见列名/键名统一归一到标准字段
# 支持中英文混用，解析 Excel/CSV/JSON 时自动匹配
field_alias_map = {
    "case_id":      ["case_id", "用例ID", "用例编号", "id", "ID", "用例id"],
    "dataset":      ["dataset", "数据集", "任务", "批次", "项目", "任务名称", "批次名称"],
    "module":       ["module", "模块", "所属模块", "功能模块"],
    "priority":     ["priority", "优先级", "级别", "重要级", "优先级别"],
    "title":        ["title", "标题", "用例标题", "用例名称", "名称"],
    "precondition": ["precondition", "前置条件", "前提条件", "前提"],
    "steps":        ["steps", "测试步骤", "步骤", "操作步骤"],
    "expected":     ["expected", "预期结果", "预期", "期望结果"],
}

# 纯文本用例文件的字段标记前缀（解析时按 "字段名: 值" 格式识别）
text_field_labels = {
    "case_id":      ["用例ID", "用例编号", "case_id"],
    "dataset":      ["数据集", "任务", "批次", "项目", "dataset"],
    "module":       ["模块", "所属模块", "module"],
    "priority":     ["优先级", "级别", "priority"],
    "title":        ["标题", "用例标题", "title"],
    "precondition": ["前置条件", "precondition"],
    "steps":        ["测试步骤", "步骤", "steps"],
    "expected":     ["预期结果", "预期", "expected"],
}

# 默认数据集名称：上传时未指定数据集则归入此默认集
default_dataset = "默认数据集"

# 默认操作人：上传用例时未指定操作人则用此默认值
default_operator = "小曹"

# 支持上传的文件格式
supported_file_types = ["xlsx", "xls", "csv", "json", "txt"]
