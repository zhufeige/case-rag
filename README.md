# 测试用例 RAG 智能问答平台

基于 LangChain + ChromaDB + 阿里百炼（通义千问）构建的测试用例管理与智能问答系统。把测试用例文件导入向量知识库后，可通过自然语言对话检索、列举、分析用例，支持按数据集/模块/优先级多维度过滤，适用于回归测试、专项测试等场景的用例管理与查询。

---

## 功能特性

- **多格式用例导入**：支持 Excel（xlsx/xls）、CSV、JSON、TXT 四种格式，自动识别字段并归一化
- **数据集隔离管理**：用例按"任务/批次"维度打标签（如 `V1.0回归测试`、`支付模块专项`），不同任务互不混淆，去重按 `case_id + dataset` 联合判断
- **查询意图双路由**：列举型问题（"有哪些用例"）走全量元数据查询，避免 top-k 漏检；语义型问题（"登录功能怎么测"）走向量相似度检索
- **元数据过滤检索**：对话与列表均支持按 数据集 / 模块 / 优先级 三维度过滤，模块和优先级列表随数据集动态变化
- **多会话聊天历史**：支持新建、切换、删除会话，历史持久化到本地文件，流式输出实时渲染
- **用例全生命周期管理**：上传预览、入库去重、列表查看、按 数据集/用例ID/模块/来源文件 四种方式删除

---

## 技术栈

| 层级 | 选型 |
|------|------|
| Web 前端 | Streamlit |
| 编排框架 | LangChain（LCEL chain + RunnableWithMessageHistory） |
| 向量数据库 | ChromaDB（本地持久化） |
| Embedding 模型 | DashScope `text-embedding-v4` |
| 对话模型 | 通义千问 `qwen3-max` |
| 数据处理 | pandas、openpyxl |
| 运行环境 | Python 3.11，conda 虚拟环境 |

---

## 项目结构

```
case-rag/
├── app.py                    # 主入口：侧边栏会话管理 + 对话Tab + 用例库管理Tab
├── rag.py                    # RAG 服务：查询意图路由、链式编排、流式输出
├── knowledge_base.py         # 知识库服务：多格式解析、去重入库、查询、删除
├── vector_stores.py          # Chroma 向量库封装，提供 retriever
├── config_data.py            # 全局配置：模型名、字段别名映射、默认值
├── file_history_store.py     # 基于文件的会话聊天历史
├── ssl_fix.py                # Windows OpenSSL 3.5.x 证书库 ASN1 错误修复
├── app_qa.py                 # 早期单页问答版本（legacy）
├── app_file_uploader.py      # 早期单页上传版本（legacy）
├── sample_cases.txt          # 示例用例文件
├── requirements.txt          # Python 依赖清单
├── .env.example              # 环境变量配置模板
└── .gitignore
```

运行时生成的目录（已加入 `.gitignore`，不上传）：

```
chroma_db/        # 向量数据库持久化目录
chat_history/     # 聊天会话历史
md5.text          # 文件去重哈希记录
__pycache__/      # Python 编译缓存
```

---

## 快速开始

### 1. 环境准备

需要 Python 3.10+，推荐使用 conda 创建独立环境：

```bash
conda create -n RAG-main python=3.11
conda activate RAG-main
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

本项目使用阿里百炼 DashScope 提供的 embedding 与对话模型，需先获取 API Key。

获取地址：[阿里百炼控制台](https://bailian.console.aliyun.com/) → 模型广场 → API Key

复制 `.env.example` 为 `.env`，填入真实 Key：

```bash
cp .env.example .env
```

```
DASHSCOPE_API_KEY=sk-your-api-key-here
```

启动前需让环境变量生效（PowerShell）：

```powershell
# 方式一：从 .env 文件加载（需 python-dotenv，或手动设置）
$env:DASHSCOPE_API_KEY="sk-your-api-key-here"

# 方式二：直接在系统环境变量中配置
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

---

## 使用指南

### 上传测试用例

在「📁 用例库管理」Tab 中：

1. 填写**数据集名称**（如 `V1.0回归测试`），或从下拉框选择已有数据集。未指定则归入"默认数据集"
2. 填写**操作人**（默认"小曹"），便于后续追溯
3. 选择文件上传，系统自动解析并预览前 10 条用例
4. 确认无误后点击「✅ 确认上传」，用例入库并自动去重

### 智能问答

在「💬 对话」Tab 中直接提问，例如：

- `登录模块有哪些用例？` —— 列举型查询，返回该模块全部用例
- `数据集3333有什么用例？` —— 自动提取数据集名，列举该数据集用例
- `支付功能怎么测？` —— 语义检索，返回最相关的用例
- `密码错误登录失败的预期结果是什么？` —— 语义检索定位具体用例

### 对话检索过滤

点击「⚙️ 对话检索范围」可设置过滤条件：

- 按数据集过滤：对话只检索指定任务/批次的用例
- 按模块过滤：进一步限定到某个功能模块
- 按优先级过滤：只检索 P0/P1/P2 等指定优先级的用例

点击「✅ 应用过滤」后生效，「↩️ 重置过滤」可清除限制。

### 用例管理

在用例库管理 Tab 下方可：

- 查看数据集用例分布统计
- 按数据集/模块/优先级筛选查看用例列表
- 按数据集 / 用例ID / 模块 / 来源文件 四种维度删除用例

---

## 数据模型

每条测试用例包含 8 个标准字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| `case_id` | 用例唯一标识（同数据集内去重依据） | `TC_001` |
| `dataset` | 数据集（任务/批次隔离维度） | `V1.0回归测试` |
| `module` | 功能模块 | `登录` |
| `priority` | 优先级 | `P0` |
| `title` | 用例标题 | `用户名密码正常登录` |
| `precondition` | 前置条件 | `用户已注册且账号状态正常` |
| `steps` | 测试步骤 | `1.打开登录页 2.输入用户名密码 3.点击登录` |
| `expected` | 预期结果 | `登录成功，跳转首页` |

入库时额外记录元数据：`source`（来源文件名）、`create_time`（入库时间）、`operator`（操作人）。

---

## 文件格式说明

### TXT 格式

每条用例由若干 `字段名: 值` 行组成，不同用例之间用空行或 `---` 分隔。详见 `sample_cases.txt`：

```
数据集: V1.0回归测试
用例ID: TC_001
模块: 登录
优先级: P0
标题: 用户名密码正常登录
前置条件: 用户已注册且账号状态正常
测试步骤: 1.打开登录页面 2.输入正确的用户名 3.输入正确的密码 4.点击登录按钮
预期结果: 登录成功，跳转到首页，显示用户昵称
---
用例ID: TC_002
...
```

### Excel / CSV 格式

第一行为表头，支持中英文混用。系统通过 `config_data.py` 中的 `field_alias_map` 自动归一化列名：

| 标准字段 | 可识别的列名 |
|----------|--------------|
| case_id | case_id, 用例ID, 用例编号, id, ID |
| dataset | dataset, 数据集, 任务, 批次, 项目 |
| module | module, 模块, 所属模块, 功能模块 |
| priority | priority, 优先级, 级别, 重要级 |
| title | title, 标题, 用例标题, 用例名称, 名称 |
| precondition | precondition, 前置条件, 前提条件 |
| steps | steps, 测试步骤, 步骤, 操作步骤 |
| expected | expected, 预期结果, 预期, 期望结果 |

### JSON 格式

支持两种结构：

```json
[
  {"用例ID": "TC_001", "模块": "登录", "标题": "..."}
]
```

或嵌套结构：

```json
{
  "cases": [
    {"case_id": "TC_001", "module": "登录"}
  ]
}
```

---

## 配置说明

所有可配置项集中在 `config_data.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding_model_name` | `text-embedding-v4` | DashScope embedding 模型 |
| `chat_model_name` | `qwen3-max` | 通义千问对话模型 |
| `collection_name` | `rag` | Chroma collection 名称 |
| `persist_directory` | `./chroma_db` | 向量库持久化目录 |
| `similarity_threshold` | `10` | 语义检索返回的 top-k 条数 |
| `chunk_size` | `1000` | 文本分割块大小 |
| `default_dataset` | `默认数据集` | 未指定数据集时的默认值 |
| `default_operator` | `小曹` | 未指定操作人时的默认值 |
| `supported_file_types` | `xlsx/xls/csv/json/txt` | 支持上传的文件格式 |

---

## 已知问题与修复

### Windows OpenSSL 证书库错误

在 conda 环境（Python 3.11 + OpenSSL 3.5.x）下运行时，可能遇到：

```
ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data (_ssl.c:4057)
```

根因是 OpenSSL 3.5.x 解析 Windows 证书库的 ASN1 格式失败，导致 `ssl.create_default_context()` 崩溃，进而让 aiohttp / dashscope 等库无法导入。

项目已内置 `ssl_fix.py` 修复模块，会在所有入口文件顶部自动加载，拦截证书库解析异常并回退到 certifi 提供的 CA 证书。正常情况下无需手动处理。

---

## 架构说明

系统数据流如下：

```
用户上传文件
    │
    ▼
parse_file() 按扩展名选择解析器
    │  (Excel/CSV/JSON/TXT)
    ▼
_normalize_case() 字段归一化
    │
    ▼
upload_cases() 按 case_id+dataset 去重
    │
    ▼
_build_case_text() 拼接检索文本
    │
    ▼
Chroma 向量库（文本向量化 + 元数据存储）
    │
    ▼
用户提问
    │
    ▼
get_context() 查询意图路由
    ├── 列举型 → _get_all_cases_as_docs() 全量元数据查询
    └── 语义型 → retriever.invoke() 向量相似度 top-k 检索
    │
    ▼
Prompt 模板拼接（参考资料 + 对话历史）
    │
    ▼
qwen3-max 流式生成回答
```

查询意图路由是本系统的核心设计：列举型查询（"有哪些用例""列出全部"）走 Chroma 的 `collection.get()` 全量元数据查询，避免向量检索的 top-k 限制漏掉用例；语义型查询走 `as_retriever()` 的向量相似度检索，返回最相关的若干条。路由通过 `rag.py` 中的 `is_list_query()` 关键词匹配实现。

### 路由示例

假设知识库中已导入 `sample_cases.txt` 的 7 条用例（TC_001~TC_007，分属 `V1.0回归测试` 和 `V2.0新功能测试` 两个数据集），以下展示两类查询的不同路由行为：

**示例 1：列举型查询**

```
用户提问：登录模块有哪些用例？
```

- `is_list_query()` 命中关键词"有哪些用例" → 判定为列举型
- 走 `_get_all_cases_as_docs()`，调用 `collection.get(where={"module": "登录"})` 全量查询
- 返回该模块下全部 3 条用例（TC_001、TC_002、TC_003），不受 `similarity_threshold=10` 的 top-k 限制
- 若 3 条都返回，就不会漏掉任何一条

```
用户提问：数据集V1.0回归测试包含哪些用例？
```

- 命中"包含哪些" → 列举型
- `extract_dataset_from_query()` 正则提取出数据集名 `V1.0回归测试`
- 走 `collection.get(where={"dataset": "V1.0回归测试"})` 全量查询
- 返回该数据集下全部 5 条用例（TC_001~TC_005）

**示例 2：语义型查询**

```
用户提问：密码错误登录失败的预期结果是什么？
```

- 未命中列举关键词 → 判定为语义型
- 走 `retriever.invoke("密码错误登录失败的预期结果是什么")`
- 向量相似度检索，从全库返回最相关的 top-10 条（此处实际命中 TC_002）
- LLM 根据检索到的用例文本生成回答：`登录失败，提示"用户名或密码错误"`

```
用户提问：支付功能怎么测？
```

- 未命中列举关键词 → 语义型
- 向量检索返回与"支付功能测试"语义最相关的用例（TC_004 余额支付成功、TC_005 余额不足失败）
- LLM 综合这两条用例的步骤和预期结果，归纳支付模块的测试要点

**两类路由对比**

| 维度 | 列举型查询 | 语义型查询 |
|------|-----------|-----------|
| 触发条件 | 命中"有哪些用例""列出全部"等关键词 | 未命中列举关键词的普通提问 |
| 检索方式 | `collection.get()` 元数据全量查询 | `retriever.invoke()` 向量相似度 top-k |
| 返回条数 | 命中条件的全部用例，无数量上限 | 最多 `similarity_threshold`（默认 10）条 |
| 典型场景 | "登录模块有哪些用例""数据集X有多少用例" | "密码错误登录怎么测""支付流程的预期结果" |
| 数据集提取 | 支持 `extract_dataset_from_query()` 从提问中提取数据集名 | 依赖对话过滤条件或检索全库 |

---

## License

本项目仅供学习交流使用。
