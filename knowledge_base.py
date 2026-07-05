"""
测试用例知识库

核心功能：
1. 解析 Excel / CSV / JSON / 纯文本 四种格式的测试用例文件
2. 按 case_id（唯一标识）去重，避免重复入库
3. 每条用例作为一个独立 chunk 存入 Chroma 向量库，元数据保存结构化字段
4. 支持按 case_id / 模块 / 文件名 删除
"""
import ssl_fix  # noqa: F401 — 必须在任何会触发 SSL 的 import 之前
import os
import io
import json
import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from datetime import datetime


# ==================== 字段映射工具 ====================

def _normalize_field_name(raw_name: str):
    """将原始列名/键名归一到标准字段名，匹配不到则返回 None"""
    raw_name = str(raw_name).strip()
    for std_field, aliases in config.field_alias_map.items():
        if raw_name in aliases:
            return std_field
    return None


def _normalize_case(raw_case: dict):
    """将一条原始用例字典归一为标准字段结构，缺省字段填空字符串"""
    case = {}
    for field in config.case_fields:
        case[field] = ""
    for k, v in raw_case.items():
        std_field = _normalize_field_name(k)
        if std_field:
            # 处理 NaN / None 等异常值
            val = v if v is not None else ""
            if isinstance(val, float) and val != val:  # NaN 判断
                val = ""
            case[std_field] = str(val).strip()
    return case


def _build_case_text(case: dict) -> str:
    """将一条用例拼成用于向量化与检索的纯文本"""
    return (
        f"用例ID: {case['case_id']}\n"
        f"数据集: {case['dataset']}\n"
        f"模块: {case['module']}\n"
        f"优先级: {case['priority']}\n"
        f"标题: {case['title']}\n"
        f"前置条件: {case['precondition']}\n"
        f"测试步骤: {case['steps']}\n"
        f"预期结果: {case['expected']}"
    )


# ==================== 文件解析器 ====================

def parse_excel(file_bytes: bytes) -> list[dict]:
    """解析 Excel 文件，每行转换为一条标准化用例"""
    import pandas as pd
    df = pd.read_excel(io.BytesIO(file_bytes))
    cases = []
    for _, row in df.iterrows():
        raw_case = row.to_dict()
        case = _normalize_case(raw_case)
        cases.append(case)
    return cases


def parse_csv(file_bytes: bytes) -> list[dict]:
    """解析 CSV 文件，每行转换为一条标准化用例"""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(file_bytes))
    cases = []
    for _, row in df.iterrows():
        raw_case = row.to_dict()
        case = _normalize_case(raw_case)
        cases.append(case)
    return cases


def parse_json(file_bytes: bytes) -> list[dict]:
    """解析 JSON 文件，支持两种结构：
       1. 列表格式: [{"case_id": ...}, ...]
       2. 嵌套格式: {"cases": [{"case_id": ...}, ...]}
    """
    data = json.loads(file_bytes.decode("utf-8"))
    # 兼容嵌套结构
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError("JSON 文件应为用例数组或 {\"cases\": [...]} 结构")

    cases = []
    for item in data:
        case = _normalize_case(item)
        cases.append(case)
    return cases


def parse_text(text: str) -> list[dict]:
    """解析纯文本用例文件。

    格式约定：每条用例由若干 "字段名: 值" 行组成，
    不同用例之间用空行或 '---' 分隔。

    示例：
        用例ID: TC_001
        模块: 登录
        优先级: P0
        标题: 用户名密码登录
        前置条件: 用户已注册
        测试步骤: 1.打开登录页 2.输入用户名密码 3.点击登录
        预期结果: 登录成功跳转首页
        ---
        用例ID: TC_002
        模块: 登录
        ...
    """
    # 构建反向映射：标签文本 -> 标准字段名
    label_to_field = {}
    for std_field, labels in config.text_field_labels.items():
        for label in labels:
            label_to_field[label] = std_field

    cases = []
    # 先按 "---" 分割，再按空行分割
    blocks = []
    for chunk in text.split("---"):
        blocks.extend(chunk.split("\n\n"))

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        case = {field: "" for field in config.case_fields}
        has_content = False
        for line in block.splitlines():
            line = line.strip()
            # 同时兼容英文冒号和中文冒号，任一存在即认为是字段行
            if not line or (":" not in line and "：" not in line):
                continue
            # 按第一个冒号分割（优先中文冒号，再英文冒号）
            sep_idx = line.find("：")
            if sep_idx == -1:
                sep_idx = line.find(":")
            label = line[:sep_idx].strip()
            value = line[sep_idx + 1:].strip()
            if label in label_to_field:
                case[label_to_field[label]] = value
                has_content = True
        if has_content:
            cases.append(case)
    return cases


def parse_file(file_bytes: bytes, filename: str) -> list[dict]:
    """根据文件扩展名自动选择解析器，返回标准化用例列表"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("xlsx", "xls"):
        return parse_excel(file_bytes)
    elif ext == "csv":
        return parse_csv(file_bytes)
    elif ext == "json":
        return parse_json(file_bytes)
    elif ext == "txt":
        return parse_text(file_bytes.decode("utf-8"))
    else:
        raise ValueError(f"不支持的文件格式: .{ext}，支持 {config.supported_file_types}")


# ==================== 知识库服务 ====================

class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)

        # Chroma 向量库，用于存储测试用例
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,
        )

    # -------------------- 入库 --------------------

    def _case_id_exists(self, case_id: str, dataset: str = None):
        """检查某 case_id 是否已存在于指定数据集中。

        去重维度：case_id + dataset 联合判断。
        同一数据集内相同 case_id 视为重复；不同数据集间允许相同 case_id 共存。
        若 dataset 为 None，则按旧逻辑仅按 case_id 查询（兼容旧调用）。
        """
        try:
            if dataset:
                # 按 case_id + dataset 联合过滤查询
                result = self.chroma._collection.get(
                    where={"$and": [
                        {"case_id": case_id},
                        {"dataset": dataset},
                    ]}
                )
            else:
                result = self.chroma._collection.get(where={"case_id": case_id})
            return len(result.get("ids", [])) > 0
        except Exception:
            # 某些 Chroma 版本不支持 $and，退回到先查 case_id 再逐条比对 dataset
            try:
                result = self.chroma._collection.get(where={"case_id": case_id})
                if not dataset:
                    return len(result.get("ids", [])) > 0
                # 手动过滤 dataset
                metadatas = result.get("metadatas", [])
                for meta in metadatas:
                    if meta.get("dataset") == dataset:
                        return True
                return False
            except Exception:
                return False

    def upload_cases(self, cases: list[dict], filename: str, dataset=None, operator=None):
        """将解析好的用例列表批量入库。

        去重策略：按 case_id + dataset 联合判断，同一数据集内已存在则跳过，
                  不同数据集间允许相同 case_id 共存。
        数据集策略：若传入 dataset，则统一覆盖每条用例的 dataset 字段；
                    否则保留用例自身解析出的 dataset，仍为空则用默认数据集名。
        操作人策略：若传入 operator 则用传入值；否则用 config.default_operator。
        返回格式: (成功数, 跳过数, 缺少case_id数)
        """
        texts = []
        metadatas = []
        skipped = 0
        missing_id = 0
        inserted = 0

        # 操作人：传入优先，否则用默认值
        op_name = operator if operator else config.default_operator

        for case in cases:
            # case_id 是唯一标识，缺少则跳过并记录
            if not case["case_id"]:
                missing_id += 1
                continue

            # 数据集归属：传入优先 > 用例自带 > 默认数据集
            # 必须在查重之前确定 dataset，因为去重是按 case_id + dataset 联合判断
            if dataset:
                case["dataset"] = dataset
            elif not case["dataset"]:
                case["dataset"] = config.default_dataset

            # 按 case_id + dataset 联合去重
            if self._case_id_exists(case["case_id"], case["dataset"]):
                skipped += 1
                continue

            # 拼接检索文本
            text = _build_case_text(case)
            texts.append(text)

            # 结构化元数据
            metadata = {
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "module": case["module"],
                "priority": case["priority"],
                "title": case["title"],
                "source": filename,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operator": op_name,
            }
            metadatas.append(metadata)
            inserted += 1

        # 批量写入向量库
        if texts:
            self.chroma.add_texts(texts, metadatas=metadatas)

        return inserted, skipped, missing_id

    def upload_file(self, file_bytes: bytes, filename: str, dataset=None, operator=None):
        """解析文件并入库，返回面向用户的操作结果字符串。

        :param dataset: 指定该批次用例归属的数据集名称，用于按任务/批次隔离管理。
        :param operator: 操作人姓名，记录到元数据中便于追溯。为 None 则用默认值。
        """
        try:
            cases = parse_file(file_bytes, filename)
        except Exception as e:
            return f"[失败]文件解析出错: {e}"

        if not cases:
            return "[警告]未从文件中解析到任何用例，请检查文件格式"

        inserted, skipped, missing_id = self.upload_cases(cases, filename, dataset=dataset, operator=operator)

        ds_name = dataset or config.default_dataset
        msg = f"[成功]数据集'{ds_name}' | 共解析 {len(cases)} 条用例，入库 {inserted} 条"
        if skipped:
            msg += f"，跳过(已存在) {skipped} 条"
        if missing_id:
            msg += f"，缺少用例ID {missing_id} 条(未入库)"
        return msg

    # -------------------- 查询 --------------------

    def list_cases(self, dataset_filter=None, module_filter=None, priority_filter=None):
        """列出知识库中的用例，可按数据集/模块/优先级过滤。
        返回 list[dict]，每项含 case_id/dataset/module/priority/title/source/content
        其中 content 为用例完整文本（含前置条件/测试步骤/预期结果）。
        """
        try:
            collection = self.chroma._collection

            # 构建元数据过滤条件
            where = {}
            if dataset_filter:
                where["dataset"] = dataset_filter
            if module_filter:
                where["module"] = module_filter
            if priority_filter:
                where["priority"] = priority_filter

            # 同时取出 metadatas 和 documents，以便展示完整用例内容
            result = collection.get(
                where=where if where else None,
                include=["metadatas", "documents"]
            )

            cases = []
            ids = result.get("ids", [])
            metadatas = result.get("metadatas", [])
            documents = result.get("documents", [])
            for cid, meta, doc in zip(ids, metadatas, documents):
                cases.append({
                    "id": cid,
                    "case_id": meta.get("case_id", ""),
                    "dataset": meta.get("dataset", ""),
                    "module": meta.get("module", ""),
                    "priority": meta.get("priority", ""),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "create_time": meta.get("create_time", ""),
                    "content": doc or "",
                })
            return cases
        except Exception as e:
            print(f"获取用例列表失败: {e}")
            return []

    def get_total_cases(self):
        """获取知识库中用例总数"""
        try:
            return self.chroma._collection.count()
        except Exception:
            return 0

    def get_datasets(self):
        """获取知识库中所有数据集名称（用于 UI 筛选下拉框）"""
        try:
            result = self.chroma._collection.get(include=["metadatas"])
            datasets = set()
            for meta in result.get("metadatas", []):
                ds = meta.get("dataset", "")
                if ds:
                    datasets.add(ds)
            return sorted(datasets)
        except Exception:
            return []

    def get_dataset_stats(self):
        """获取每个数据集的用例数量统计，返回 list[dict]: [{dataset, count}]"""
        try:
            result = self.chroma._collection.get(include=["metadatas"])
            stats = {}
            for meta in result.get("metadatas", []):
                ds = meta.get("dataset", config.default_dataset) or config.default_dataset
                stats[ds] = stats.get(ds, 0) + 1
            # 按用例数降序排列
            return sorted(
                [{"dataset": k, "count": v} for k, v in stats.items()],
                key=lambda x: x["count"], reverse=True
            )
        except Exception:
            return []

    def get_modules(self, dataset_filter=None):
        """获取知识库中所有模块名（可按数据集范围），用于 UI 筛选下拉框"""
        try:
            where = {}
            if dataset_filter:
                where["dataset"] = dataset_filter
            result = self.chroma._collection.get(
                where=where if where else None,
                include=["metadatas"]
            )
            modules = set()
            for meta in result.get("metadatas", []):
                m = meta.get("module", "")
                if m:
                    modules.add(m)
            return sorted(modules)
        except Exception:
            return []

    def get_priorities(self, dataset_filter=None):
        """获取知识库中所有优先级（可按数据集范围），用于 UI 筛选下拉框"""
        try:
            where = {}
            if dataset_filter:
                where["dataset"] = dataset_filter
            result = self.chroma._collection.get(
                where=where if where else None,
                include=["metadatas"]
            )
            priorities = set()
            for meta in result.get("metadatas", []):
                p = meta.get("priority", "")
                if p:
                    priorities.add(p)
            return sorted(priorities)
        except Exception:
            return []

    # -------------------- 删除 --------------------

    def delete_by_case_id(self, case_id: str, dataset: str = None):
        """按用例ID删除单条用例。

        :param dataset: 指定数据集则只删该数据集中的用例；
                        为 None 则删除所有数据集中该 case_id 的用例。
        """
        try:
            if dataset:
                self.chroma._collection.delete(
                    where={"$and": [{"case_id": case_id}, {"dataset": dataset}]}
                )
            else:
                self.chroma._collection.delete(where={"case_id": case_id})
            scope = f" (数据集: {dataset})" if dataset else ""
            return f"[成功]已删除用例: {case_id}{scope}"
        except Exception:
            # 某些 Chroma 版本不支持 $and，退回到先查再删
            try:
                result = self.chroma._collection.get(where={"case_id": case_id})
                ids_to_delete = []
                for cid, meta in zip(result.get("ids", []), result.get("metadatas", [])):
                    if dataset is None or meta.get("dataset") == dataset:
                        ids_to_delete.append(cid)
                if ids_to_delete:
                    self.chroma._collection.delete(ids=ids_to_delete)
                scope = f" (数据集: {dataset})" if dataset else ""
                return f"[成功]已删除用例: {case_id}{scope}"
            except Exception as e:
                return f"[失败]删除用例 {case_id} 时出错: {e}"

    def delete_by_module(self, module: str):
        """按模块删除该模块下所有用例"""
        try:
            self.chroma._collection.delete(where={"module": module})
            return f"[成功]已删除模块 '{module}' 下的所有用例"
        except Exception as e:
            return f"[失败]删除模块 {module} 时出错: {e}"

    def delete_by_dataset(self, dataset: str):
        """按数据集删除该数据集下所有用例（整批清理某个任务/批次的用例）"""
        try:
            self.chroma._collection.delete(where={"dataset": dataset})
            return f"[成功]已删除数据集 '{dataset}' 下的所有用例"
        except Exception as e:
            return f"[失败]删除数据集 {dataset} 时出错: {e}"

    def delete_by_file(self, filename: str):
        """按来源文件名删除该文件导入的所有用例"""
        try:
            self.chroma._collection.delete(where={"source": filename})
            return f"[成功]已删除文件 {filename} 的所有用例"
        except Exception as e:
            return f"[失败]删除文件 {filename} 时出错: {e}"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    # 简单测试：用文本格式入库一条用例
    sample = "用例ID: TC_TEST_001\n模块: 测试\n优先级: P1\n标题: 临时测试用例\n前置条件: 无\n测试步骤: 无\n预期结果: 无"
    cases = parse_text(sample)
    print("解析结果:", cases)
    result = service.upload_cases(cases, "testfile")
    print("入库结果:", result)
    print("用例列表:", service.list_cases())
