"""
通用RAG平台 - 测试用例管理版
左侧聊天历史 + 智能问答（支持过滤检索）+ 测试用例库管理
"""
import time
import streamlit as st
from knowledge_base import KnowledgeBaseService, parse_file
from rag import RagService
from file_history_store import list_sessions
import config_data as config

# ==================== 页面配置 ====================
st.set_page_config(page_title="测试用例RAG平台", page_icon="🧪", layout="wide")

# ==================== 初始化 session_state ====================
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

if "kb_service" not in st.session_state:
    st.session_state["kb_service"] = KnowledgeBaseService()

if "current_session" not in st.session_state:
    st.session_state["current_session"] = config.session_config["configurable"]["session_id"]

if "session_messages" not in st.session_state:
    st.session_state["session_messages"] = {}

if st.session_state["current_session"] not in st.session_state["session_messages"]:
    st.session_state["session_messages"][st.session_state["current_session"]] = [
        {"role": "assistant", "content": "你好，我是测试用例助手，有什么可以帮助你？"}
    ]

# 对话检索过滤条件（默认不过滤）
if "chat_module_filter" not in st.session_state:
    st.session_state["chat_module_filter"] = None
if "chat_priority_filter" not in st.session_state:
    st.session_state["chat_priority_filter"] = None
if "chat_dataset_filter" not in st.session_state:
    st.session_state["chat_dataset_filter"] = None


def get_current_messages():
    """获取当前会话的消息列表"""
    sid = st.session_state["current_session"]
    if sid not in st.session_state["session_messages"]:
        st.session_state["session_messages"][sid] = [
            {"role": "assistant", "content": "你好，我是测试用例助手，有什么可以帮助你？"}
        ]
    return st.session_state["session_messages"][sid]


# ==================== 左侧边栏：聊天历史 ====================
with st.sidebar:
    st.title("🧪 用例RAG平台")

    # 新建会话
    col_new, col_refresh = st.columns([3, 1])
    with col_new:
        if st.button("➕ 新建会话", use_container_width=True):
            import uuid
            new_id = f"user_{uuid.uuid4().hex[:8]}"
            st.session_state["current_session"] = new_id
            st.session_state["session_messages"][new_id] = [
                {"role": "assistant", "content": "你好，我是测试用例助手，有什么可以帮助你？"}
            ]
            st.rerun()
    with col_refresh:
        if st.button("🔄", help="刷新会话列表"):
            st.rerun()

    st.divider()
    st.caption("📋 聊天历史")

    # 列出所有会话
    sessions = list_sessions()
    file_session_ids = {s["session_id"] for s in sessions}
    for sid in st.session_state["session_messages"]:
        if sid not in file_session_ids:
            msgs = st.session_state["session_messages"][sid]
            preview = ""
            for m in msgs:
                if m["role"] == "user":
                    preview = m["content"][:25]
                    break
            sessions.append({
                "session_id": sid,
                "message_count": len(msgs),
                "preview": preview or "(新会话)",
            })

    if not sessions:
        st.caption("暂无聊天记录")
    else:
        for s in sessions:
            sid = s["session_id"]
            is_active = sid == st.session_state["current_session"]
            btn_label = f"{'🔹 ' if is_active else '  '}{s['preview']}"
            if st.button(
                btn_label,
                key=f"session_{sid}",
                use_container_width=True,
                help=f"会话: {sid} | 消息数: {s['message_count']}",
                type="primary" if is_active else "secondary",
            ):
                if sid != st.session_state["current_session"]:
                    st.session_state["current_session"] = sid
                    if sid not in st.session_state["session_messages"]:
                        st.session_state["session_messages"][sid] = [
                            {"role": "assistant", "content": "你好，我是测试用例助手，有什么可以帮助你？"}
                        ]
                    st.rerun()

    st.divider()
    # 当前会话删除按钮
    cur = st.session_state["current_session"]
    if st.button("🗑️ 删除当前会话", use_container_width=True):
        import os
        history_path = os.path.join("./chat_history", cur)
        if os.path.exists(history_path):
            os.remove(history_path)
        st.session_state["session_messages"].pop(cur, None)
        remaining = [s for s in sessions if s["session_id"] != cur]
        if remaining:
            st.session_state["current_session"] = remaining[0]["session_id"]
        else:
            new_id = "user_001"
            st.session_state["current_session"] = new_id
            st.session_state["session_messages"][new_id] = [
                {"role": "assistant", "content": "你好，我是测试用例助手，有什么可以帮助你？"}
            ]
        st.rerun()


# ==================== 主区域 ====================
st.title("💬 智能问答（测试用例）")
sid = st.session_state["current_session"]
st.caption(f"当前会话: `{sid}`")

tab_qa, tab_kb = st.tabs(["💬 对话", "📁 用例库管理"])

# ==================== 对话 Tab ====================
with tab_qa:
    # 对话检索范围过滤：可选择只检索某个数据集/模块/优先级的用例
    with st.expander("⚙️ 对话检索范围（可选过滤，数据集用于隔离不同任务的用例）"):
        kb = st.session_state["kb_service"]
        col_ds, col_mod, col_pri = st.columns(3)
        with col_ds:
            datasets = kb.get_datasets()
            sel_dataset = st.selectbox(
                "按数据集过滤检索",
                ["全部"] + datasets,
                index=0,
                help="选择数据集后，对话只检索该任务/批次的用例，避免不同任务用例混淆"
            )
        with col_mod:
            # 模块列表跟随选中的数据集动态变化
            ds_for_mod = sel_dataset if sel_dataset != "全部" else None
            modules = kb.get_modules(dataset_filter=ds_for_mod)
            sel_module = st.selectbox(
                "按模块过滤检索",
                ["全部"] + modules,
                index=0,
                help="选择模块后，对话只会检索该模块下的用例"
            )
        with col_pri:
            ds_for_pri = sel_dataset if sel_dataset != "全部" else None
            priorities = kb.get_priorities(dataset_filter=ds_for_pri)
            sel_priority = st.selectbox(
                "按优先级过滤检索",
                ["全部"] + priorities,
                index=0,
                help="选择优先级后，对话只会检索该优先级的用例"
            )

        # 点击应用过滤时重建 RAG 链
        col_apply, col_reset = st.columns(2)
        with col_apply:
            if st.button("✅ 应用过滤"):
                md_filter = {}
                if sel_dataset != "全部":
                    md_filter["dataset"] = sel_dataset
                if sel_module != "全部":
                    md_filter["module"] = sel_module
                if sel_priority != "全部":
                    md_filter["priority"] = sel_priority
                st.session_state["rag"] = RagService(metadata_filter=md_filter if md_filter else None)
                # 记录当前过滤条件，供刷新按钮复用
                st.session_state["chat_module_filter"] = sel_module if sel_module != "全部" else None
                st.session_state["chat_priority_filter"] = sel_priority if sel_priority != "全部" else None
                st.session_state["chat_dataset_filter"] = sel_dataset if sel_dataset != "全部" else None
                st.success("已应用过滤，检索范围已更新")
        with col_reset:
            if st.button("↩️ 重置过滤"):
                st.session_state["rag"] = RagService()
                st.session_state["chat_module_filter"] = None
                st.session_state["chat_priority_filter"] = None
                st.session_state["chat_dataset_filter"] = None
                st.success("已重置，检索全部用例")

    messages = get_current_messages()

    # 消息展示区域（唯一容器，历史消息 + 流式回复都在这里渲染）
    chat_container = st.container(height=450, border=True)
    with chat_container:
        for msg in messages:
            with st.chat_message(msg["role"]):
                # 用 markdown 渲染，支持 AI 回复中的表格、列表等格式
                st.markdown(msg["content"])

    # 底部按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("🔄 刷新知识库", help="上传新用例后刷新检索"):
            # 保留当前过滤条件重建（含数据集维度）
            md_filter = {}
            if st.session_state.get("chat_dataset_filter"):
                md_filter["dataset"] = st.session_state["chat_dataset_filter"]
            if st.session_state.get("chat_module_filter"):
                md_filter["module"] = st.session_state["chat_module_filter"]
            if st.session_state.get("chat_priority_filter"):
                md_filter["priority"] = st.session_state["chat_priority_filter"]
            st.session_state["rag"] = RagService(metadata_filter=md_filter if md_filter else None)
            st.success("已刷新")
    with col_btn2:
        if st.button("🧹 清空对话"):
            st.session_state["session_messages"][st.session_state["current_session"]] = [
                {"role": "assistant", "content": "你好，我是测试用例助手，有什么可以帮助你？"}
            ]
            st.rerun()

    # 输入框
    st.divider()
    if prompt := st.chat_input("请输入你的问题，如：登录模块有哪些用例？"):
        # 1. 用户消息立即加入历史
        messages.append({"role": "user", "content": prompt})
        st.session_state["session_messages"][st.session_state["current_session"]] = messages

        # 2. 在同一个容器内渲染：用户消息气泡 + 流式 AI 回复
        #    不再另开容器，避免出现两个消息界面
        with chat_container:
            # 渲染刚加入的用户消息
            with st.chat_message("user"):
                st.markdown(prompt)

            # 流式输出 AI 回复：边生成边显示
            with st.chat_message("assistant"):
                with st.spinner("AI思考中..."):
                    session_config = {
                        "configurable": {"session_id": st.session_state["current_session"]}
                    }
                    # write_stream 会实时把每个 chunk 追加显示到页面上
                    full_response = st.write_stream(
                        st.session_state["rag"].chain.stream(
                            {"input": prompt}, session_config
                        )
                    )

        # 3. 流式结束后，把完整回复存入历史，下次进入页面可正常展示
        messages.append({"role": "assistant", "content": full_response})
        st.session_state["session_messages"][st.session_state["current_session"]] = messages
        st.rerun()

# ==================== 用例库管理 Tab ====================
with tab_kb:
    st.subheader("📤 上传测试用例文件")

    # 上传前先指定数据集归属，实现按任务/批次隔离管理
    existing_datasets = st.session_state["kb_service"].get_datasets()
    col_ds_new, col_ds_exist, col_op = st.columns([2, 2, 1])
    with col_ds_new:
        upload_dataset = st.text_input(
            "数据集名称（本次上传的用例将归入此数据集）",
            value="",
            placeholder="如：V1.0回归测试 / 支付模块专项 / 202607迭代",
            help="给本批用例打上数据集标签，便于按任务隔离检索与管理。可填新名称或选择已有数据集。"
        )
    with col_ds_exist:
        if existing_datasets:
            pick_exist = st.selectbox(
                "或选择已有数据集（覆盖上方输入）",
                ["(不选择，使用上方输入)"] + existing_datasets,
                index=0
            )
            if pick_exist != "(不选择，使用上方输入)":
                upload_dataset = pick_exist
    with col_op:
        # 操作人输入框，默认值为 config.default_operator，可按次修改
        upload_operator = st.text_input(
            "操作人",
            value=config.default_operator,
            help="上传用例的操作人姓名，记录到元数据便于追溯"
        )

    # 支持多种格式上传
    uploader_files = st.file_uploader(
        "选择测试用例文件（支持 Excel/CSV/JSON/TXT）",
        type=config.supported_file_types,
        accept_multiple_files=True,
        key="kb_uploader"
    )

    if uploader_files:
        # 最终数据集名称：未指定则用默认数据集
        final_dataset = upload_dataset.strip() if upload_dataset and upload_dataset.strip() else config.default_dataset
        # 最终操作人：未指定则用默认操作人
        final_operator = upload_operator.strip() if upload_operator and upload_operator.strip() else config.default_operator

        for uploader_file in uploader_files:
            file_name = uploader_file.name
            file_size = uploader_file.size / 1024

            with st.expander(f"📄 {file_name} ({file_size:.2f} KB) → 数据集: {final_dataset} | 操作人: {final_operator}", expanded=True):
                try:
                    file_bytes = uploader_file.getvalue()
                    # 先解析预览，不入库
                    preview_cases = parse_file(file_bytes, file_name)

                    if not preview_cases:
                        st.warning("未解析到任何用例，请检查文件格式")
                        continue

                    # 以表格形式预览解析结果（展示全部字段）
                    st.caption(f"解析到 {len(preview_cases)} 条用例，预览前 10 条：")
                    preview_df_data = []
                    for c in preview_cases[:10]:
                        preview_df_data.append({
                            "用例ID": c["case_id"],
                            "数据集": c["dataset"],
                            "模块": c["module"],
                            "优先级": c["priority"],
                            "标题": c["title"],
                            "前置条件": c["precondition"],
                            "测试步骤": c["steps"],
                            "预期结果": c["expected"],
                        })
                    st.dataframe(preview_df_data, use_container_width=True, hide_index=True)

                    if st.button(f"✅ 确认上传 - {file_name}", key=f"btn_{file_name}"):
                        with st.spinner(f"正在将 {file_name} 中的用例加载到数据集'{final_dataset}'..."):
                            result = st.session_state["kb_service"].upload_file(
                                file_bytes, file_name, dataset=final_dataset, operator=final_operator
                            )
                            if "成功" in result:
                                st.success(result)
                            elif "跳过" in result or "警告" in result:
                                st.info(result)
                            else:
                                st.warning(result)
                            time.sleep(1)
                            st.rerun()

                except Exception as e:
                    st.error(f"文件处理失败: {e}")

    st.divider()
    st.subheader("📋 已入库测试用例")

    # 数据集统计概览：直观展示各任务/批次的用例分布
    ds_stats = st.session_state["kb_service"].get_dataset_stats()
    if ds_stats:
        st.caption("📊 数据集用例分布")
        stat_cols = st.columns(min(len(ds_stats), 5))
        for i, stat in enumerate(ds_stats[:5]):
            with stat_cols[i]:
                st.metric(stat["dataset"], f"{stat['count']} 条")
        if len(ds_stats) > 5:
            st.caption(f"…共 {len(ds_stats)} 个数据集")

    st.divider()

    # 刷新与过滤（数据集为最高维度，模块/优先级跟随数据集变化）
    col_ref, col_ds, col_mod, col_pri = st.columns([1, 2, 2, 2])
    with col_ref:
        if st.button("🔄 刷新列表"):
            st.rerun()
    with col_ds:
        all_datasets = st.session_state["kb_service"].get_datasets()
        filter_dataset = st.selectbox("按数据集筛选", ["全部"] + all_datasets, key="list_dataset")
    with col_mod:
        # 模块列表跟随数据集筛选动态变化
        ds_for_list_mod = filter_dataset if filter_dataset != "全部" else None
        all_modules = st.session_state["kb_service"].get_modules(dataset_filter=ds_for_list_mod)
        filter_module = st.selectbox("按模块筛选", ["全部"] + all_modules, key="list_module")
    with col_pri:
        ds_for_list_pri = filter_dataset if filter_dataset != "全部" else None
        all_priorities = st.session_state["kb_service"].get_priorities(dataset_filter=ds_for_list_pri)
        filter_priority = st.selectbox("按优先级筛选", ["全部"] + all_priorities, key="list_priority")

    # 获取用例列表（含数据集过滤）
    df_ = filter_dataset if filter_dataset != "全部" else None
    mf = filter_module if filter_module != "全部" else None
    pf = filter_priority if filter_priority != "全部" else None
    cases = st.session_state["kb_service"].list_cases(dataset_filter=df_, module_filter=mf, priority_filter=pf)
    total = st.session_state["kb_service"].get_total_cases()

    if not cases:
        st.info("知识库为空或无匹配用例，请上传文件")
    else:
        st.caption(f"当前筛选: {len(cases)} 条 | 知识库总计: {total} 条用例")

        # 以表格展示用例列表（含数据集列与完整用例内容）
        table_data = []
        for c in cases:
            # 从 content 中提取前置条件/测试步骤/预期结果，便于分列展示
            # 注意：content 中可能是英文冒号或中文冒号，两种都要兼容
            content = c.get("content", "")
            precondition = ""
            steps = ""
            expected = ""

            def _extract_field(text, label):
                """从文本中提取指定标签后的值，兼容中英文冒号"""
                for line in text.splitlines():
                    line = line.strip()
                    # 优先匹配中文冒号
                    if line.startswith(label + "："):
                        return line[len(label + "："):].strip()
                    if line.startswith(label + ":"):
                        return line[len(label + ":"):].strip()
                return ""

            precondition = _extract_field(content, "前置条件")
            steps = _extract_field(content, "测试步骤")
            expected = _extract_field(content, "预期结果")
            table_data.append({
                "用例ID": c["case_id"],
                "数据集": c["dataset"],
                "模块": c["module"],
                "优先级": c["priority"],
                "标题": c["title"],
                "前置条件": precondition,
                "测试步骤": steps,
                "预期结果": expected,
                "来源文件": c["source"],
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # 删除操作
        st.divider()
        st.subheader("🗑️ 删除用例")
        del_tab1, del_tab2, del_tab3, del_tab4 = st.tabs(["按数据集删除", "按用例ID删除", "按模块删除", "按来源文件删除"])

        # 按数据集删除：一键清理整个任务/批次的用例
        with del_tab1:
            if all_datasets:
                sel_ds = st.selectbox("选择数据集", all_datasets, key="del_dataset")
                # 显示该数据集的用例数以作删除前确认
                ds_case_count = sum(1 for c in cases if c["dataset"] == sel_ds) if df_ is None else None
                st.warning(f"⚠️ 此操作将删除数据集 '{sel_ds}' 下的所有用例，不可恢复！")
                if st.button("确认删除该数据集所有用例"):
                    result = st.session_state["kb_service"].delete_by_dataset(sel_ds)
                    if "成功" in result:
                        st.success(result)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(result)
            else:
                st.caption("暂无数据集数据")

        with del_tab2:
            case_ids = [c["case_id"] for c in cases if c["case_id"]]
            sel_cid = st.selectbox("选择用例ID", case_ids, key="del_case_id")
            if st.button("删除该用例"):
                # 若当前筛选了数据集，只删该数据集内的用例，避免误删其他数据集
                result = st.session_state["kb_service"].delete_by_case_id(sel_cid, dataset=df_)
                if "成功" in result:
                    st.success(result)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(result)

        with del_tab3:
            if all_modules:
                sel_mod = st.selectbox("选择模块", all_modules, key="del_module")
                if st.button("删除该模块所有用例"):
                    result = st.session_state["kb_service"].delete_by_module(sel_mod)
                    if "成功" in result:
                        st.success(result)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(result)
            else:
                st.caption("暂无模块数据")

        with del_tab4:
            source_files = sorted({c["source"] for c in cases if c["source"]})
            if source_files:
                sel_file = st.selectbox("选择来源文件", source_files, key="del_file")
                if st.button("删除该文件所有用例"):
                    result = st.session_state["kb_service"].delete_by_file(sel_file)
                    if "成功" in result:
                        st.success(result)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(result)
            else:
                st.caption("暂无来源文件数据")
