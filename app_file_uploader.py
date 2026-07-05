"""
基于Streamlit完成WEB网页上传服务

pip install streamlit

Streamlit：当WEB页面元素发生变化，则代码重新执行一遍
"""
import ssl_fix  # noqa: F401 — 必须在任何会触发 SSL 的 import 之前
import time

import streamlit as st
from knowledge_base import KnowledgeBaseService
import config_data as config

# 添加网页标题
st.title("知识库更新服务")

# file_uploader
# 支持的文件格式统一从 config 读取，与主程序 app.py 保持一致（Excel/CSV/JSON/TXT）
uploader_file = st.file_uploader(
    f"请上传测试用例文件（支持 {('/'.join(config.supported_file_types)).upper()}）",
    type=config.supported_file_types,
    accept_multiple_files=False,    # False表示仅接受一个文件的上传
)

# session_state就是一个字典
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploader_file is not None:
    # 提取文件的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024    # KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    # 直接取原始字节交给 upload_file，由其内部按扩展名自动选择解析器
    # 这样可统一支持 Excel/CSV/JSON/TXT，无需在此手动 decode
    file_bytes = uploader_file.getvalue()

    with st.spinner("载入知识库中。。。"):       # 在spinner内的代码执行过程中，会有一个转圈动画
        time.sleep(1)
        # 调用现行的 upload_file 方法（已替代旧的 upload_by_str）
        # 未指定 dataset 时会归入 config.default_dataset
        result = st.session_state["service"].upload_file(file_bytes, file_name)
        st.write(result)




