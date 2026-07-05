import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")


def list_sessions(storage_path="./chat_history"):
    """列出所有会话ID及其基本信息"""
    sessions = []
    if not os.path.exists(storage_path):
        return sessions
    
    for filename in os.listdir(storage_path):
        file_path = os.path.join(storage_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    messages_data = json.load(f)
                
                # 取第一条用户消息作为会话预览
                preview = ""
                message_count = len(messages_data)
                for msg in messages_data:
                    if msg.get("type") == "human":
                        preview = msg.get("data", {}).get("content", "")[:30]
                        break
                
                sessions.append({
                    "session_id": filename,
                    "message_count": message_count,
                    "preview": preview or "(空会话)",
                })
            except Exception:
                sessions.append({
                    "session_id": filename,
                    "message_count": 0,
                    "preview": "(读取失败)",
                })
    
    # 按文件名排序
    sessions.sort(key=lambda x: x["session_id"])
    return sessions


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, self.session_id)

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)
        all_messages.extend(messages)

        new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
