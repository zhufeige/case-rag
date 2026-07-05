"""
SSL 修复模块

问题根因：conda 环境中 Python 3.11 捆绑的 OpenSSL 3.5.x 在解析 Windows 证书库时
会触发 ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]，导致 ssl.create_default_context()
直接崩溃，进而让 aiohttp / dashscope / requests 等库无法导入。

修复方案：拦截 _load_windows_store_certs 的异常，改用 certifi 提供的 CA 证书。

使用方式：在任何会触发 SSL 的 import 之前，首先 import ssl_fix
"""

import ssl
import os

# 1. 将 certifi 的 CA 证书路径写入环境变量，让后续的 SSL 库自动使用
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    _CERTIFI_PATH = certifi.where()
except ImportError:
    _CERTIFI_PATH = None

# 2. 保存原始方法
_original_load_windows_store_certs = ssl.SSLContext._load_windows_store_certs


def _safe_load_windows_store_certs(self, storename, purpose):
    """安全加载 Windows 证书库，解析失败时回退到 certifi"""
    try:
        _original_load_windows_store_certs(self, storename, purpose)
    except ssl.SSLError:
        # Windows 证书库中的部分证书无法被当前 OpenSSL 解析，直接跳过
        # 改用 certifi 的证书作为信任根
        if _CERTIFI_PATH and os.path.exists(_CERTIFI_PATH):
            try:
                self.load_verify_locations(cafile=_CERTIFI_PATH)
            except ssl.SSLError:
                pass


# 3. 替换方法
ssl.SSLContext._load_windows_store_certs = _safe_load_windows_store_certs

# 4. 验证修复是否生效
try:
    _test_ctx = ssl.create_default_context()
    _FIX_OK = True
except ssl.SSLError:
    # 如果补丁后仍然失败，再尝试直接用 certifi 创建 context
    if _CERTIFI_PATH:
        ssl.create_default_context = lambda purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None: \
            ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).__class__._create_default_context(
                purpose, _CERTIFI_PATH, None, None
            ) if hasattr(ssl.SSLContext, '_create_default_context') else None
    _FIX_OK = False
