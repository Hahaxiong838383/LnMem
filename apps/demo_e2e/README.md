---
version: 0.1.0
audience: cc / 工程师
purpose: Streamlit 端到端演示 LnMem M0 的 parser 到 KG 闭环
---

# LnMem M0 Demo（端到端浏览器）

## 启动

```bash
# 1) 首次安装 streamlit
.venv-poc-fresh/bin/pip install streamlit

# 2) 设置 LLM 代理凭据
export GEMINI_PROXY_TOKEN="fq8-..."
export GEMINI_PROXY_ENDPOINT="http://23.226.135.149:4000/v1/chat/completions"  # 可选
export LNMEM_PARSER_MODEL="gemini-3-flash-preview"                              # 可选

# 3) 运行 demo
PYTHONPATH=. .venv-poc-fresh/bin/streamlit run apps/demo_e2e/main.py
```

默认浏览器地址：`http://localhost:8501`。

## 验收步骤（M0）

1. 输入 5 条中文事件，点击 `解析 + 写入 KG`。
2. 每条可见 20 槽 JSON 分组（P0/P1/P2/P3）和 P0 填充率。
3. 输入查询（例：`昨天和谁吃饭`），点击 `查询`。
4. 看到命中列表（score + 命中事件详情）与下方 KG 节点总览表。

## 限制（demo 范围）

- KG 后端为 InMemory，进程重启后清空。
- parser 使用单一 Gemini proxy callable，无多后端 fallback。
- 当前仅支持文本路径，不含 ASR/视觉输入。
