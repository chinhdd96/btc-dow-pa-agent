# Tầng 1 — Knowledge (PDF nguồn học)

Bot **không đọc PDF trực tiếp** khi trade. Luồng:

```
knowledge/dow/*.pdf
knowledge/price_action/*.pdf
        │
        ▼
scripts/build_core_playbook.py
  1. Extract text PDF
  2. DeepSeek DISTILL prompt → đúc kết rule / setup / cấm
  3. (PDF dài) chunk → distill từng phần → merge
        │
        ▼
generated/core_playbook.md   ← gọn, actionable
        │
        ▼
Mỗi chu kỳ trade: llm_agent nhét ~12k ký tự vào {core_playbook_content}
```

## Đặt file PDF ở đâu?

| Mục đích | Thư mục | Vai trò |
|----------|---------|---------|
| Lý thuyết Dow (xu hướng HH/HL, LH/LL) | `knowledge/dow/` | Xác định hướng |
| Price Action (pinbar, engulfing, key level) | `knowledge/price_action/` | Timing entry |

## Build / đúc kết playbook

Cần có `LLM_API_KEY` trong `.env` (khuyến nghị OpenRouter từ VN):

```powershell
pip install pypdf openai python-dotenv
python scripts/build_core_playbook.py
```

Prompt đúc kết yêu cầu LLM xuất: nguyên tắc → setup IF/THEN → pattern → cấm → SL/TP → heuristic (không copy nguyên sách).

PDF rất dài sẽ được **chia chunk → tóm tắt từng phần → gộp**, rồi ghi playbook mục tiêu ~4.5k ký tự / section (vừa context lúc trade).
