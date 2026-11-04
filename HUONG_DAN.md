# Hướng dẫn — Champion Research Agent (live data, paper fill)

Agent tự nghiên cứu chiến lược → validate → chọn Champion → paper trên nến Binance live.
**Không đặt lệnh live.** LLM chỉ propose / review / post-mortem — không quyết định BUY/SELL mỗi giờ.

```
LLM đề xuất → Evolution + backtest + validation → Champion
  → Paper (Python) → Post-mortem / decay → Research lại
```

---

## 1. Cài đặt

```powershell
cd c:\Users\chinh.dd_extremevn\Desktop\study\btc_dow_pa_agent
copy .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Điền trong `.env`:

| Biến | Ý nghĩa |
|------|---------|
| `LLM_API_KEY` | Groq / OpenAI-compatible (research + post-mortem) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Notify promote / open / close |
| `PAPER_MODE=true` | Bắt buộc phase này |
| `INITIAL_CAPITAL` | Vốn paper (mặc định 1000) |
| `LEVERAGE_CANDIDATES` | Ví dụ `1,2,3,5,10` |
| `RESEARCH_EVERY_HOURS` | Chu kỳ research (mặc định 24) |
| `PAPER_POLL_SECONDS` | Poll nến live (mặc định 20) |

Binance API key **không cần** cho paper (chỉ public market data).

---

## 2. Chạy

```powershell
python main.py
```

Khi start:

1. Archive một lần state học cũ (paper_account, hypotheses, …) → `generated/archive/`
2. Learner + news (context cho researcher, không gate entry)
3. Chạy **research** ngay (evolution + optional LLM propose + AI review + promote)
4. Loop: paper tick mỗi `PAPER_POLL_SECONDS`; research lại theo lịch hoặc khi decay

One-shot research (debug):

```powershell
python scripts/run_research.py --no-llm --days 180 --generations 2 --population 8
```

---

## 3. Flow chi tiết

| Bước | Ai | Việc |
|------|-----|------|
| Data | Python | Klines + funding + OI + MTF (public futures) |
| Analysis | Python | Features (EMA/RSI/ATR/VWAP/…) |
| Propose | LLM | Genome JSON trong schema |
| Evolution | Python | Mutate / crossover / evaluate train·val·test |
| AI Review | LLM | Giải thích report — **không** promote |
| Promote | Python | `maybe_promote` chỉ candidate `qualified` |
| Paper | Python | Signal nến đóng → fill open sau; SL/TP/LIQ/funding |
| Post-mortem | LLM + Python | Lesson + challenger hints + decay → early research |

Không Champion qualified → **đứng flat** (không LLM entry thay thế).

---

## 4. Artifact

| Path | Nội dung |
|------|----------|
| `generated/research/btc_agent.sqlite3` | Champion state, paper trades, research runs |
| `generated/research/last_report.json` | Report research mới nhất |
| `generated/daily_playbook.md` | Learner → context researcher |
| `generated/context.json` | News → context researcher |
| `generated/archive/` | State học cũ (migrate một lần) |
| `memory.json` | Lesson sau paper close |

Reference monolith cũ: `btcusdt_agent_v3_bundle/` (không chạy production).

---

## 5. Docker

```powershell
docker compose up -d --build
docker compose logs -f
```

Volume `generated/` giữ SQLite research + playbook.

---

## 6. Lưu ý risk

- Margin tối đa `MAX_MARGIN_FRACTION` (mặc định 10% equity) × leverage genome — không phải max loss 10%.
- Liquidation / depth fill là ước lượng paper, không khớp 1:1 sàn.
- Backtest dương ≠ edge tương lai; luôn theo dõi paper + decay.

---

## 7. Legacy Dow/PA

`modules/llm_agent.py` (hourly decision), `hypothesis_store`, replay-on-start **không còn hot path**.
`open_bracket` live bị hard-disable (`FORCE_PAPER_ONLY`).
