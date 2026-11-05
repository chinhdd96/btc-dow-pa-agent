# Hướng dẫn — Champion Research Agent (live data, paper fill)

Agent nghiên cứu **Strategy Spec** (ghép chỉ báo catalog) → validate → chọn Champion → paper trên nến Binance live.
**Không đặt lệnh live.** LLM propose/patch Spec + review/post-mortem — **không** quyết định BUY/SELL mỗi nến.

**Nhiệm vụ duy nhất: PnL / expectancy sau phí dương** — chấp nhận rủi ro trong bound leverage; không lấy “đứng ngoài mãi” làm chiến lược.

```
LLM ghép/sửa chỉ báo (Spec) → Evolution + backtest → Promote Spec
  → Paper (Python) → đủ N lệnh patch/replace hoặc starve/decay → Research lại
```

**Timeframe:** entry trên nến **15m** đã đóng (`TIMEFRAME_MAIN=15m`). Bias HTF: **1h / 4h / 1d**. Signal nến T → fill open nến T+1 (~15 phút sau); không trade nến đang forming.

**Telegram mỗi nến 15m:** notify ngắn (close / ATR% / RSI / action / vì sao SKIP hoặc PENDING). OPEN/CLOSE vẫn có tin riêng.

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
| `TIMEFRAME_MAIN` | TF vào lệnh (mặc định `15m`) |
| `TIMEFRAME_MID` / `TREND` / `HTF` | Bias filter (`1h` / `4h` / `1d`) |
| `HISTORY_DAYS` | Research lookback (15m mặc định 180) |
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
| Catalog features | Python | Tính chỉ báo **đúng param** Spec yêu cầu |
| Propose / patch | LLM | Strategy Spec từ catalog; sau ≥12 lệnh: tune/add/remove 1 chỉ báo; ≥20: replace |
| Evolution | Python | Seed 11 family→Spec + LLM specs + mutate / evaluate |
| AI Review | LLM | Giải thích — **không** promote |
| Promote | Python | Hard/soft Spec; starve → nới Spec / đổi Spec (vẫn soft-promote để học) |
| Paper | Python | Signal theo Champion Spec → fill nến sau; SL/TP/LIQ |
| Tick history | Python | SKIP reason → starve → research sớm |

`SPEC_PATCH_MIN_TRADES=12`, `SPEC_REPLACE_MIN_TRADES=20`, cooldown 6h, tối đa ~1 research/24h ceiling.
Flat tạm chỉ khi chưa có Spec compile được — research vẫn chạy (PnL-first).

---

## 4. Artifact

| Path | Nội dung |
|------|----------|
| `generated/research/btc_agent.sqlite3` | Champion state, paper trades, research runs |
| `generated/research/last_report.json` | Report research mới nhất |
| `generated/tick_history.jsonl` | Tick 15m gần nhất (research context) |
| GitHub `history-data` → `champion_snapshot.json` | Backup Champion + trades + tick (Blitz rebuild) |
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
