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
| Genesis | LLM | Thiết kế **1 family duy nhất** (3–5 loại chỉ báo): 6 bản nháp → walk-forward chọn v1 |
| Cải tiến | LLM | Mỗi lần research: 4 biến thể add/remove/replace/tune chỉ báo hoặc exit (patch 1–2 thao tác; rework tối đa 4, được thay lõi) |
| Evolution | Python | Chỉ tinh chỉnh số (ngưỡng, SL/TP, đòn bẩy) của các biến thể — không đổi bộ chỉ báo |
| AI Review | LLM | Giải thích — **không** promote |
| Promote | Python | Lên version mới khi thắng OOS; version đang khỏe chỉ bị thay khi thắng ≥8 điểm, nếu không ứng viên phải thắng forward ở shadow |
| Paper | Python | Signal theo Champion Spec → fill nến sau; SL/TP/LIQ |
| Tick history | Python | SKIP reason → starve → research sớm |

`SPEC_PATCH_MIN_TRADES=12` → patch; decay, ≥20 lệnh mà lỗ, hoặc 3 lần research liên tiếp version yếu → rework. Cooldown 6h, tối đa ~1 research/24h ceiling.
Đếm chỉ báo theo loại: EMA12 + EMA50 = 1 (ema), Donchian high/low = 1, giá close không tính, `mtf_filter` tính 1.
Nhật ký version (đổi gì, vì sao, kết quả backtest + paper) được gửi lại cho LLM ở lần sau.
Genesis cần LLM; nếu LLM lỗi thì giữ Champion cũ (hoặc flat nếu chưa có) và thử lại sau 30 phút.

- **Vòng phản hồi:** trong 1 lần research, nếu chưa có bản đạt chuẩn, LLM thấy kết quả OOS của từng bản (lệnh, PF, win, DD, fold + verdict `too_few_trades` / `losing` / `high_drawdown`) rồi thiết kế lại — tối đa 3 vòng trong 15 phút.
- **Version chưa đạt chuẩn (khám phá):** size ×0.5, research lại sau 6h ở chế độ rework.
- **Đòn bẩy tối đa 10x** (`FAMILY_MAX_LEVERAGE`, chọn trong 1/2/3/5/10) cho mọi version; version cao hơn được hạ xuống và ghi thành version mới.
- **Không vào lệnh 24h** (không lệnh mở / chờ): research `no_trade` — LLM nới / thay chỉ báo đang chặn; chỉ chọn bản có số lệnh OOS ≥ 1.5× (và ≥ +5) bản hiện tại. Không có bản đạt chuẩn nào như vậy thì vẫn đổi sang bản vào lệnh nhiều hơn có điểm cao hơn (chế độ khám phá) — không đứng ngoài mãi.

### Vòng lặp chính (mỗi 20s)

1. Research xong (chạy nền) → main thread áp dụng Champion/shadow, gửi Telegram.
2. Live watch: có pending → khớp ở giá mở nến kế tiếp (nến 1m đầu); có lệnh → kiểm SL/TP/LIQ bằng high/low nến 1m (1 request nhẹ).
3. Có nến 15m mới đóng → tải đủ dữ liệu, xử lý Champion + shadow (warm-up 1000 nến, giống backtest), trailing/hòa vốn/time-stop cập nhật ở nến đóng.
4. Cần research (chưa có Spec / research_due / quá 24h) → chạy nền tối đa 15 phút; lỗi thì chờ 30 phút mới thử lại.

### Học và chọn Spec

- LLM nhận: lịch sử research (Spec tốt/tệ, chỉ báo nào hay đạt chuẩn, kết quả paper thật), bài học sau lệnh, 20 lệnh gần nhất kèm giá trị chỉ báo + regime lúc vào lệnh.
- Đánh giá walk-forward 4 đoạn sau train; đạt chuẩn khi ≥3/4 đoạn có lời, PF > 1.05; phạt Spec quá nhiều điều kiện.
- Shadow: tối đa 3 version ứng viên (hoặc version cũ để quay lại) chạy paper ảo; shadow có ≥8 lệnh lời hơn Champion cùng khoảng thời gian được lên version mới.
- Exit tùy chọn trong Spec: `breakeven_r`, `trail_atr`, `max_bars`, `exit_on_opposite` — cùng một hàm cho backtest, paper và live.

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
