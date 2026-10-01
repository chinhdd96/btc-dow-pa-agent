# Hướng dẫn chạy BTC Dow & PA Mono-Agent

Checklist trước khi chạy: điền secrets, chỉnh nguồn crawl (tuỳ chọn), cài dependency, rồi `python main.py`.

---

## 1. Bắt buộc điền — file `.env`

Copy mẫu rồi mở file `.env` (không commit lên git):

```powershell
cd c:\Users\chinh.dd_extremevn\Desktop\study\btc_dow_pa_agent
copy .env.example .env
```

| Biến | Bắt buộc? | Lấy ở đâu / ý nghĩa |
|------|-----------|---------------------|
| `BINANCE_API_KEY` | Live only | Có thể để trống nếu `PAPER_MODE=true` |
| `BINANCE_API_SECRET` | Live only | Có thể để trống nếu paper |
| `PAPER_MODE` | Khuyến nghị lúc đầu | `true` = không đặt lệnh Binance; giả lập khớp entry/SL/TP theo giá live |
| `PAPER_BALANCE_USDT` | Paper | Vốn đầu: **1000** |
| `PAPER_MARGIN_PER_TRADE` | Paper | Margin mỗi lệnh: **100** (notional = 100 × leverage) |
| `LLM_API_KEY` | Có | Groq: [console.groq.com/keys](https://console.groq.com/keys) (prefix `gsk_`) |
| `LLM_BASE_URL` | Có | Mặc định Groq: `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | Có | Mặc định: `openai/gpt-oss-120b` (xem model key còn access) |
| `TELEGRAM_BOT_TOKEN` | Nên có | [@BotFather](https://t.me/BotFather) → `/newbot` |
| `TELEGRAM_CHAT_ID` | Nên có | [@userinfobot](https://t.me/userinfobot) / `getUpdates` |

### LLM (mặc định Groq)

| Provider | `LLM_BASE_URL` | `LLM_MODEL` ví dụ | Lấy key |
|----------|----------------|-------------------|---------|
| **Groq** (mặc định) | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` (hoặc `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`) | [console.groq.com/keys](https://console.groq.com/keys) |
| OpenRouter | `https://openrouter.ai/api/v1` | `deepseek/deepseek-chat` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com) |

Alias cũ vẫn nhận: `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` (map sang `LLM_*`).

### Quyền Binance API (quan trọng)

- Bật **Enable Futures**
- **Tắt Withdraw** (không cần rút tiền)
- Nếu Binance yêu cầu: bật **IP whitelist** = IP máy chạy bot
- Tài khoản Futures nên ở chế độ **One-way** (không Hedge), vì client đặt lệnh theo one-way + `closePosition`

Ví dụ `.env` (Groq):

```env
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
LLM_API_KEY=gsk_xxxxxxxx
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=987654321
```

Alias Binance/Telegram: `BINANCE_KEY` / `BINANCE_SECRET` / `TELEGRAM_TOKEN` (xem `config.py`).

### Paper giả lập (vốn 1000u, mỗi lệnh 100u)

Không đặt lệnh Binance. Luồng mỗi giờ:

1. Đọc giá live (+ high/low nến 1H)
2. Nếu đang **pending**: giá khớp entry → FILLED (margin 100u × leverage)
3. Nếu đang **open**: giá chạm SL/TP → CLOSED, cộng/trừ PnL vào balance
4. Flat + đủ margin → LLM quyết định → đăng ký pending/filled paper
5. Telegram báo balance, PnL, `day=X/30`

State lưu tại `generated/paper_account.json`. Sau ~30 ngày xem `balance` còn bao nhiêu.

Reset vốn về 1000u: xóa `generated/paper_account.json` rồi restart bot.

---

## 2. Tầng 1 — Core Playbook (đã có sẵn — bỏ qua build PDF)

File quyết định: `generated/core_playbook.md`

**Đã copy sẵn** từ `study/core_playbook.md` — **không cần** chạy `scripts/build_core_playbook.py` nữa trừ khi muốn distill PDF mới.

Bot mỗi chu kỳ trade đọc file này vào `{core_playbook_content}` (tối đa ~28k ký tự, chỉnh `CORE_PLAYBOOK_MAX_CHARS` trong config nếu cần).

### (Tuỳ chọn) Distill PDF mới

Chỉ khi muốn thay playbook từ `knowledge/dow` + `knowledge/price_action`:

```powershell
python scripts/build_core_playbook.py
```

### Tầng 2A & 2B — cùng kiểu đúc kết (tự chạy trong bot)

Không dump transcript/tin thô vào prompt trade. Crawl → DeepSeek **distill** → file gọn:

| Tầng | Nguồn | Prompt distill | File ra | Lịch |
|------|-------|----------------|---------|------|
| **2A** | YouTube transcript (`sources.json`) | Bias, key levels, setup IF/THEN, PA, cấm, heuristic | `generated/daily_playbook.md` | Start + mỗi 24h |
| **2B** | RSS news | risk level, summary, key_events, risk_instruction, actionable_rules | `generated/context.json` | Start + mỗi 24h |

Nguồn dài → **chunk → đúc từng phần → merge** (giống PDF Tầng 1). Lúc trade, `llm_agent` chỉ đọc 2 file đã đúc kết (daily ~8k ký tự, news JSON).

Code: `modules/learner_crawler.py`, `modules/news_crawler.py`.

---

## 3. Tuỳ chọn — `sources.json`

Không bắt buộc để bot start, nhưng nên sửa cho đúng nguồn học / tin:

- `youtube_channels[].channel_id` — ID kênh YouTube (dạng `UCxxxx`)
- `rss_feeds[].url` — feed tin tức
- `max_videos_per_channel`, `max_rss_items`

---

## 4. Cài đặt & chạy

```powershell
cd c:\Users\chinh.dd_extremevn\Desktop\study\btc_dow_pa_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

### Docker deploy

```powershell
cd c:\Users\chinh.dd_extremevn\Desktop\study\btc_dow_pa_agent
# Đảm bảo đã có .env (LLM + Telegram; PAPER_MODE=true)
docker compose up -d --build
docker compose logs -f
docker compose down
```

Hoặc chỉ Dockerfile:

```powershell
docker build -t btc-dow-pa-agent .
docker run -d --name btc_dow_pa_agent --restart unless-stopped --env-file .env `
  -v ${PWD}/generated:/app/generated `
  -v ${PWD}/memory.json:/app/memory.json `
  btc-dow-pa-agent
```

Volume `generated/` giữ `paper_account.json` (vốn paper) và playbook/news khi restart container.

### Khi start, bot sẽ

1. Gửi Telegram: “Bot started…”
2. Chạy **ngay 1 lần**: learner (YouTube) → news (RSS) → trade decision
3. Loop 24/7:
   - Trade mỗi **60 phút** (`DECISION_INTERVAL_MINUTES`)
   - Learner + news mỗi **24 giờ**
   - Compact memory Chủ nhật **00:00 UTC**

Dừng: `Ctrl+C`.

---

## 5. Tham số cố định trong `config.py` (không nằm trong `.env`)

| Tham số | Giá trị mặc định |
|---------|------------------|
| `SYMBOL` | `BTCUSDT` |
| `TIMEFRAME_MAIN` | `1h` |
| `TIMEFRAME_TREND` | `4h` |
| `LEVERAGE` | `3` |
| `RISK_PER_TRADE_PCT` | `0.015` (1.5% NAV) |
| Min setup score | `6.0` → `8.5` sau 2 lệnh thua liên tiếp |

Đổi risk/leverage/timeframe thì sửa `config.py` rồi restart.

---

## 6. Checklist nhanh trước lần chạy đầu (mainnet)

- [ ] Đã tạo `.env` và điền đủ 5 biến bắt buộc/nên có ở mục 1
- [ ] API Binance chỉ Futures Trade, tắt Withdraw
- [ ] Tài khoản có USDT trên **Futures wallet**
- [ ] Đã nhắn ít nhất 1 tin cho Telegram bot (để bot gửi được tin)
- [ ] `pip install -r requirements.txt` thành công
- [ ] Hiểu rõ: lệnh BUY/SELL khớp **mainnet thật**, có SL/TP bracket

---

## 7. Lỗi thường gặp

| Hiện tượng | Việc cần làm |
|------------|--------------|
| `Missing BINANCE_API_KEY` | Chưa tạo/điền `.env` hoặc chạy sai thư mục |
| Binance `-2015` / invalid key | Sai key, sai quyền Futures, hoặc IP chưa whitelist |
| Telegram không về | Sai `TELEGRAM_CHAT_ID` hoặc chưa `/start` bot |
| Learner “no transcripts” | Channel không có caption / sai `channel_id` — bot vẫn trade được |
| DeepSeek / LLM lỗi auth hoặc timeout | Sai `LLM_API_KEY` / `LLM_BASE_URL`; từ VN dùng OpenRouter thay `api.deepseek.com` |
