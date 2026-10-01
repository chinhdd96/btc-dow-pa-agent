import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

SYMBOL = "BTCUSDT"
TIMEFRAME_MAIN = "1h"
TIMEFRAME_TREND = "4h"
TIMEFRAME_HTF = "1d"  # primary regime như trader thực
LEVERAGE = 3
RISK_PER_TRADE_PCT = 0.015
MAX_CONSECUTIVE_LOSSES = 2
BASE_MIN_SCORE = 6.0
PENALTY_MIN_SCORE = 8.5
MAX_SL_DISTANCE_PCT = 0.05
# Local history depth (Dow HTF ~1y; 1H ~3m). Structure uses full file.
HISTORY_DIR = os.getenv("HISTORY_DIR", str(BASE_DIR / "generated" / "history"))
HISTORY_BARS_1D = int(os.getenv("HISTORY_BARS_1D", "365"))
HISTORY_BARS_4H = int(os.getenv("HISTORY_BARS_4H", "2190"))
HISTORY_BARS_1H = int(os.getenv("HISTORY_BARS_1H", "2160"))
# Mỗi chu kỳ chỉ backfill thêm tối đa N nến cũ (bồi đắp dần tới target)
HISTORY_BACKFILL_PER_SYNC = int(os.getenv("HISTORY_BACKFILL_PER_SYNC", "500"))
# Legacy page/zoom limits (not Dow ceiling)
CANDLE_LIMIT_MAIN = int(os.getenv("CANDLE_LIMIT_MAIN", "150"))
CANDLE_LIMIT_TREND = int(os.getenv("CANDLE_LIMIT_TREND", "200"))
CANDLE_LIMIT_HTF = int(os.getenv("CANDLE_LIMIT_HTF", "120"))
MAX_LESSONS_PER_REGIME = 5
DECISION_INTERVAL_MINUTES = 60

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET", "")

# Paper mode: full flow, public market data, NO real orders (Telegram signal only).
# Auto-on when Binance keys missing, or set PAPER_MODE=true explicitly.
_paper_env = os.getenv("PAPER_MODE", "").strip().lower()
PAPER_MODE = (
    _paper_env in {"1", "true", "yes", "on"}
    or not (BINANCE_API_KEY and BINANCE_API_SECRET)
)
PAPER_BALANCE_USDT = float(os.getenv("PAPER_BALANCE_USDT", "1000"))
# Margin USDT mỗi lệnh paper (notional = margin * LEVERAGE)
PAPER_MARGIN_PER_TRADE = float(os.getenv("PAPER_MARGIN_PER_TRADE", "100"))
# Windows/corporate proxy đôi khi lỗi SSL tới Binance — paper có thể tắt verify
BINANCE_SSL_VERIFY = os.getenv("BINANCE_SSL_VERIFY", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# OpenAI-compatible LLM. Default: Groq (accessible from VN).
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("DEEPSEEK_BASE_URL")
    or "https://api.groq.com/openai/v1"
)
LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("DEEPSEEK_MODEL")
    or "openai/gpt-oss-120b"
)
# Prompt size — Groq TPM ~8k tokens: ưu tiên structure + core; CSV mỏng
CORE_PLAYBOOK_MAX_CHARS = int(os.getenv("CORE_PLAYBOOK_MAX_CHARS", "3800"))
DAILY_PLAYBOOK_MAX_CHARS = int(os.getenv("DAILY_PLAYBOOK_MAX_CHARS", "1500"))
MEMORY_LESSONS_MAX_CHARS = int(os.getenv("MEMORY_LESSONS_MAX_CHARS", "500"))
NEWS_SUMMARY_MAX_CHARS = int(os.getenv("NEWS_SUMMARY_MAX_CHARS", "500"))
# CSV trong prompt (structure vẫn tính trên toàn bộ fetch)
# 1D chỉ vài nến gần — regime lấy từ structure; 4H/1H đủ soi PA
CANDLE_PROMPT_BARS = int(os.getenv("CANDLE_PROMPT_BARS", "24"))
CANDLE_PROMPT_BARS_1D = int(os.getenv("CANDLE_PROMPT_BARS_1D", "0"))
CANDLE_PROMPT_BARS_4H = int(os.getenv("CANDLE_PROMPT_BARS_4H", "0"))
CANDLE_PROMPT_BARS_1H = int(os.getenv("CANDLE_PROMPT_BARS_1H", "24"))
LLM_MAX_PROMPT_CHARS = int(os.getenv("LLM_MAX_PROMPT_CHARS", "11000"))
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "1800"))

# Backward-compatible aliases used by existing modules
DEEPSEEK_API_KEY = LLM_API_KEY
DEEPSEEK_BASE_URL = LLM_BASE_URL
DEEPSEEK_MODEL = LLM_MODEL

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

PATHS = {
    "CORE_PLAYBOOK": str(BASE_DIR / "generated" / "core_playbook.md"),
    "DAILY_PLAYBOOK": str(BASE_DIR / "generated" / "daily_playbook.md"),
    "CONTEXT_NEWS": str(BASE_DIR / "generated" / "context.json"),
    "MEMORY": str(BASE_DIR / "memory.json"),
    "SOURCES": str(BASE_DIR / "sources.json"),
    "STATE": str(BASE_DIR / "generated" / "runtime_state.json"),
    "PAPER_ACCOUNT": str(BASE_DIR / "generated" / "paper_account.json"),
}
