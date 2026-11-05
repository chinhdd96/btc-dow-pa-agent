import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

SYMBOL = "BTCUSDT"
# Champion entry TF — 15m for faster paper signals; override via env if needed
TIMEFRAME_MAIN = (os.getenv("TIMEFRAME_MAIN", "15m").strip() or "15m")
TIMEFRAME_TREND = os.getenv("TIMEFRAME_TREND", "4h").strip() or "4h"
TIMEFRAME_HTF = os.getenv("TIMEFRAME_HTF", "1d").strip() or "1d"  # primary regime
# Intermediate HTF when main is sub-hour (15m → 1h bias filter)
TIMEFRAME_MID = os.getenv("TIMEFRAME_MID", "1h").strip() or "1h"
LEVERAGE = 3
RISK_PER_TRADE_PCT = 0.015
MAX_CONSECUTIVE_LOSSES = 2
BASE_MIN_SCORE = float(os.getenv("BASE_MIN_SCORE", "5.0"))
PENALTY_MIN_SCORE = float(os.getenv("PENALTY_MIN_SCORE", "6.5"))
MIN_RR = float(os.getenv("MIN_RR", "1.2"))
DOW_DIRECTION_GATE = os.getenv("DOW_DIRECTION_GATE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
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
# Legacy interval (unused when clock-aligned schedule is on). Keep for env compat.
DECISION_INTERVAL_MINUTES = int(os.getenv("DECISION_INTERVAL_MINUTES", "60"))
# Run trade job every hour at HH:MM — after H1 close at :00.
# Sweet spot ổn định API: 2–5 phút sau đóng H1 (:00).
DECISION_AT_MINUTE = max(0, min(59, int(os.getenv("DECISION_AT_MINUTE", "5"))))
# Short-term decision history injected into the next prompt
DECISION_HISTORY_MAX = int(os.getenv("DECISION_HISTORY_MAX", "20"))
DECISION_SUMMARY_MAX_CHARS = int(os.getenv("DECISION_SUMMARY_MAX_CHARS", "200"))
DECISION_HISTORY_PROMPT_MAX_CHARS = int(
    os.getenv("DECISION_HISTORY_PROMPT_MAX_CHARS", "1400")
)
DECISION_SUMMARY_LLM_THRESHOLD = int(
    os.getenv("DECISION_SUMMARY_LLM_THRESHOLD", "280")
)
# Persist decision history across Blitz rebuilds via GitHub Contents API
HISTORY_GITHUB_TOKEN = (
    os.getenv("HISTORY_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
).strip()
HISTORY_GITHUB_REPO = os.getenv(
    "HISTORY_GITHUB_REPO", "chinhdd96/btc-dow-pa-agent"
).strip()
HISTORY_GITHUB_PATH = os.getenv(
    "HISTORY_GITHUB_PATH", "generated/decision_history.jsonl"
).strip()
HISTORY_GITHUB_BRANCH = os.getenv("HISTORY_GITHUB_BRANCH", "history-data").strip()
# Paper account JSON on same branch (survives Blitz rebuild wipe of generated/)
PAPER_GITHUB_PATH = os.getenv(
    "PAPER_GITHUB_PATH", "generated/paper_account.json"
).strip()
MEMORY_GITHUB_PATH = os.getenv("MEMORY_GITHUB_PATH", "generated/memory.json").strip()
ADAPTIVE_RULES_GITHUB_PATH = os.getenv(
    "ADAPTIVE_RULES_GITHUB_PATH", "generated/adaptive_rules.json"
).strip()

# Self-improve / opportunity-first learning loop
# Default: KHÔNG chặn cứng setup (đứng ngoài mãi = không học được).
# Expectancy âm → chỉ nâng ngưỡng điểm (soft), vẫn cho vào lệnh nếu đạt.
SETUP_BLOCK_HARD = os.getenv("SETUP_BLOCK_HARD", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SETUP_SOFT_PENALTY = float(os.getenv("SETUP_SOFT_PENALTY", "0.5"))
ADAPTIVE_CHOP_BONUS = float(os.getenv("ADAPTIVE_CHOP_BONUS", "0.5"))
CHOP_MIN_RR = float(os.getenv("CHOP_MIN_RR", "1.2"))
CHOP_REQUIRE_L3 = os.getenv("CHOP_REQUIRE_L3", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# L1 = quan sát; BUY/SELL tối thiểu L2 để vẫn có chất lượng mẫu
ENTRY_MIN_SIGNAL_LEVEL = int(os.getenv("ENTRY_MIN_SIGNAL_LEVEL", "2"))
QUALITY_WIN_MIN_R = float(os.getenv("QUALITY_WIN_MIN_R", "0.3"))
SETUP_BLOCK_MIN_TRADES = int(os.getenv("SETUP_BLOCK_MIN_TRADES", "8"))
META_REVIEW_EVERY_N_CLOSES = int(os.getenv("META_REVIEW_EVERY_N_CLOSES", "10"))
META_REVIEW_ROLLING_N = int(os.getenv("META_REVIEW_ROLLING_N", "30"))
META_REVIEW_TARGET_QUALITY_WR = float(
    os.getenv("META_REVIEW_TARGET_QUALITY_WR", "0.40")
)
META_REVIEW_MIN_TP_RATIO = float(os.getenv("META_REVIEW_MIN_TP_RATIO", "0.20"))
MANAGE_NEAR_SL_PCT = float(os.getenv("MANAGE_NEAR_SL_PCT", "0.35"))
MANAGE_MAX_HOURS = float(os.getenv("MANAGE_MAX_HOURS", "18"))
MANAGE_DEEP_LOSS_R = float(os.getenv("MANAGE_DEEP_LOSS_R", "0.5"))
MANAGE_CLOSE_MIN_SCORE = float(os.getenv("MANAGE_CLOSE_MIN_SCORE", "4.5"))
# Idle quá lâu → nới nhẹ để không đứng ngoài mãi
ZERO_TRADE_HOURS = float(os.getenv("ZERO_TRADE_HOURS", "12"))
TRADE_STATS_MAX_CHARS = int(os.getenv("TRADE_STATS_MAX_CHARS", "1200"))
ADAPTIVE_RULES_MAX = int(os.getenv("ADAPTIVE_RULES_MAX", "12"))
# Diversity / streak when bleeding same setup
SETUP_STREAK_LEN = int(os.getenv("SETUP_STREAK_LEN", "3"))
SETUP_STREAK_PENALTY = float(os.getenv("SETUP_STREAK_PENALTY", "1.0"))
DIVERSITY_AFTER_STREAK = int(os.getenv("DIVERSITY_AFTER_STREAK", "3"))
DIVERSITY_SCORE_BONUS = float(os.getenv("DIVERSITY_SCORE_BONUS", "1.0"))
# Kill adaptive rules that fail after enough closes
RULE_KILL_MIN_TRADES = int(os.getenv("RULE_KILL_MIN_TRADES", "6"))

# Pole-3: AI-owned hypotheses (mechanism only — no setup blacklist)
HYP_MAX_ACTIVE = int(os.getenv("HYP_MAX_ACTIVE", "12"))
HYP_MIN_EVAL = int(os.getenv("HYP_MIN_EVAL", "6"))
HYP_MAG_STEP = float(os.getenv("HYP_MAG_STEP", "0.25"))
HYP_AVOID_HOLD_MAG = float(os.getenv("HYP_AVOID_HOLD_MAG", "2.0"))
HYP_AVOID_HOLD_EVIDENCE = int(os.getenv("HYP_AVOID_HOLD_EVIDENCE", "3"))
HYP_OVERRIDE_RR_BONUS = float(os.getenv("HYP_OVERRIDE_RR_BONUS", "0.5"))
HYP_META_EVERY_N = int(os.getenv("HYP_META_EVERY_N", "10"))
HYPOTHESES_GITHUB_PATH = os.getenv(
    "HYPOTHESES_GITHUB_PATH", "generated/hypotheses.json"
).strip()

# Historical replay (legacy Dow/PA) — default OFF for Champion flow
REPLAY_FORCE = os.getenv("REPLAY_FORCE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REPLAY_DAYS = int(os.getenv("REPLAY_DAYS", "30"))
REPLAY_STEP_HOURS = max(1, int(os.getenv("REPLAY_STEP_HOURS", "1")))
REPLAY_MANAGE_ENABLED = os.getenv("REPLAY_MANAGE_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REPLAY_SEED_MEMORY = os.getenv("REPLAY_SEED_MEMORY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REPLAY_SEED_RULES = os.getenv("REPLAY_SEED_RULES", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REPLAY_LLM_SLEEP_SEC = float(os.getenv("REPLAY_LLM_SLEEP_SEC", "1.5"))
REPLAY_PROGRESS_EVERY = int(os.getenv("REPLAY_PROGRESS_EVERY", "24"))
REPLAY_DIR = os.getenv("REPLAY_DIR", str(BASE_DIR / "generated" / "replay"))
REPLAY_GITHUB_PATH = os.getenv(
    "REPLAY_GITHUB_PATH", "generated/replay_summary.json"
).strip()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET", "")

# Champion Research Agent: live Binance data + PAPER fills only (no live orders).
# PAPER_MODE defaults TRUE for this paradigm; set PAPER_MODE=false only after
# a future live-execution implementation is explicitly approved.
_paper_env = os.getenv("PAPER_MODE", "true").strip().lower()
PAPER_MODE = _paper_env not in {"0", "false", "no", "off"}
# Force paper-only in this release — live bracket is hard-disabled.
FORCE_PAPER_ONLY = True
PAPER_BALANCE_USDT = float(os.getenv("PAPER_BALANCE_USDT", "1000"))
# Margin USDT mỗi lệnh paper (legacy Dow/PA path; Champion uses MAX_MARGIN_FRACTION)
PAPER_MARGIN_PER_TRADE = float(os.getenv("PAPER_MARGIN_PER_TRADE", "100"))
# Legacy LLM manage — disabled in Champion flow
PAPER_MANAGE_ENABLED = os.getenv("PAPER_MANAGE_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# --- Champion research / paper (v3-style) ---
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", str(PAPER_BALANCE_USDT)))
MAX_MARGIN_FRACTION = float(os.getenv("MAX_MARGIN_FRACTION", "0.10"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.0005"))
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0002"))
MAINTENANCE_MARGIN_RATE = float(os.getenv("MAINTENANCE_MARGIN_RATE", "0.005"))
# 15m history default shorter (730d×96 bars ≈ 70k — quá nặng research trên free plan)
_HISTORY_DEFAULT = "180" if TIMEFRAME_MAIN in {"1m", "3m", "5m", "15m", "30m"} else "730"
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", _HISTORY_DEFAULT))
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", "36"))
GENERATIONS = int(os.getenv("GENERATIONS", "12"))
ELITE_FRACTION = float(os.getenv("ELITE_FRACTION", "0.20"))
MUTATION_RATE = float(os.getenv("MUTATION_RATE", "0.35"))
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.55"))
VALIDATION_FRACTION = float(os.getenv("VALIDATION_FRACTION", "0.20"))
MIN_VALIDATION_TRADES = int(os.getenv("MIN_VALIDATION_TRADES", "12"))
MIN_TEST_TRADES = int(os.getenv("MIN_TEST_TRADES", "8"))
# Soft-promote: khi không ai hard-qualify, vẫn promote top soft để paper học được
PAPER_SOFT_PROMOTE = os.getenv("PAPER_SOFT_PROMOTE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SOFT_MIN_VALIDATION_TRADES = int(os.getenv("SOFT_MIN_VALIDATION_TRADES", "6"))
SOFT_MIN_TEST_TRADES = int(os.getenv("SOFT_MIN_TEST_TRADES", "4"))
SOFT_MIN_PROFIT_FACTOR = float(os.getenv("SOFT_MIN_PROFIT_FACTOR", "1.0"))
# Soft-entry live: OFF mặc định — học qua tick_history → research đổi genome
PAPER_SOFT_ENTRY = os.getenv("PAPER_SOFT_ENTRY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PAPER_SOFT_VOLUME_CAP = float(os.getenv("PAPER_SOFT_VOLUME_CAP", "1.1"))
PAPER_SOFT_MAX_ATR_FLOOR = float(os.getenv("PAPER_SOFT_MAX_ATR_FLOOR", "0.04"))
TICK_HISTORY_MAX = int(os.getenv("TICK_HISTORY_MAX", "80"))
TICK_HISTORY_PROMPT_MAX_CHARS = int(os.getenv("TICK_HISTORY_PROMPT_MAX_CHARS", "1800"))
# Chronic SKIP → early research (genome mới), không nới filter live
STARVE_MIN_TICKS = int(os.getenv("STARVE_MIN_TICKS", "16"))
STARVE_SKIP_RATIO = float(os.getenv("STARVE_SKIP_RATIO", "0.85"))
STARVE_TRIGGER_RESEARCH = os.getenv(
    "STARVE_TRIGGER_RESEARCH", "true"
).strip().lower() in {"1", "true", "yes", "on"}
STARVE_COOLDOWN_HOURS = float(os.getenv("STARVE_COOLDOWN_HOURS", "6"))
PAPER_POLL_SECONDS = int(os.getenv("PAPER_POLL_SECONDS", "20"))
RESEARCH_EVERY_HOURS = float(os.getenv("RESEARCH_EVERY_HOURS", "24"))
RESEARCH_RANDOM_SEED = int(os.getenv("RESEARCH_RANDOM_SEED", "42"))
LLM_RESEARCH_ENABLED = os.getenv("LLM_RESEARCH_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Safer default leverage set for paper learning (override via env)
_lev_raw = os.getenv("LEVERAGE_CANDIDATES", "1,2,3,5,10").strip()
LEVERAGE_CANDIDATES: tuple[int, ...] = tuple(
    int(x.strip()) for x in _lev_raw.split(",") if x.strip()
) or (1, 2, 3, 5, 10)
DECAY_MIN_TRADES = int(os.getenv("DECAY_MIN_TRADES", "8"))
DECAY_EXPECTANCY_RATIO = float(os.getenv("DECAY_EXPECTANCY_RATIO", "0.0"))
ARCHIVE_LEARN_STATE_ON_START = os.getenv(
    "ARCHIVE_LEARN_STATE_ON_START", "true"
).strip().lower() in {"1", "true", "yes", "on"}
# Disable legacy LLM hourly replay seed by default
REPLAY_ON_START = os.getenv("REPLAY_ON_START", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MANAGE_MIN_SCORE = float(os.getenv("MANAGE_MIN_SCORE", "6.0"))
# TRAIL must lock at least this many R beyond entry (R = |entry - initial_sl|)
TRAIL_MIN_LOCK_R = float(os.getenv("TRAIL_MIN_LOCK_R", "0.3"))
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
CORE_PLAYBOOK_MAX_CHARS = int(os.getenv("CORE_PLAYBOOK_MAX_CHARS", "2200"))
DAILY_PLAYBOOK_MAX_CHARS = int(os.getenv("DAILY_PLAYBOOK_MAX_CHARS", "1500"))
MEMORY_LESSONS_MAX_CHARS = int(os.getenv("MEMORY_LESSONS_MAX_CHARS", "900"))
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

RESEARCH_DIR = BASE_DIR / "generated" / "research"
RESEARCH_DB_PATH = os.getenv(
    "RESEARCH_DB_PATH", str(RESEARCH_DIR / "btc_agent.sqlite3")
)
RESEARCH_REPORT_PATH = os.getenv(
    "RESEARCH_REPORT_PATH", str(RESEARCH_DIR / "last_report.json")
)
ARCHIVE_DIR = BASE_DIR / "generated" / "archive"

PATHS = {
    "CORE_PLAYBOOK": str(BASE_DIR / "generated" / "core_playbook.md"),
    "DAILY_PLAYBOOK": str(BASE_DIR / "generated" / "daily_playbook.md"),
    "CONTEXT_NEWS": str(BASE_DIR / "generated" / "context.json"),
    "MEMORY": str(BASE_DIR / "memory.json"),
    "SOURCES": str(BASE_DIR / "sources.json"),
    "STATE": str(BASE_DIR / "generated" / "runtime_state.json"),
    "PAPER_ACCOUNT": str(BASE_DIR / "generated" / "paper_account.json"),
    "DECISION_HISTORY": str(BASE_DIR / "generated" / "decision_history.jsonl"),
    "ADAPTIVE_RULES": str(BASE_DIR / "generated" / "adaptive_rules.json"),
    "HYPOTHESES": str(BASE_DIR / "generated" / "hypotheses.json"),
    "REPLAY_SUMMARY": str(BASE_DIR / "generated" / "replay" / "summary.json"),
    "REPLAY_DONE": str(BASE_DIR / "generated" / "replay" / "done.json"),
    "RESEARCH_DB": RESEARCH_DB_PATH,
    "RESEARCH_REPORT": RESEARCH_REPORT_PATH,
    "TICK_HISTORY": str(BASE_DIR / "generated" / "tick_history.jsonl"),
}
