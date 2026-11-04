BTCUSDT AUTONOMOUS QUANT RESEARCH + PAPER AGENT v3.0
===================================================

WHAT THIS VERSION IMPLEMENTS

1) LLM RESEARCHER
- Optional OpenAI-compatible Chat Completions interface.
- LLM proposes strategy hypotheses as structured Genome JSON.
- Every proposal is schema-checked and then evaluated by deterministic Python code.
- The LLM does NOT decide whether a strategy is profitable and cannot change the risk allocation setting.
- If OPENAI_API_KEY is absent, evolutionary search continues without the LLM.

2) ADVANCED QUANT / DERIVATIVES ENGINE
- Closed Binance USD-M Futures OHLCV candles, quote volume and taker-buy quote flow.
- Historical funding-rate events and funding cashflows in backtest / paper accounting.
- Recent open-interest changes (Binance endpoint retention is limited; typically around 30 days).
- Causal 4h and 1d trend context. Higher-timeframe candles become available only after their close.
- Rolling VWAP deviation, volatility rank, taker-flow imbalance and 11 strategy families.
- Approximate liquidation price checks in backtest and paper trading.
- Funding, fees and slippage are included in net results.

3) CONTINUOUS STRATEGY EVOLUTION
- Population search, mutation, crossover and random exploration.
- Optional LLM hypotheses are injected as challengers.
- Train / validation / test chronological evaluation.
- Champion is retained unless a qualifying challenger demonstrates a meaningful score improvement.
- Paper loop re-runs research on a schedule (default every 24 hours).
- Missed closed candles are replayed in chronological order after downtime where retained in the recent data window.
- If no candidate passes validation, no new Champion is installed; the agent can stay flat.

4) PAPER EXECUTION MODEL
- Persistent paper state and trade history in SQLite.
- Pending signal -> next-candle simulated fill -> SL / TP / estimated liquidation handling.
- Public order-book depth snapshot is used as an approximate paper fill model; fallback slippage is used if unavailable.
- Funding cashflows are credited/debited while a paper position is open.
- No authenticated Binance endpoints and no live orders exist in this version.

USER-APPROVED CONFIGURATION
- Symbol: BTCUSDT USD-M Futures
- Starting capital: 1,000 USDT
- Maximum margin allocation per position: 10% of current paper equity.
- Leverage: selected by the strategy genome from candidate values 1, 2, 3, 5, 10, 20, 25, 50, 75, 100, 125.
- Mode: PAPER_MANUAL. Live mode is not implemented; future live execution requires a separate implementation and explicit user approval.

IMPORTANT RISK / MODEL LIMITATIONS
- 10% is margin allocation, NOT a 10% maximum loss. Notional = margin x leverage.
- At 100x, a 100 USDT margin allocation represents about 10,000 USDT notional. A small adverse move can cause liquidation.
- Leverage candidates are research choices, not a guarantee that the user's Binance account or BTCUSDT risk tier permits that leverage/notional.
- Liquidation price uses a simplified maintenance-margin estimate (default 0.5%). Actual Binance maintenance margin, liquidation fee, tier, wallet mode, mark-price triggers and account state differ. Treat LIQUIDATION_ESTIMATE as approximate, not exchange-exact.
- Historical funding is applied when a funding event timestamp aligns with the candle timestamp. Historical funding coverage depends on Binance public endpoint availability.
- Open-interest history endpoint retention is limited; older candles have missing OI values.
- Order-book depth is a current public snapshot, not historical order-book replay. It is only an approximate paper fill model, not queue/latency/partial-fill truth.
- OHLC candle simulation cannot know intrabar order sequence. If liquidation, SL or TP ambiguity occurs, liquidation is prioritized, then SL, then TP.
- Multi-timeframe alignment is causal, but should still be audited against the exact exchange candle conventions before research conclusions are trusted.
- The evolutionary grammar contains 11 strategy families. LLM research expands hypotheses within the supported schema; it does not execute arbitrary generated Python or rewrite its own source code.
- No system can guarantee positive returns. Backtest / paper results are not proof of future profitability.

INSTALL
  python -m pip install -r requirements_btcusdt_agent_v3.txt

OPTIONAL LLM SETUP (PowerShell)
  $env:OPENAI_API_KEY="YOUR_API_KEY"
  $env:OPENAI_MODEL="gpt-5-mini"
  # Optional for OpenAI-compatible providers:
  # $env:OPENAI_BASE_URL="https://api.openai.com/v1"

OPTIONAL LLM SETUP (Linux/macOS)
  export OPENAI_API_KEY="YOUR_API_KEY"
  export OPENAI_MODEL="gpt-5-mini"
  # export OPENAI_BASE_URL="https://api.openai.com/v1"

RUN RESEARCH
  python btcusdt_autonomous_agent_v3.py --mode research --capital 1000 --days 730

RUN RESEARCH WITHOUT LLM
  python btcusdt_autonomous_agent_v3.py --mode research --capital 1000 --days 730 --no-llm

RUN CONTINUOUS PAPER TRADING + DAILY EVOLUTION
  python btcusdt_autonomous_agent_v3.py --mode paper --capital 1000 --research-hours 24

RUN MARKET SNAPSHOT
  python btcusdt_autonomous_agent_v3.py --mode snapshot

COMPUTE SETTINGS
  --generations 12 --population 36 --interval 1h --research-hours 24

OUTPUTS
- btc_agent_v3.sqlite3: Champion, paper state, trades and research runs.
- btc_agent_v3_report.json: ranked candidates and evaluation metrics.

BEFORE TRUSTING ANY RESULTS
- Verify actual account commission tier and set fee_rate in Config accordingly.
- Compare historical funding events against a separate data source.
- Validate the liquidation approximation against Binance contract specifications and account risk tier.
- Run long forward paper tests across trending, ranging and high-volatility regimes.
- Inspect every report; a strategy is not qualified merely because its training return is positive.
