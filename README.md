# News Sentiment for Quantitative Investing

This repository is a reproducible, leakage-aware implementation of the project outlined in *Research on the Impact of Financial News Text Data on Quantitative Investment*. It tests whether stock-level finance-news sentiment improves a short-horizon China A-share model beyond traditional technical, valuation, and fundamental variables.

It implements two deliberately comparable strategies:

| Strategy | Inputs |
| --- | --- |
| `pca_only` | Traditional variables selected by L1-logistic regression and compressed with PCA |
| `text_pca` | The same rolling LASSO-PCA factors plus a daily aggregated news-sentiment score |

The repository does **not** ship Tushare data, news articles, credentials, or the headline performance figures in the supplied outline. Those data can be licensed/restricted, and the reported 2023-2025 results cannot honestly be claimed as rerun results without the original data, sample membership, and model checkpoint. Instead, this implementation supplies the full pipeline plus deterministic synthetic data that tests every stage offline.

## Research design

1. Build a CSI 500 market panel with `date`, `ticker`, OHLC prices, and numeric traditional factors.
2. Fetch Eastmoney news, match only articles that map to one listed company, score title + content with a user-selected Chinese finance checkpoint, and define `P(positive) - P(negative)` as article sentiment.
3. Shift each article to the next available trading session and average sentiment by ticker and date. This prevents news released after the close from affecting that close.
4. For each 120-trading-day window, fit L1-logistic selection and PCA on the first 90 dates only. Use the next 30 dates only for Platt probability calibration.
5. Tune RBF-SVM `C` and `gamma` using expanding date-based validation inside the 90-day fit period; predict the following 20 dates; then roll forward by 20 dates.
6. Build a long-only Top-10 portfolio from probabilities above a configurable threshold. Weights are proportional to `max(p - 0.5, 0)`, capped per stock, and adjusted only when the desired/current difference exceeds 2 percentage points.
7. Charge transaction costs on turnover and report annualized return, Sharpe ratio, maximum drawdown, average turnover, and average holdings.

No scaler, LASSO selector, PCA component, SVM hyperparameter, or calibration model sees a later prediction date.

## Quick start: offline end-to-end demo

The demo uses synthetic, non-investment data; its output is a software smoke test, not a finance claim.

```bash
conda activate base  # or use the Python environment containing numpy/pandas/scikit-learn
python -m pip install -e .
python scripts/make_demo_data.py --output-dir data/demo
python scripts/run_pipeline.py \
  --market data/demo/market.csv \
  --news data/demo/news_scored.csv \
  --output-dir outputs/demo
pytest -q
```

Outputs include per-window diagnostics, out-of-sample stock probabilities, daily backtests, and a strategy comparison table under `outputs/demo/`.

## Live data workflow

### 1. Download market data and uniquely matched news

Tushare endpoints and entitlements change by account. Install the optional dependency, set a token in the shell (do not commit it), and request a period:

```bash
python -m pip install -e '.[live-data]'
export TUSHARE_TOKEN='your-token'
python scripts/fetch_tushare.py --start 2023-01-01 --end 2025-12-31
```

The script downloads CSI 500 constituents (`000905.SH`), daily prices, daily-basic data, and `stk_factor_pro`, then saves `data/raw/market.csv`. It also obtains Eastmoney news and keeps only uniquely name-matched articles in `data/raw/matched_news.csv`. Endpoint failures are recorded in `data/raw/download_failures.txt`; do not ignore them when interpreting a rerun.

For point-in-time fundamental data, merge each company’s reports using its actual announcement date, never its reporting period end date. The included downloader is intentionally limited to endpoints that create a daily panel; it does not pretend to reproduce the original financial-report availability timestamps.

### 2. Score the news

Choose and document a Chinese financial-sentiment checkpoint whose license and label order you have verified. The code requires an explicit label order rather than assuming it.

```bash
python -m pip install -e '.[nlp]'
python scripts/score_news.py \
  --input data/raw/matched_news.csv \
  --output data/processed/news_scored.csv \
  --model-id 'YOUR_CHINESE_FINANCE_CHECKPOINT' \
  --label-map '{"positive":"LABEL_2","neutral":"LABEL_1","negative":"LABEL_0"}'
```

### 3. Run the comparison

```bash
python scripts/run_pipeline.py \
  --market data/raw/market.csv \
  --news data/processed/news_scored.csv \
  --config config/default.json \
  --output-dir outputs/live_run
```

The market file needs `date`, `ticker`, `open`, `close`, and at least one numeric traditional factor. The scored-news file needs `published_at`, `ticker`, `title`, `content`, and `sentiment_score`.

## Repository layout

```text
config/default.json          Parameters for rolling windows, portfolio rules, and costs
scripts/fetch_tushare.py     Optional Tushare acquisition and conservative name matching
scripts/score_news.py        Checkpoint-agnostic Chinese finance sentiment scoring
scripts/run_pipeline.py      PCA-only vs Text-PCA experiment entry point
src/qnews/                   Validated pipeline, features, rolling SVM, and backtester
tests/                       Offline tests using deterministic synthetic data
data/                        Ignored local inputs and intermediate data
outputs/                     Ignored local models, reports, and backtests
```

## Important interpretation limits

- This is a research reproduction, not investment advice or a production trading system.
- The Top-10 daily design is a high-turnover strategy. A single 0.15% transaction-cost assumption is a scenario, not a complete A-share execution model; slippage, taxes, borrowability, suspensions, price limits, and liquidity constraints need separate treatment.
- Name matching is intentionally conservative and can miss aliases, abbreviations, and multi-company articles. Manual audit and a company-entity dictionary are recommended before research use.
- Historical CSI 500 constituent changes must be handled point-in-time. The current downloader unions index-weight responses across the requested dates; a production study should reconstruct membership per session.
- Synthetic demo results are intentionally not included in this README. Evaluate live-data results only after checking coverage, label alignment, out-of-sample dates, and cost assumptions.

## License

MIT. Data, model weights, and third-party APIs retain their own licenses and access terms.
