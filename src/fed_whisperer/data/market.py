"""Market data loader for Treasury yields (FRED) and equity indices (yfinance)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """Pulls and caches daily Treasury yields from FRED and equity index prices via yfinance."""

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_dir: str | Path = ".cache/fed_whisperer/market",
    ) -> None:
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_fred_series(
        self,
        series_ids: Dict[str, str],
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily Treasury yield & fed funds series from FRED."""
        cache_file = self.cache_dir / f"fred_{start_date}_{end_date or 'latest'}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        if not self.fred_api_key:
            raise EnvironmentError(
                "FRED_API_KEY environment variable is not set. "
                "Export FRED_API_KEY=<your_key> before calling fetch_fred_series."
            )

        from fredapi import Fred

        fred = Fred(api_key=self.fred_api_key)
        frames: Dict[str, pd.Series] = {}
        for label, sid in series_ids.items():
            raw = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
            frames[label] = pd.to_numeric(raw, errors="coerce")

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().ffill()
        df.to_parquet(cache_file)
        return df

    def fetch_equity_series(
        self,
        ticker: str = "SPY",
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Fetch daily adjusted close for S&P 500 / SPY using yfinance."""
        safe_ticker = ticker.replace("^", "")
        cache_file = self.cache_dir / f"eq_{safe_ticker}_{start_date}_{end_date or 'latest'}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df["close"]

        import yfinance as yf

        hist = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if hist.empty:
            raise RuntimeError(f"Empty price history returned by yfinance for {ticker}")

        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.sort_index().rename("close")
        close.to_frame(name="close").to_parquet(cache_file)
        return close

    @staticmethod
    def compute_event_windows(
        event_dates: Iterable[str],
        market_df: pd.DataFrame,
        windows: List[int] = [1, 2, 5],
    ) -> pd.DataFrame:
        """
        Compute yield changes (in basis points) and equity returns (in %)
        around each FOMC statement release date for multiple window lengths.

        Also computes a rate surprise proxy: the 1-day change in the 2-year Treasury yield
        or target rate delta around the announcement.
        """
        df = market_df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        trading_days = df.index

        records: List[Dict[str, float | str]] = []
        for date_str in event_dates:
            dt = pd.to_datetime(date_str)
            # Find index on or immediately after release date
            pos = trading_days.searchsorted(dt)
            if pos <= 0 or pos >= len(trading_days):
                continue

            pre_idx = pos - 1
            row: Dict[str, float | str] = {"date": date_str}

            # Policy rate delta and rate surprise proxy
            if "dfedtaru" in df.columns:
                row["policy_rate_change_bps"] = float(
                    (df["dfedtaru"].iloc[pos] - df["dfedtaru"].iloc[pre_idx]) * 100.0
                )
            else:
                row["policy_rate_change_bps"] = 0.0

            # Rate surprise proxy: orthogonalized short-end move or unexpected target shift
            for w in windows:
                post_idx = min(pos + w - 1, len(trading_days) - 1)
                if "dgs2" in df.columns:
                    row[f"dgs2_chg_{w}d_bps"] = float(
                        (df["dgs2"].iloc[post_idx] - df["dgs2"].iloc[pre_idx]) * 100.0
                    )
                if "dgs10" in df.columns:
                    row[f"dgs10_chg_{w}d_bps"] = float(
                        (df["dgs10"].iloc[post_idx] - df["dgs10"].iloc[pre_idx]) * 100.0
                    )
                if "sp500" in df.columns:
                    p0 = float(df["sp500"].iloc[pre_idx])
                    p1 = float(df["sp500"].iloc[post_idx])
                    row[f"sp500_ret_{w}d_pct"] = ((p1 - p0) / p0) * 100.0 if p0 else np.nan

            records.append(row)

        return pd.DataFrame(records)
