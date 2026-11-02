"""FOMC statement scraper with robots.txt compliance, rate limiting, and disk caching."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class FOMCStatement:
    """Represents a single FOMC monetary policy press release."""

    date: str  # YYYY-MM-DD
    url: str
    title: str
    text: str
    sentences: List[str]


def split_sentences(text: str) -> List[str]:
    """Split FOMC statement prose into clean analytical sentences."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
    sentences: List[str] = []
    for part in raw_parts:
        s = part.strip()
        if len(s) < 25:
            continue
        # Filter boilerplate media contact lines or voting tallies if desired
        if s.lower().startswith("for media inquiries") or s.lower().startswith("implementation note"):
            continue
        sentences.append(s)
    return sentences


class FOMCScraper:
    """Scrapes FOMC monetary policy statements from federalreserve.gov."""

    STATEMENT_HREF_RE = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")

    def __init__(
        self,
        base_url: str = "https://www.federalreserve.gov",
        calendar_url: str = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        user_agent: str = "FedWhispererResearchBot/1.0",
        rate_limit_seconds: float = 1.5,
        cache_dir: str | Path = ".cache/fed_whisperer/statements",
        timeout: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.calendar_url = calendar_url
        self.user_agent = user_agent
        self.rate_limit_seconds = rate_limit_seconds
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._last_request_ts: float = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self.robot_parser = self._init_robots()

    def _init_robots(self) -> Optional[RobotFileParser]:
        robots_url = f"{self.base_url}/robots.txt"
        rp = RobotFileParser()
        try:
            resp = self.session.get(robots_url, timeout=self.timeout)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                return rp
        except requests.RequestException as exc:
            logger.warning("Could not fetch robots.txt (%s); proceeding with caution.", exc)
        return None

    def is_allowed(self, url: str) -> bool:
        """Verify URL against robots.txt rules."""
        if self.robot_parser is None:
            return True
        return self.robot_parser.can_fetch(self.user_agent, url)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_ts = time.monotonic()

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{key}_{digest}.json"

    def fetch_statement_links(self) -> Dict[str, str]:
        """Return mapping of release date (YYYY-MM-DD) -> full statement URL."""
        if not self.is_allowed(self.calendar_url):
            raise PermissionError(f"Blocked by robots.txt: {self.calendar_url}")

        self._throttle()
        resp = self.session.get(self.calendar_url, timeout=self.timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        date_to_url: Dict[str, str] = {}
        for anchor in soup.find_all("a", href=True):
            match = self.STATEMENT_HREF_RE.search(anchor["href"])
            if match:
                raw_date = match.group(1)
                iso_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                full_url = urljoin(self.base_url, anchor["href"])
                date_to_url[iso_date] = full_url

        return dict(sorted(date_to_url.items()))

    def parse_statement_html(self, html: str, date: str, url: str) -> FOMCStatement:
        """Extract clean policy statement paragraphs from Federal Reserve HTML."""
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.find("h3", class_="title") or soup.find("title")
        title = title_el.get_text(strip=True) if title_el else f"FOMC Statement {date}"

        article = (
            soup.find("div", id="article")
            or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
            or soup
        )

        paragraphs: List[str] = []
        for p in article.find_all("p"):
            txt = " ".join(p.get_text(" ", strip=True).split())
            if not txt:
                continue
            if txt.lower().startswith("for release at") or txt.lower().startswith("for media inquiries"):
                continue
            if txt.lower().startswith("implementation note issued"):
                continue
            paragraphs.append(txt)

        full_text = "\n\n".join(paragraphs)
        sentences = split_sentences(full_text)
        return FOMCStatement(
            date=date,
            url=url,
            title=title,
            text=full_text,
            sentences=sentences,
        )

    def fetch_statement(self, date: str, url: str, force_refresh: bool = False) -> FOMCStatement:
        """Fetch a single FOMC statement with local JSON caching."""
        cache_file = self._cache_path(date)
        if cache_file.exists() and not force_refresh:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return FOMCStatement(**payload)

        if not self.is_allowed(url):
            raise PermissionError(f"Blocked by robots.txt: {url}")

        self._throttle()
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        statement = self.parse_statement_html(resp.text, date=date, url=url)
        cache_file.write_text(json.dumps(asdict(statement), indent=2), encoding="utf-8")
        return statement

    def fetch_all(self, limit: Optional[int] = None, force_refresh: bool = False) -> List[FOMCStatement]:
        """Fetch and cache all statements found on the FOMC calendar page."""
        links = self.fetch_statement_links()
        items = list(links.items())
        if limit is not None:
            items = items[-limit:]

        statements: List[FOMCStatement] = []
        for date, url in items:
            try:
                stmt = self.fetch_statement(date, url, force_refresh=force_refresh)
                if stmt.text:
                    statements.append(stmt)
            except Exception as exc:
                logger.error("Failed to fetch statement for %s (%s): %s", date, url, exc)
        return sorted(statements, key=lambda s: s.date)
