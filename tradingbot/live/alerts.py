"""
Minimal Telegram alerting. Non-blocking-ish (requests with short timeout) so a
network hiccup here never stalls the trading loop for long.

Extra: daily_summary() can be called at the start of each UTC day to recap
       prior-day performance without needing the dashboard.
"""
import logging
import requests
import time
from datetime import date, datetime

from config import AlertConfig

logger = logging.getLogger("alerts")


class Alerter:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self._last_sent: dict[str, float] = {}
        self.debounce_sec = 300  # 5 minutes for identical messages

        # Daily PnL tracking (reset at UTC midnight)
        self._daily: dict = self._fresh_daily()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fresh_daily() -> dict:
        return {
            "date":        date.today(),
            "trades":      0,
            "wins":        0,
            "losses":      0,
            "total_pnl":   0.0,
            "largest_loss": 0.0,
        }

    def _rotate_if_new_day(self):
        today = date.today()
        if self._daily["date"] != today:
            self._daily = self._fresh_daily()
            self._daily["date"] = today

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def record_trade(self, pnl: float):
        """Call after every trade close to keep intra-day stats current."""
        self._rotate_if_new_day()
        self._daily["trades"] += 1
        self._daily["total_pnl"] += pnl
        if pnl > 0:
            self._daily["wins"] += 1
        else:
            self._daily["losses"] += 1
            if pnl < self._daily["largest_loss"]:
                self._daily["largest_loss"] = pnl

    def send_daily_summary(self):
        """
        Send a recap of the current day's closed trades.
        Safe to call repeatedly — debouncing prevents duplicate messages.
        """
        self._rotate_if_new_day()
        d = self._daily
        if d["trades"] == 0:
            msg = "📊 [DAILY SUMMARY] No trades closed today."
        else:
            win_rate = (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else 0.0
            pnl_emoji = "✅" if d["total_pnl"] >= 0 else "❌"
            msg = (
                f"📊 [DAILY SUMMARY] {d['date']}\n"
                f"{pnl_emoji} Net PnL: ${d['total_pnl']:,.2f}\n"
                f"📈 Trades: {d['trades']} | Wins: {d['wins']} | Losses: {d['losses']}\n"
                f"🎯 Win Rate: {win_rate:.1f}%\n"
                f"📉 Largest Loss: ${d['largest_loss']:,.2f}"
            )
        self.send(msg, force=True)

    def send(self, message: str, force: bool = False):
        """
        Send a Telegram message.

        Args:
            message: The text to send.
            force:   If True, bypass the 5-minute deduplication debounce.
                     Use for critical alerts like kill switch or daily summaries.
        """
        logger.info("ALERT: %s", message)
        if not self.cfg.enabled:
            return

        now = time.time()
        if not force and message in self._last_sent:
            if now - self._last_sent[message] < self.debounce_sec:
                logger.debug("Debouncing identical alert: %s", message)
                return

        self._last_sent[message] = now

        # Cleanup old messages from cache to prevent unbounded growth
        self._last_sent = {
            k: v for k, v in self._last_sent.items()
            if now - v < self.debounce_sec
        }

        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        try:
            requests.post(
                url,
                json={"chat_id": self.cfg.telegram_chat_id, "text": message},
                timeout=(5, 15),
            )
        except requests.RequestException as e:
            logger.warning("Failed to send Telegram alert: %s", e)
