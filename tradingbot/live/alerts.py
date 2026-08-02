"""
Minimal Telegram alerting. Non-blocking-ish (requests with short timeout) so a
network hiccup here never stalls the trading loop for long.
"""
import logging
import requests
import time

from config import AlertConfig

logger = logging.getLogger("alerts")


class Alerter:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self._last_sent = {}
        self.debounce_sec = 300  # 5 minutes for identical messages

    def send(self, message: str):
        logger.info("ALERT: %s", message)
        if not self.cfg.enabled:
            return
            
        now = time.time()
        if message in self._last_sent:
            if now - self._last_sent[message] < self.debounce_sec:
                logger.debug("Debouncing identical alert: %s", message)
                return
                
        self._last_sent[message] = now
        
        # Cleanup old messages from cache to prevent unbounded growth
        self._last_sent = {k: v for k, v in self._last_sent.items() if now - v < self.debounce_sec}
        
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        try:
            requests.post(
                url,
                json={"chat_id": self.cfg.telegram_chat_id, "text": message},
                timeout=(5, 15),
            )
        except requests.RequestException as e:
            logger.warning("Failed to send Telegram alert: %s", e)

