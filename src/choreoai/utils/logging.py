"""
logging.py — Structured JSON logging for production.
"""

import json
import logging
import time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "latency": getattr(record, "latency", None),
            "gpu_usage": getattr(record, "gpu_usage", None)
        }
        return json.dumps(log_record)

def get_json_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
