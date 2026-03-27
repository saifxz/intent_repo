import logging
import os
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

# Context variable (safe for async FastAPI)
request_id_var = ContextVar("request_id", default="N/A")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


class Logger:
    _instance = None

    def __new__(cls, name="AppLogger", log_file="logs/app.log"):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)

            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)

            if not logger.handlers:
                formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] [%(name)s] "
                    "[Thread:%(thread)d] [ReqID:%(request_id)s] : %(message)s"
                )

                # Handlers
                console_handler = logging.StreamHandler()
                file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)

                console_handler.setFormatter(formatter)
                file_handler.setFormatter(formatter)

                # Add filter (IMPORTANT)
                console_handler.addFilter(RequestIdFilter())
                file_handler.addFilter(RequestIdFilter())

                logger.addHandler(console_handler)
                logger.addHandler(file_handler)

            cls._instance.logger = logger

        return cls._instance.logger