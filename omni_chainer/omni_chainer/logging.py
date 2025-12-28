# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import logging
import logging.config

from omni_chainer.core.config import settings


_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "correlation_id": {
            "()": "asgi_correlation_id.CorrelationIdFilter",
            "name": "correlation_id",
            "uuid_length": 36,
            "default_value": "-"
        }
    },
    "formatters": {
        'access': {
            '()': 'uvicorn.logging.AccessFormatter',
            'fmt': '%(levelprefix)s %(correlation_id)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True
        },
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(correlation_id)s %(asctime)s - %(pathname)s:%(lineno)d %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True
        },
    },
    "handlers": {
        'access': {
            'class': 'logging.StreamHandler',
            'formatter': 'access',
            'stream': 'ext://sys.stdout',
            'filters': ['correlation_id']
        },
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "filters": ['correlation_id']
        },
    },
    "loggers": {
        "omni_chainer": {
            "handlers": ["default"],
            "level": "DEBUG" if settings.ENV_STAGE == "development" else "INFO",
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["default"],
            "level": "DEBUG" if settings.ENV_STAGE == "development" else "INFO",
            "propagate": True
        },
        'uvicorn.access': {
            'handlers': ['access'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.error': {
            'level': 'INFO',
            'propagate': False
        }
    },
}


logging.config.dictConfig(_LOG_CONFIG)
logger = logging.getLogger("omni_chainer")
