import pika
import json
import time
import random

import os

from prioritize_strategy import SentimentPriorityStrategy, IntentPriorityStrategy, HybridPriorityStrategy
from logger_config import Logger  

logger = Logger("AppLogger")  

# print("FILES:", os.listdir())
class QueryProducer:
    def __init__(self, name, engine, host='rabbitmq', exchange='query_router'):
        self.name = name
        self.engine = engine
        self.host = host
        self.exchange = exchange
        self.connection = None
        self.channel = None

        strategy_type = os.getenv("PRIORITY_STRATEGY", "sentiment")

        strategy_map = {
            "intent": IntentPriorityStrategy,
            "hybrid": HybridPriorityStrategy,
            "sentiment": SentimentPriorityStrategy
        }
        
        strategy_class = strategy_map.get(strategy_type, SentimentPriorityStrategy)
        self.priority_strategy = strategy_class()

        logger.info(f"[{self.name}] Using priority strategy: {strategy_type}")

    def _connect(self):
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host)
            )
            self.channel = self.connection.channel()
            self.channel.exchange_declare(
                exchange=self.exchange,
                exchange_type='direct'
            )

            logger.info(f"[{self.name}] Connected to RabbitMQ")

        except Exception as e:
            logger.error(f"[{self.name}] Connection Failed: {e}", exc_info=True)
            raise

    def publish_query(self, query_text , sentiment=None):

        try:
            # Ensure connection is alive
            if not self.channel or self.channel.is_closed:
                logger.warning(f"[{self.name}] Channel closed. Reconnecting...")
                self._connect()

            # Perform Inference
            result = self.engine.predict(query_text)
            intent = result.get("intent", "unknown")

            message = {
                "producer_name": self.name,
                "query": query_text,
                "category": intent,
                "sentiment": sentiment or "Neutral",
                "timestamp": time.time()
            }

            priority = self.priority_strategy.get_priority(message)
            logger.info(f"[{self.name}] Publishing intent '{intent}' for query: {query_text} with priority: {priority}")

            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key=str(intent),
                body=json.dumps(message),
                properties=pika.BasicProperties(priority=priority)
            )

            logger.info(f"[{self.name}] Successfully published message")

        except Exception as e:
            logger.error(f"[{self.name}] Publish Error: {e}", exc_info=True)

    def start_auto_produce(self, query_list, interval=0.5):

        logger.info(f"[{self.name}] Starting automated production...")

        while True:
            query = random.choice(query_list)
            self.publish_query(query)
            time.sleep(interval)