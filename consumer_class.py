import pika
import os
import json
import time
import sys

from logger_config import Logger, request_id_var  

logger = Logger("AppLogger") 


class QueryConsumer:
    def __init__(self, topic=None, host='rabbitmq', exchange='query_router'):
        self.topic = topic or os.getenv('TOPIC', 'general')
        self.host = host
        self.exchange = exchange
        self.connection = None
        self.channel = None
        self.queue_name = None

    def _establish_connection(self):
        """Retries connection until RabbitMQ is available."""
        while True:
            try:
                logger.info(f"[*] Consumer [{self.topic}] connecting to {self.host}...")

                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=self.host)
                )
                self.channel = self.connection.channel()

                # Declare exchange
                self.channel.exchange_declare(
                    exchange=self.exchange,
                    exchange_type='direct'
                )

                # Create temporary queue
                result = self.channel.queue_declare(queue='', exclusive=True)
                self.queue_name = result.method.queue

                # Bind queue
                self.channel.queue_bind(
                    exchange=self.exchange,
                    queue=self.queue_name,
                    routing_key=self.topic
                )

                logger.info(f"[{self.topic}] Connected and bound to exchange '{self.exchange}'")
                return

            except Exception as e:
                logger.warning(
                    f"[{self.topic}] RabbitMQ not ready, retrying in 2s... Error: {e}"
                )
                time.sleep(2)

    def _callback(self, ch, method, properties, body):
        """Internal wrapper for processing messages."""
        try:
            data = json.loads(body)

            # 🔥 Set request_id from message into context
            req_id = data.get("request_id", "N/A")
            request_id_var.set(req_id)

            logger.info(f"[{self.topic}] Message received")

            self.on_message_received(data)

        except Exception as e:
            logger.error(
                f"[{self.topic}] Failed to process message: {e}",
                exc_info=True
            )

    def on_message_received(self, data):
        """
        Override this method to customize processing logic.
        """
        logger.info(
            f"[{self.topic}] Processing query: {data.get('query')} "
            f"(Intent: {data.get('category')})"
        )

    def start(self):
        """Starts the consuming loop."""
        self._establish_connection()

        logger.info(
            f"[*] Consumer [{self.topic}] waiting for queries. CTRL+C to exit"
        )

        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self._callback,
            auto_ack=True
        )

        try:
            self.channel.start_consuming()

        except KeyboardInterrupt:
            logger.warning(f"[!] Consumer {self.topic} stopping...")
            if self.connection:
                self.connection.close()


if __name__ == "__main__":
    consumer = QueryConsumer()
    consumer.start()