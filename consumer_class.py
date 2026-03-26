import pika
import os
import json
import time
import sys

class QueryConsumer:
    def __init__(self, topic=None, host='rabbitmq', exchange='query_router'):
        # Fallback to env var if topic isn't passed explicitly
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
                print(f"[*] Consumer [{self.topic}] connecting to {self.host}...", flush=True)
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=self.host)
                )
                self.channel = self.connection.channel()
                
                # Declare exchange
                self.channel.exchange_declare(
                    exchange=self.exchange, 
                    exchange_type='direct'
                )
                
                # Create a temporary exclusive queue for this specific consumer instance
                result = self.channel.queue_declare(queue='', exclusive=True)
                self.queue_name = result.method.queue
                
                # Bind the queue to the specific topic (routing_key)
                self.channel.queue_bind(
                    exchange=self.exchange, 
                    queue=self.queue_name, 
                    routing_key=self.topic
                )
                return
            except Exception as e:
                print(f"[!] RabbitMQ not ready for {self.topic}, retrying in 2s...", flush=True)
                time.sleep(2)

    def _callback(self, ch, method, properties, body):
        """Internal wrapper for processing messages."""
        try:
            data = json.loads(body)
            self.on_message_received(data)
        except Exception as e:
            print(f"[ERROR] Consumer {self.topic} failed to process: {e}", flush=True)

    def on_message_received(self, data):
        """
        Override this method or pass a logic function to 
        customize what happens when a message arrives.
        """
        print(f" [Consumer: {self.topic}] Received: {data['query']}", flush=True)

    def start(self):
        """Starts the consuming loop."""
        self._establish_connection()
        print(f" [*] Consumer [{self.topic}] waiting for queries. CTRL+C to exit", flush=True)
        
        self.channel.basic_consume(
            queue=self.queue_name, 
            on_message_callback=self._callback, 
            auto_ack=True
        )
        
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print(f"\n[!] Consumer {self.topic} stopping...")
            self.connection.close()


if __name__ == "__main__":
    consumer = QueryConsumer() 
    consumer.start()