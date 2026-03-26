import pika
import json
import time
import random

class QueryProducer:
    def __init__(self, name, engine, host='rabbitmq', exchange='query_router'):
        self.name = name
        self.engine = engine
        self.host = host
        self.exchange = exchange
        self.connection = None
        self.channel = None

    def _connect(self):
        """Internal method to establish RabbitMQ connection."""
        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
            self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange=self.exchange, exchange_type='direct')
            print(f"--- [Producer: {self.name}] Connected to RabbitMQ ---")
        except Exception as e:
            print(f"--- [Producer: {self.name}] Connection Failed: {e} ---")
            raise

    def publish_query(self, query_text):
        """Predicts intent and publishes to the exchange."""
        try:
            # Ensure connection is alive
            if not self.channel or self.channel.is_closed:
                self._connect()

            # Perform Inference
            result  = self.engine.predict(query_text)
            intent = result.get("intent", "unknown")
            message = {
                "producer_name": self.name,
                "query": query_text, 
                "category": intent,
                "timestamp": time.time()
            }

            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key=str(intent), 
                body=json.dumps(message)
            )
            print(f"[{self.name}] Published intent '{intent}': {query_text}")
            
        except Exception as e:
            print(f"[{self.name}] Publish Error: {e}")

    def start_auto_produce(self, query_list, interval=0.5):
        """Starts a loop to produce random queries indefinitely."""
        print(f"[{self.name}] Starting automated production...")
        while True:
            query = random.choice(query_list)
            self.publish_query(query)
            # time.sleep(interval)