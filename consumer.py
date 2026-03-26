import pika
import os
import json
import time
import sys

MY_TOPIC = os.getenv('TOPIC', 'general')

def callback(ch, method, properties, body):
    try:
        data = json.loads(body)
        print(f" [Consumer: {MY_TOPIC}] Received: {data['query']}", flush=True)
    except Exception as e:
        print(f"Error processing message: {e}", flush=True)

def get_connection():
    while True:
        try:
            print(f"Consumer [{MY_TOPIC}] connecting to RabbitMQ...", flush=True)
            return pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
        except Exception as e:
            print(f"RabbitMQ not ready for {MY_TOPIC}, retrying in 2s...", flush=True)
            time.sleep(2)

connection = get_connection()
channel = connection.channel()



channel.exchange_declare(exchange='query_router', exchange_type='direct')


result = channel.queue_declare(queue='', exclusive=True)
queue_name = result.method.queue


channel.queue_bind(exchange='query_router', queue=queue_name, routing_key=MY_TOPIC)

print(f" [*] Consumer [{MY_TOPIC}] waiting for queries. To exit press CTRL+C", flush=True)
channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
channel.start_consuming()