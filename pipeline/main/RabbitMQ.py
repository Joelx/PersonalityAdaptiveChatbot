import pika
import os


rabbit_username = os.environ['RABBITMQ_USERNAME']
rabbit_password = os.environ['RABBITMQ_PASSWORD']
erlang_cookie = os.environ['RABBITMQ_ERLANG_COOKIE']
rabbit_host = "10.1.81.44"
rabbit_port = 5672

"""
Credentials for localhost development
"""
# rabbit_username = "guest"
# rabbit_password ="guest"
# rabbit_host = "localhost"
# rabbit_port = 5672
rabbit_credentials = pika.PlainCredentials(rabbit_username, rabbit_password)


"""
RabbitMQ send and receive functions.
Since we need intervals and callback functions in dash to pull the information,
we need to create a separate rabbit connection for each pull. 
This is not efficient, however, otherwise we would need 
to implement Threading and we dont have unlimited time. 
"""
def send_to_rabbitmq(data, queue, sender_id):
    if data and queue and sender_id:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbit_host, rabbit_port, '/', rabbit_credentials))
        channel = connection.channel()

        queue_name = queue + '-queue-' + sender_id
        exchange_name = queue + '-exchange-' + sender_id
        routing_key = queue + '-' + sender_id

        # Declare the queue
        channel.queue_declare(queue=queue_name)

        # Declare the exchange
        channel.exchange_declare(exchange=exchange_name, exchange_type='direct')

        # Bind the queue to the exchange
        channel.queue_bind(queue=queue_name, exchange=exchange_name, routing_key=routing_key)

        # Publish the data to the exchange
        channel.basic_publish(exchange=exchange_name, routing_key=routing_key, body=data)

        connection.close()

def receive_rabbitmq(queue, sender_id):
    if queue and sender_id:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbit_host, rabbit_port, '/', rabbit_credentials))
        channel = connection.channel()
        exchange_name = queue + '-exchange-' + sender_id
        queue_name = queue + '-queue-' + sender_id
        routing_key = queue + '-' + sender_id

        #print(f"RabbitMQ  RECEIVE data: {exchange_name}, {routing_key}, {queue_name}")

        channel.exchange_declare(exchange=exchange_name, exchange_type='direct')
        channel.queue_declare(queue=queue_name)
        channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=routing_key)
        method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
        connection.close()
        return body
