import pika

# Connect to RabbitMQ
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

# Declare a queue
channel.queue_declare(queue='my_queue')

# Send a message to the queue
channel.basic_publish(exchange='',
                      routing_key='my_queue',
                      body='Hello, World!')

# Close the connection
connection.close()