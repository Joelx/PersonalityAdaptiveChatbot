import pika

class RabbitMQ:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.credentials = pika.PlainCredentials(username, password)
        self.connection = None
        self.channel = None

    def connect(self):
        if not self.connection or self.connection.is_closed:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=self.host, port=self.port, credentials=self.credentials))
            self.channel = self.connection.channel()

    def send_message(self, exchange_name, routing_key, message):
        self.connect()
        self.channel.exchange_declare(exchange=exchange_name, exchange_type='direct')
        self.channel.basic_publish(exchange=exchange_name, routing_key=routing_key, body=message)

    def receive_message(self, exchange_name, queue_name, routing_key):
        self.connect()
        self.channel.exchange_declare(exchange=exchange_name, exchange_type='direct')
        self.channel.queue_declare(queue=queue_name)
        self.channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=routing_key)
        method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
        if method_frame is not None:
            self.channel.basic_ack(method_frame.delivery_tag)
        return body

    def close(self):
        if self.connection is not None and self.connection.is_open:
            self.connection.close()