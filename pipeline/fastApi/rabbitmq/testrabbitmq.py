import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output
import pika

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('RabbitMQ Example'),
    html.Div(id='message'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('message', 'children'),
              [dash.dependencies.Input('interval-component', 'n_intervals')])
def update_message(n):
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    # Declare the queue
    channel.queue_declare(queue='my_queue')

    # Get a message from the queue
    method_frame, header_frame, body = channel.basic_get('my_queue')

    # If a message was received, decode and return the message
    if method_frame:
        channel.basic_ack(method_frame.delivery_tag)
        connection.close()
        return body.decode()
    else:
        connection.close()
        return 'No new messages'

if __name__ == '__main__':
    app.run_server(debug=True)