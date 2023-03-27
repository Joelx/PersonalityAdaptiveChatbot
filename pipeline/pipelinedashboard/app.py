import asyncio
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import pika
import json
import dash
from dash.dependencies import Input, Output
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
import evaluations
import os
import time

EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css", dbc.themes.BOOTSTRAP]

os.environ["OPENAI_API_KEY"] = "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"
# rabbit_username = os.environ['RABBITMQ_USERNAME']
# rabbit_password = os.environ['RABBITMQ_PASSWORD']
# erlang_cookie = os.environ['RABBITMQ_ERLANG_COOKIE']
# rabbit_host = "10.1.81.44"
rabbit_username = "guest"
rabbit_password ="guest"
rabbit_host = "localhost"
rabbit_port = 5672
rabbit_credentials = pika.PlainCredentials(rabbit_username, rabbit_password)

previous_tsne_fig = None
previous_wordcloud = previous_fq_fig = previous_tm = previous_alert = {}
previous_sentiment = None
previous_features = None
previous_classification = None
previous_prompt = None
current_user_text = ""

# List of available machine learning models
models = ['Model A', 'Model B', 'Model C']


INTERVAL_COMPONENTS = [
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    dcc.Interval(id='interval-component-15', interval=15000, n_intervals=0)
]

# MODEL_SELECT = [
#     dcc.Dropdown(
#         id='model-dropdown',
#         options=[{'label': model, 'value': model} for model in models],
#         value=models[0]
#     ),
#     html.Div(id='models-output')
# ]


EVALUATION_STARTER = [
    dbc.Card(
        [
            dbc.CardHeader(html.H4(children="Evaluation", className="lead")),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "Start Evaluation Run",
                                id="eval-button",
                                color="primary",
                                #block=True,
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            dbc.Spinner(
                                html.Div(
                                    id="output-div",
                                    className="output-div",
                                )
                            )
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Alert(
                            "Evaluation is currently running...",
                            id="eval-alert",
                            color="warning",
                            style={"display": "none"},
                        )
                    )
                ]
            ),
        ]
    )
        ]
    )    
]

TSNE_PLOT = [
    dcc.Graph(id='scatter-plot')
]

SENTIMENT_OUTPUT = [
    dbc.CardHeader(html.H4(children="Sentiments", className="lead")),
    dbc.CardBody(
    html.Div(children=[
        html.P(id='sentiment-output-row', children='Calulated no sentiments yet', 
            style={"fontSize": 14, "margin-left": 6, "margin-top": 6},),
            ]))
]

FEATURE_OUTPUT = [
    dbc.CardHeader(html.H4(children="Selected features", className="lead")),
    dbc.CardBody(
    html.Div(id='features-output-row', children=[
        html.P(children='No features calculated yet',
               style={"fontSize": 14, "margin-left": 6, "margin-top": 6},),
    ]),)
]

CLASSIFICATION_THRESHOLDS = dbc.Card([
    dbc.CardHeader(html.H4(children="Classification Thresholds", className="lead")),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H6("Input"),
                html.Label('Neuroticism: ', style={'margin-right': '10px'}),
                dcc.Input(
                    id='neuroticism-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.578,
                ),
                html.Br(),
                html.Label('Extraversion: ', style={'margin-right': '10px'}),
                dcc.Input(
                    id='extraversion-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.478,
                ),
                html.Br(),
                html.Label('Openness: ', style={'margin-right': '10px'}),
                dcc.Input(
                    id='openness-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.178,
                ),
                html.Br(),
                html.Label('Agreeableness: ', style={'margin-right': '10px'}),
                dcc.Input(
                    id='agreeableness-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.494,
                ),
                html.Br(),
                html.Label('Conscientiousness: ', style={'margin-right': '10px'}),
                dcc.Input(
                    id='conscientiousness-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.299,
                ),
            ], md=6),
            dbc.Col([
                html.H6("Check actual config fetched by pipeline server"),
                html.Div(id='output'),
            ], md=6),
        ]),
    ]),
])

CLASSIFICATION_RESULT = [
    dbc.CardHeader(html.H4(children="Big5 Classification Results", className="lead")),
    dbc.CardBody(
    html.Div(children=[
        html.Div(id='cf-output-row', children='Calulated no classifications yet', 
            style={"fontSize": 14, "margin-left": 6, "margin-top": 6},),
            ]))
]

WORDCLOUD_PLOT = [
    dbc.CardHeader(html.H4(children="Most frequently used words in conversation", className="lead")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-frequencies",
                            children=[dcc.Graph(id="frequency_figure")],
                            type="default",
                        )
                    ),
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Treemap",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[dcc.Graph(id="bank-treemap")],
                                                type="default",
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Wordcloud",
                                        children=[
                                            dcc.Loading(
                                                id="loading-wordcloud",
                                                children=[
                                                    dcc.Graph(id="bank-wordcloud")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                        md=8,
                    ),
                ]
            )
        ]
    ),
]

CURRENT_PROMPT = [
    dbc.CardHeader(html.H4(children="Current Prompt", className="lead")),
    dbc.CardBody(
        html.Div(id="prompt-output-row", 
            children=html.P(children="No prompt fetched yet", style={"white-space": "pre-wrap"}),
            style={"height": "200px", "overflow-y": "scroll"}
        )
    )
]

HEADER = dbc.Container(
    [
        dbc.Card(INTERVAL_COMPONENTS)
    ]
)

BODY = dbc.Container(
    [
        #dbc.Row([dbc.Col(dbc.Card(MODEL_SELECT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(EVALUATION_STARTER),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TSNE_PLOT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(SENTIMENT_OUTPUT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(FEATURE_OUTPUT)),], style={"marginTop": 30}),
        dbc.Card(WORDCLOUD_PLOT, style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(CLASSIFICATION_THRESHOLDS)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(CLASSIFICATION_RESULT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(CURRENT_PROMPT)),], style={"marginTop": 30, "marginBottom": 30}),
       # dbc.Row([dbc.Col([])], style={"marginTop": 50}),
    ],
    className="mt-12",
)

server = FastAPI()
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/dashboard/')
app.layout = html.Div(children=[HEADER, BODY])

"""
RabbitMQ send and receive functions.
Since we need intervals and callback functions in dash to pull the information,
we need to create a separate rabbit connection for each pull. 
This is not efficient, however, otherwise we would need 
to implement Threading and we dont have unlimited time. 
"""
def send_to_rabbitmq(data, queue):
    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbit_host, rabbit_port, '/', rabbit_credentials))
    channel = connection.channel()

    exchange_name = queue + '-exchange'
    routing_key = queue + '-routing-key'

    channel.exchange_declare(exchange=exchange_name, exchange_type='direct')
    channel.basic_publish(exchange=exchange_name, routing_key=routing_key, body=data)

def receive_rabbitmq(queue):
    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbit_host, rabbit_port, '/', rabbit_credentials))
    channel = connection.channel()
    exchange_name = queue + '-exchange'
    queue_name = queue + '-queue'
    routing_key = queue + '-routing-key'
    channel.exchange_declare(exchange=exchange_name, exchange_type='direct')
    channel.queue_declare(queue=queue_name)
    channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=routing_key)
    method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
    connection.close()
    return body



@app.callback(
    Output("eval-button", "disabled"),
    Output("eval-alert", "style"),
    Output("output-div", "children"),
    Input("eval-button", "n_clicks"),
)
def eval_button_callback(n_clicks):
    if not n_clicks:
        return False, {"display": "none"}, ""
    
    time.sleep(2) # wait for 2 seconds
    return asyncio.run(start_eval_run(n_clicks)) # execute start_eval_run coroutine after 2 seconds

async def start_eval_run(n_clicks):
    if not n_clicks:
        return False, {"display": "none"}, ""

    bot = evaluations.RasaSocketIOBot()
    await bot.connect("https://joel-schlotthauer.com/rasax/socket.io/")
    await bot.start_conversation()
    conversation_result = bot.conversator.memory.buffer
    conversation_text = "\n".join(conversation_result)
    
    return True, {"display": "block"}, dcc.Textarea(
        value=conversation_text,
        readOnly=True,
        className="conversation-text",
        style={"width": "100%", "height": "calc(100vh - 200px)", "resize": "none"},
    )



"""
MODEL SELECT
"""
# @app.callback(Output('models-output', 'children'), [Input('model-dropdown', 'value')])
# def display_selected_model(model):
#     return html.Div([
#         html.H4(children='Selected model: {}'.format(model), className="lead", style={"margin-left": 6, "margin-top": 6}),
#         html.P('Description of the {} model: ...'.format(model),
#                style={"fontSize": 14, "margin-left": 6, "margin-top": 6})
#     ])


"""
TSNE PLOT
"""
@app.callback(dash.dependencies.Output('scatter-plot', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_tsne_plot(n):
    global previous_tsne_fig, x,y 
    body = receive_rabbitmq("embeddings")
    # If a message was received, decode and return the message    
    if body:
        data = json.loads(body)
        vectors = data['vectors']
        words = data['words']
        n_samples = len(vectors)

        # Calculate the maximum value of perplexity
        max_perplexity = min(30, n_samples - 1)
        # Perform t-SNE on the vectors
        tsne_data = TSNE(n_components=2, perplexity=max_perplexity).fit_transform(np.asarray(vectors))
        x=tsne_data[:,0]
        y=tsne_data[:,1]
        # Create a scatter plot using Plotly Express
        trace = go.Scatter(x=x, y=y, text=words, mode='markers', marker=dict(size=10))
        fig = go.Figure(data=[trace])
        fig.update_layout(title='TSNE Embeddings')
        # store the current data in the previous_tsne_fig variable
        previous_tsne_fig = fig

    else:
        # If there is no new data, use the previous_tsne_fig to update the plot
        if previous_tsne_fig:
            # Create a scatter plot using Plotly Express
            fig = previous_tsne_fig
        else: 
            fig = go.Figure()
    return fig


@app.callback(
    Output('sentiment-output-row', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_sentiment(n):
    global previous_sentiment 
    body = receive_rabbitmq("sentiment")
    # If a message was received, decode and return the message  
    if body:
        data = json.loads(body)
        sentiment = data['sentiment']
        #sentiment = json.dumps(sentiment)
        sentiment_string = f"""Sentiments in Percent:
        Negative: {round(sentiment[0]*100,2)} % |
        Neutral: {round(sentiment[1]*100,2)} % |
        Positive: {round(sentiment[2]*100,2)} %
        """
        #sentiment_html = [html.P(sentiment)]
        previous_sentiment = sentiment_string
    else:
        # If there is no new data, use the previous_tsne_fig to update the plot
        if previous_sentiment:
            # Create a scatter plot using Plotly Express
            sentiment_string = previous_sentiment
        else: 
            sentiment_string = "Calculated no sentiment yet"
    return sentiment_string

@app.callback(
    Output('features-output-row', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_features(n):
    global previous_features
    body = receive_rabbitmq("features")
    # If a message was received, decode and return the message  
    if body:
        data = json.loads(body)
        features = data['features']
        features_html = []
        for feature in features:
            features_html.append(html.Strong(children=feature.title(),
                                         style={"fontSize": 12, "fontWeight": "bold", "margin-left": 6, "margin-top": 6}, className="lead"),)
            features_html.append(html.P(children="Num of Features: " + str(features[feature]["num_of_features"]),
                                        style={"fontSize": 14, "margin-left": 6, "margin-top": 6},),)
            features_html.append(html.P(children="Features: " + ', '.join(features[feature]["feature_names"]),
                                        style={"fontSize": 14, "margin-left": 6, "margin-top": 6},),)
        previous_features = features_html
    else:
        # If there is no new data, use the previous_tsne_fig to update the plot
        if previous_features:
            # Create a scatter plot using Plotly Express
            features_html = previous_features
        else: 
            features_html = [html.H4(children="No features calculated yet", 
                                     style={"fontSize": 14, "margin-left": 6, "margin-top": 6},),]
    return features_html

"""
WORDCLOUD
"""
@app.callback(
    [
        Output("bank-wordcloud", "figure"),
        Output("frequency_figure", "figure"),
        Output("bank-treemap", "figure"),
        Output("no-data-alert", "style"),
    ],
    [Input('interval-component-15', 'n_intervals')]
)
def update_wordcloud(n):
    global previous_wordcloud, previous_fq_fig, previous_alert, previous_tm, current_user_text
    # Connect to RabbitMQ queue
    body = receive_rabbitmq("text")

    # If a message was received, decode and return the message
    if body:
        text = json.loads(body)['conversation_history']
        current_user_text = text
        wordcloud, frequency_figure, treemap = plotly_wordcloud(text)
        alert_style = {"display": "none"}
        if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}):
            alert_style = {"display": "block"}
        previous_wordcloud = wordcloud
        previous_fq_fig = frequency_figure
        previous_tm = treemap
        previous_alert = alert_style
    else:
        # If there is no new data, use the previous_wordcloud
        if previous_wordcloud and previous_fq_fig and previous_alert and previous_tm:
            # Create wordcloud
            wordcloud = previous_wordcloud
            frequency_figure = previous_fq_fig
            treemap = previous_tm
            alert_style = previous_alert
        else:
            wordcloud = frequency_figure = treemap = previous_fq_fig = {}
            alert_style = {"display": "block"}
    #print("redrawing conversation-wordcloud...done")
    return (wordcloud, frequency_figure, treemap, alert_style)


# Define the callback function to update the output
@app.callback(
    Output('output', 'children'),
    [Input('openness-threshold', 'value'),
     Input('conscientiousness-threshold', 'value'),
     Input('extraversion-threshold', 'value'),
     Input('agreeableness-threshold', 'value'),
     Input('neuroticism-threshold', 'value')]
)
def update_output(openness_threshold, conscientiousness_threshold, extraversion_threshold, agreeableness_threshold, neuroticism_threshold):
    threshold_json =  {
        "neuroticism": neuroticism_threshold,
        "extraversion": extraversion_threshold,
        "openness": openness_threshold,
        "agreeableness": agreeableness_threshold,
        "conscientiousness": conscientiousness_threshold
    }
    send_to_rabbitmq(json.dumps(threshold_json), "thresholds")

    body = receive_rabbitmq("actual-thresholds")

    if body:
        actual_thresholds = json.loads(body)
        neuroticism_threshold = actual_thresholds["neuroticism"]
        extraversion_threshold = actual_thresholds["extraversion"]
        openness_threshold = actual_thresholds["openness"]
        agreeableness_threshold = actual_thresholds["agreeableness"]
        conscientiousness_threshold = actual_thresholds["conscientiousness"]

        return html.Div([
            html.Span(f'Neuroticism Threshold: {neuroticism_threshold}'),
            html.Br(),
            html.Span(f'Extraversion Threshold: {extraversion_threshold}'),
            html.Br(),
            html.Span(f'Openness Threshold: {openness_threshold}'),
            html.Br(),
            html.Span(f'Agreeableness Threshold: {agreeableness_threshold}'),
            html.Br(),
            html.Span(f'Conscientiousness Threshold: {conscientiousness_threshold}'),
        ])
    else:
        return html.Div(html.Span("Fetched no real configuration from pipeline server yet."))


@app.callback(
    Output('cf-output-row', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_classification(n):
    global previous_classification 
    body = receive_rabbitmq("classification")
    # If a message was received, decode and return the message  
    if body:        
        data = json.loads(body)
        classes = data['classes']
        probas = data['probabilities']
        overall_classification_html = []
        classification_html = []
        classification_html.append(html.H6(children="Pipeline Classification:", 
                                           style={"fontSize": 14}, className="lead"))
        for cs in classes:
            classification_html.append(html.Strong(children=cs.title(),
                                         style={"display": "inline", "fontSize": 14, "fontWeight": "bold", }, className="lead"),)
            classification_html.append(html.P(children=f"CF{classes[cs]}, Proba: [{round(float(probas[cs][0]),2)}]",
                                        style={"fontSize": 12, "font-weight": "lighter"},),)
        print(classes)
        if (current_user_text):
            gpt_classification_html = []
            gpt_classification_html.append(html.H6(children="Evaluation Classification by gpt-3.5-turbo:", 
                                        style={"fontSize": 12}, className="lead"))
            gpt35_classifier = evaluations.BigFiveClassificationEvaluator()
            gpt35_cf_res = gpt35_classifier.classify(current_user_text)
            print(gpt35_cf_res)
            cf_lines = gpt35_cf_res.split('\n')
            print(cf_lines)
            for line in cf_lines:
                gpt_classification_html.append(html.P(children=line, style={"fontSize": 14, "font-weight": "lighter"}))
            #gpt_classification_html.append(html.P(children=[html.Strong(f"{cf_lines[i].split(':')[0]}: ") + f"{cf_lines[i].split(':')[1]}<br/>" if i < 5 else cf_lines[i] for i in range(len(cf_lines))],
            #                            style={"fontSize": 14, "margin-left": 6, "margin-top": 6}, className="lead"))
            classification_col1 = html.Div(classification_html)
            classification_col2 = html.Div(gpt_classification_html)
            overall_classification_html = dbc.Row([   dbc.Col(classification_col1, width=6),    dbc.Col(classification_col2, width=6)])
        else: 
            overall_classification_html = html.Div(classification_html)
        previous_classification = overall_classification_html
    else:
        # If there is no new data, use the previous_tsne_fig to update the plot
        if previous_classification:
            # Create a scatter plot using Plotly Express
            overall_classification_html = previous_classification
        else: 
            overall_classification_html = "Calculated no classification yet"
    return overall_classification_html

@app.callback(
    Output('prompt-output-row', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_prompt(n):
    global previous_prompt
    body = receive_rabbitmq("prompt")
    # If a message was received, decode and return the message  
    if body:        
        data = json.loads(body)
        token_size = data['token_size']
        prompt_lines = data['prompt'].split('\n')
        prompt_html = [html.P(line) for line in prompt_lines]
        prompt_html = [html.P(f"Current Token-Size: {token_size}")] + prompt_html
        prompt_html = html.Div(prompt_html)
        previous_prompt = prompt_html
    else:
        # If there is no new data, use the previous_tsne_fig to update the plot
        if previous_prompt:
            # Create a scatter plot using Plotly Express
            prompt_html = previous_prompt
        else: 
            prompt_html = "Fetched no prompt yet"
    return prompt_html


def plotly_wordcloud(text):
    """A wonderful function that returns figure data for three equally
    wonderful plots: wordcloud, frequency histogram and treemap"""

    if len(text) < 1:
        return {}, {}, {}

    word_cloud = WordCloud(max_words=100, max_font_size=90)
    word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 450],
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        }
    )

    wordcloud_figure_data = {"data": [trace], "layout": layout}
    word_list_top = word_list[:25]
    word_list_top.reverse()
    freq_list_top = freq_list[:25]
    freq_list_top.reverse()

    frequency_figure_data = {
        "data": [
            {
                "y": word_list_top,
                "x": freq_list_top,
                "type": "bar",
                "name": "",
                "orientation": "h",
            }
        ],
        "layout": {"height": "550", "margin": dict(t=20, b=20, l=100, r=20, pad=4)},
    }
    treemap_trace = go.Treemap(
        labels=word_list_top, parents=[""] * len(word_list_top), values=freq_list_top
    )
    treemap_layout = go.Layout({"margin": dict(t=10, b=10, l=5, r=5, pad=4)})
    treemap_figure = {"data": [treemap_trace], "layout": treemap_layout}
    return wordcloud_figure_data, frequency_figure_data, treemap_figure


server.mount('/', WSGIMiddleware(app.server))