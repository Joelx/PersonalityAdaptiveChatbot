import asyncio
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from RabbitMQ import *
import json
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import html
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import numpy as np
from wordcloud import WordCloud
import evaluations
import os
import time
import urllib.parse

EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css", dbc.themes.BOOTSTRAP]

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
                    #dbc.Spinner(
                    #),
                    html.Div(
                        id="eval-conversation-html",
                        children=[
                                html.Div("No evaluation run has been started yet.")
                                ],
                        style={"width": "100%", "height": "200px", "overflow-y": "scroll", "margin-top": "8px"}
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
    dbc.CardHeader(html.H4(children="Embeddings TSNE Plot", className="lead")),
    dbc.CardBody(
        dcc.Graph(id='scatter-plot')
    )
]

SENTIMENT_OUTPUT = [
    dbc.CardHeader(html.H4(children="Sentiments", className="lead")),
    dbc.CardBody(
    html.Div(children=[
        html.P(id='sentiment-output-row', children='Calulated no sentiments yet', 
            style={"fontSize": 16, "margin-left": 6, "margin-top": 6},),
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
               dbc.Table([
            html.Tr([
                html.Td(html.Label('Neuroticism: ', style={'margin-right': '10px'})),
                html.Td(dcc.Input(
                    id='neuroticism-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.578,
                )),
            ]),
            html.Tr([
                html.Td(html.Label('Extraversion: ', style={'margin-right': '10px'})),
                html.Td(dcc.Input(
                    id='extraversion-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.478,
                )),
            ]),
            html.Tr([
                html.Td(html.Label('Openness: ', style={'margin-right': '10px'})),
                html.Td(dcc.Input(
                    id='openness-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.178,
                )),
            ]),
            html.Tr([
                html.Td(html.Label('Agreeableness: ', style={'margin-right': '10px'})),
                html.Td(dcc.Input(
                    id='agreeableness-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.494,
                )),
            ]),
            html.Tr([
                html.Td(html.Label('Conscientiousness: ', style={'margin-right': '10px'})),
                html.Td(dcc.Input(
                    id='conscientiousness-threshold',
                    type='number',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.299,
                )),
            ]),
        ], bordered=True)], md=6),
            dbc.Col([
                html.H6("Double-check actual config fetched by pipeline server"),
                html.Div(id='threshold-output-row'),
            ], md=6),
        ]),
    ]),
])

CLASSIFICATION_RESULT = [
    dbc.CardHeader(html.H4(children="Big5 Classification Results", className="lead")),
    dbc.CardBody(
    html.Div(className="row", children=[
    
        html.Div(id="cf-output-row", className="col-md-4"),
        html.Div(id="eval-cf-output-row", className="col-md-4"),
        html.Div(id="test-output-row", className="col-md-4"),
        #html.Div(id='cf-output-row', children='Calulated no classifications yet', 
       # style={"fontSize": 14, "margin-left": 6, "margin-top": 6},),
            ]))
        
]

WORDCLOUD_PLOT = [
    dbc.CardHeader(html.H4(children="Most frequently used words in conversation", className="lead")),
    dbc.Alert(
        "Not enough data to render these plots.",
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

HEADER = dbc.Container([
        dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
        dcc.Interval(id='interval-component-15', interval=10000, n_intervals=0),
        html.Div([dcc.Location(id='url', refresh=False)]),
        dcc.Store(id='wordcloud-data-store'),
        dcc.Store(id='current-user-text'),
       # dcc.Store(id="classification-data-store"),
       # dcc.Store(id="test-results-data-store"),
])

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(CLASSIFICATION_THRESHOLDS)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(CLASSIFICATION_RESULT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(CURRENT_PROMPT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TSNE_PLOT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(SENTIMENT_OUTPUT)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(FEATURE_OUTPUT)),], style={"marginTop": 30}),
        dbc.Card(WORDCLOUD_PLOT, style={"marginTop": 30}),
        dbc.Row([dbc.Col(EVALUATION_STARTER),], style={"marginTop": 30, "marginBottom": 30}),
        dbc.Row([dbc.Col(html.Div(id="test-content", children=[
            html.Div(children=[
                    html.Strong("Sender_ID: "), html.Span(id="session-id-field", children=[])
                ])
        ]))], style={"marginBottom": 30})
    ],
    className="mt-12",
)

server = FastAPI()
server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#server.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(16))
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/dashboard/')
app.layout = html.Div(children=[HEADER, BODY])


@app.callback(
    Output("eval-button", "disabled"),
    Output("eval-alert", "style"),
    Output('eval-conversation-html', 'children', allow_duplicate=True),
    Input("eval-button", "n_clicks"),
    prevent_initial_call=True
)
def eval_button_callback(n_clicks):
    print("Eval button clicked")
    if not n_clicks:
        return [False, {"display": "none"}, ""]
    
    time.sleep(2) # wait for 2 seconds
    return asyncio.run(start_eval_run(n_clicks)) # execute start_eval_run coroutine after 2 seconds

async def start_eval_run(n_clicks):
    if not n_clicks:
        return [False, {"display": "none"}, ""]

    bot = evaluations.RasaChatClient("https://joel-schlotthauer.com", socketio_path="/rasax/socket.io/")
    bot.connect()
    bot.utter("/start_conversation")

    return [False, {"display": "none"}, ""]


@app.callback(
    Output('eval-conversation-html', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('session-id-field', 'children')],
    [State('eval-conversation-html', 'children')],
)
def update_eval_conversation(n, session_id, current_children):
    children = [] if not current_children else current_children
    body = receive_rabbitmq(queue="eval_bot_message", sender_id=session_id)
    if body:
        message = json.loads(body)
        for key, msg in message.items():
            print(f"RECEIVED {key}: {msg}")
            if key == "human_bot_uttered":
                key_text = "Human AI"
            elif key == "ai_bot_uttered":
                key_text = "Bot AI"
            else:
                key_text = key
            line = html.Div(children=[
                html.B(key_text + ": "),
                html.Span(msg),
            ])
            children.append(line)
        return children
    else:
        raise PreventUpdate


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
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals'), 
     dash.dependencies.Input('session-id-field', 'children')],
    [dash.dependencies.State('scatter-plot', 'figure')]
)
def update_tsne_plot(n, sender_id, previous_tsne_fig):
    if not previous_tsne_fig:
        previous_tsne_fig = {}
    body = receive_rabbitmq(queue="embeddings", sender_id=sender_id)
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
        fig.update_layout()
        # store the current data in the previous_tsne_fig variable
        #previous_tsne_fig = fig.to_dict()

    else:
        # If there is no new data, use the previous_tsne_fig to update the plot
        if previous_tsne_fig:
            # Create a scatter plot using Plotly Express
            fig = previous_tsne_fig
            #fig = go.Figure.from_dict(previous_tsne_fig)
        else: 
            fig = go.Figure()

    return fig


@app.callback(
    Output('sentiment-output-row', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('session-id-field', 'children')],
    [State('sentiment-output-row', 'children')]
)
def update_sentiment(n, sender_id, previous_sentiment):
    body = receive_rabbitmq(queue="sentiment", sender_id=sender_id)
    # If a message was received, decode and return the message  
    if body:
        data = json.loads(body)
        sentiment = data['sentiment']
        sentiment_string = f"""Sentiments in Percent:
        Negative: {round(sentiment[0]*100,2)} % |
        Neutral: {round(sentiment[1]*100,2)} % |
        Positive: {round(sentiment[2]*100,2)} %
        """
        # Update the State variable with the new sentiment value
        previous_sentiment = sentiment_string
    else:
        # If there is no new data, use the previous sentiment value
        if previous_sentiment:
            sentiment_string = previous_sentiment
        else: 
            sentiment_string = "Calculated no sentiment yet"
    return sentiment_string

@app.callback(
    Output('features-output-row', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('session-id-field', 'children')],
    [State('features-output-row', 'children')]
)
def update_features(n, sender_id, previous_features):
    body = receive_rabbitmq(queue="features", sender_id=sender_id)
    # If a message was received, decode and return the message  
    if body:
        data = json.loads(body)
        features = data['features']
        features_html = []
        for feature in features:
            features_html.append(html.Strong(children=feature.title(),
                                         style={"fontSize": 14, "fontWeight": "bold", "margin-left": 6, "margin-top": 6}, className="lead"),)
            features_html.append(html.P(children="Num of Features: " + str(features[feature]["num_of_features"]),
                                        style={"fontSize": 12, "margin-left": 6, "margin-top": 6},),)
            features_html.append(html.P(children="Features: " + ', '.join(features[feature]["feature_names"]),
                                        style={"fontSize": 12, "margin-left": 6, "margin-top": 6},),)
        previous_features = features_html
    else:
        # If there is no new data, use the previous_tsne_fig to update the plot
        if previous_features:
            # Create a scatter plot using Plotly Express
            features_html = previous_features
        else: 
            features_html = [html.H4(children="No features calculated yet", 
                                     style={"font-weight": "normal", "fontSize": 14, "margin-left": 6, "margin-top": 6},),]
    return features_html


# Update current_user_text
@app.callback(
    Output('current-user-text', 'data'),
    [Input('interval-component', 'n_intervals'),
     Input('session-id-field', 'children')],
    State('current-user-text', 'data')
)
def update_current_user_text(n, sender_id, current_user_text):
    # Connect to RabbitMQ queue
    body = receive_rabbitmq(queue="text", sender_id=sender_id)

    if body:
        text = json.loads(body)['conversation_history']
        return text
    else:
        raise PreventUpdate


"""
WORDCLOUD
"""

# Store wordcloud-related data
@app.callback(
    Output("wordcloud-data-store", "data"),
    [Input('interval-component-15', 'n_intervals'),
     Input('session-id-field', 'children')],
    State("wordcloud-data-store", "data"),
    State('current-user-text', 'data')
)
def store_wordcloud_data(n, sender_id, stored_data, current_user_text):
    # Connect to RabbitMQ queue
    if current_user_text:
        wordcloud, frequency_figure, treemap = plotly_wordcloud(current_user_text)
        alert_style = {"display": "none"}
        if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}):
            alert_style = {"display": "block"}

        return {
            "previous_wordcloud": wordcloud,
            "previous_fq_fig": frequency_figure,
            "previous_tm": treemap,
            "previous_alert": alert_style,
            "current_user_text": current_user_text
        }
    else:
        raise PreventUpdate

# Update wordcloud callback
@app.callback(
    [
        Output("bank-wordcloud", "figure"),
        Output("frequency_figure", "figure"),
        Output("bank-treemap", "figure"),
        Output("no-data-alert", "style"),
    ],
    [Input('interval-component-15', 'n_intervals')],
    State('session-id-field', 'children'),
    State("wordcloud-data-store", "data"),
)
def update_wordcloud(n, sender_id, stored_data):
    if not stored_data:
        raise PreventUpdate

    wordcloud = stored_data["previous_wordcloud"]
    frequency_figure = stored_data["previous_fq_fig"]
    treemap = stored_data["previous_tm"]
    alert_style = stored_data["previous_alert"]

    return (wordcloud, frequency_figure, treemap, alert_style)


@app.callback(
    Output('threshold-output-row', 'children'),
    [Input('openness-threshold', 'value'),
     Input('conscientiousness-threshold', 'value'),
     Input('extraversion-threshold', 'value'),
     Input('agreeableness-threshold', 'value'),
     Input('neuroticism-threshold', 'value'),
     Input('session-id-field', 'children')],
    State("threshold-output-row", "children"),
)
def update_thresholds(openness_threshold, 
                  conscientiousness_threshold, 
                  extraversion_threshold, 
                  agreeableness_threshold, 
                  neuroticism_threshold, 
                  sender_id,
                  previous_actual_threshold_data):
    threshold_json =  {
        "neuroticism": neuroticism_threshold,
        "extraversion": extraversion_threshold,
        "openness": openness_threshold,
        "agreeableness": agreeableness_threshold,
        "conscientiousness": conscientiousness_threshold
    }
    send_to_rabbitmq(json.dumps(threshold_json), queue="thresholds", sender_id=sender_id)

    body = receive_rabbitmq(queue="actual-thresholds", sender_id=sender_id)
    html_data = []
    if body:
        actual_thresholds = json.loads(body)
        neuroticism_threshold = actual_thresholds["neuroticism"]
        extraversion_threshold = actual_thresholds["extraversion"]
        openness_threshold = actual_thresholds["openness"]
        agreeableness_threshold = actual_thresholds["agreeableness"]
        conscientiousness_threshold = actual_thresholds["conscientiousness"]

        html_data = html.Div([
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
        # If there is no new data, use the previous_tsne_fig to update the plot
        if previous_actual_threshold_data:
            # Create a scatter plot using Plotly Express
            html_data = previous_actual_threshold_data
        else: 
            html_data = html.Div(html.Span("Fetched no real configuration from pipeline server yet."))
    return html_data



def get_classification_html(classification_data, current_user_text):
    if classification_data:
        classes = classification_data['classes']
        probas = classification_data['probabilities']
        classification_html = [html.H6(children="Pipeline Classification:", style={"fontSize": 16})]
        for cs in classes:
            classification_html.extend([
                html.Strong(children=cs.title(), style={"display": "inline", "fontSize": 14, "fontWeight": "bold"}, className="lead"),
                html.P(children=f"CF{classes[cs]}, Proba: [{round(float(probas[cs][0]), 2)}]", style={"fontSize": 12, "font-weight": "lighter"})
            ])
        return html.Div(classification_html)
    else:
        return html.Div("No pipeline classification data")


def get_test_results_html(test_results_data):
    if test_results_data:
        test_results_html = [html.H6(children="Result of actual NEO-FFI-Test:", style={"fontSize": 16})]
        for key, data in test_results_data.items():
            test_results_html.extend([
                html.Strong(children=key.title(), style={"display": "inline", "fontSize": 14, "fontWeight": "bold"}, className="lead"),
                html.P(children=f"{data}", style={"fontSize": 12, "font-weight": "lighter"})
            ])
        return html.Div(test_results_html)
    else:
        return html.Div("No test results data")


def get_eval_classification(current_user_text):
    return html.Div("No evaluation classification available") # Currently out of service
    if current_user_text:
        eval_classification_html = [html.H6(children="Evaluation Classification by gpt-3.5-turbo:", style={"fontSize": 16})]
        eval_classifier = evaluations.BigFiveClassificationEvaluator()
        eval_cf_res = eval_classifier.classify(current_user_text)
        cf_lines = eval_cf_res.split('\n')
        for line in cf_lines:
            eval_classification_html.append(html.P(children=line, style={"fontSize": 14, "font-weight": "lighter"}))
            return html.Div(eval_classification_html)
        else:
            return html.Div("No evaluation classification available")


@app.callback(
    Output('cf-output-row', 'children'),
    Output('eval-cf-output-row', 'children'),
    Output('test-output-row', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('session-id-field', 'children')],
    State('cf-output-row', 'children'),
    State('eval-cf-output-row', 'children'),
    State('test-output-row', 'children'),
    State('current-user-text', 'data')
)
def update_classification(n, sender_id, previous_classification, prev_eval_cf, prev_test_results, current_user_text):
    classification_body = receive_rabbitmq(queue="classification", sender_id=sender_id)
    test_results_body = receive_rabbitmq(queue="big_five_test_results", sender_id=sender_id)

    return_values = [
        "No classification data available", # CF Output row
        "No evaluation classification data available", # Evaluation Classification row
        "No test results available" # Test result row of NEO-FFI-30-Test
    ]

    if classification_body: 
        classification_data = json.loads(classification_body) if classification_body else None   
        if classification_data:
            classification_html = get_classification_html(classification_data, current_user_text)
            return_values[0] = classification_html
    else:
        if previous_classification:
            return_values[0] = previous_classification

    if test_results_body:
        test_results_data = json.loads(test_results_body) if test_results_body else None     
        if test_results_data:
            test_results_html = get_test_results_html(test_results_data)
            return_values[1] = test_results_html
    else: 
        if prev_eval_cf:
            return_values[1] = prev_eval_cf

    if current_user_text:
        current_user_text_html = get_eval_classification(current_user_text)
        return_values[2] = current_user_text_html
    else:
        if prev_test_results:
            return_values[2] = prev_test_results
        
    #print(return_values)
    
    return return_values



@app.callback(
    Output('prompt-output-row', 'children'),
    [Input('interval-component', 'n_intervals'),
    Input('session-id-field', 'children')],
    State('prompt-output-row', 'children')
)
def update_prompt(n, sender_id, previous_prompt):
    body = receive_rabbitmq(queue="prompt", sender_id=sender_id)
    # If a message was received, decode and return the message  
    if body:        
        data = json.loads(body)
        token_size = data['token_size']
        prompt_lines = data['prompt'].split('\n')
        prompt_html = [html.P(f"Current Token-Size: {token_size}")]
        for line in prompt_lines:
            prompt_html.append(html.Span(line))
            prompt_html.append(html.Br())
        prompt_html = html.Pre(prompt_html)

        #prompt_html = [html.P(line) for line in prompt_lines]
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



# @app.callback([Output('session-id-field', 'children'),
#               Output('session-id-storage', 'data')],
#              [Input("url", "search")],
#              [State('session-id-storage', 'data')])



# Get Rasa Session ID via GET
@app.callback(Output('session-id-field', 'children'),
              #Output('session-id-storage', 'data'),],
             [Input("url", "search")],
             State('session-id-field', 'children'))
def fetch_session_id(params, children):
    try:
        parsed = urllib.parse.urlparse(params)
        parsed_dict = urllib.parse.parse_qs(parsed.query)
        sender_id = parsed_dict['session_id'][0]
        print(f"DASH: {sender_id}")

        return sender_id
    except Exception as e:
        return "An error occurred fetching session id: {e}"

server.mount('/', WSGIMiddleware(app.server))