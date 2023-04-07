import plotly.graph_objects as go
import base64

class SpiderChart:
    def __init__(self, categories, values):
        self.categories = [*categories, categories[0]]
        self.values = [*values, values[0]]

    """
    Creates Spiderchart of neo-ffi-results
    :return str 
        (base64 encoded representation of spiderchart image)
    """
    def plot(self):
        fig = go.Figure(
            data=[
                go.Scatterpolar(r=self.values, theta=self.categories, marker={'size': 25}, fill='toself', name='Ergebnis in %'),
            ],
            layout=go.Layout(
                polar={'radialaxis': {'visible': True, 'range': [0,100], 'gridcolor': "#1049a3", "gridwidth": 2},
                       'angularaxis': {'gridcolor': "#1049a3", "gridwidth": 2, }},
                showlegend=True,
                font=dict(
                    size=34,
                ),
            )
        )
        png_base64 = base64.b64encode(fig.to_image(width=2000, height=1230)).decode('ascii')

        return png_base64
