import panel as pn

from .dashboard import Dashboard

def createApp():
    db = Dashboard()
    return pn.Row(db.param, db.plot).servable()