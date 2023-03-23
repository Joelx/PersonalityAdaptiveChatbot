import panel as pn
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def my_panel_app():
    pn.extension()
    text_input = pn.widgets.TextInput(name='Type something:')
    output = pn.pane.Markdown('')
    
    @pn.depends(text_input.param.value)
    def update_output(value):
        return f'You typed: **{value}**'
    
    app_layout = pn.Column(
        text_input,
        output
    )
    
    return app_layout

# Serve your Panel app via a FastAPI route
@app.get("/")
async def root():
    return my_panel_app()
    #return pn.pane.HTML(my_panel_app().html, sizing_mode="stretch_width")

# Start the FastAPI server
#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8007)