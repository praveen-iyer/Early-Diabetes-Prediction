from pass2_intro import intro_content
from pass2_model_form_and_prediction import display_cells_to_get_data
from pass2_early_diabetes import get_early_diabetes_page
import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
        choose=st.radio("",self.apps,format_func=lambda app: app['title'])
        choose['function']()

app = MultiApp()
app.add_app(" Introduction", intro_content)
app.add_app("Early Diabetes", get_early_diabetes_page)
app.add_app("Self diagnosis", display_cells_to_get_data)
# The main app
app.run()