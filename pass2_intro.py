import streamlit as st
from PIL import Image

def intro_content():
	with open("diabetes_intro_writeup.txt") as f:
		text = f.read().split("\n")
	st.write(text[0])
	image = Image.open('diabetes_stats.png')
	st.image(image, caption='Statistics on diabetes')