import streamlit as st
from pass1 import get_feature_importances
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def get_early_diabetes_page():
	with open("early_diabetes_writeup.txt") as f:
		text = f.read().split("\n")
	st.write(text[0])
	early_image = Image.open('early_diabetes.jpg')
	st.image(early_image)
	prevention_image = Image.open("diabetes_symptoms.jpg")
	st.image(prevention_image, caption='Symptoms of Early diabtetes')

	st.write(text[1])

	corr_im = Image.open("Correlation_Plot_Diabetes.PNG")
	st.image(corr_im)

	importances,std,feats = get_feature_importances()
	forest_importances = pd.Series(importances, index=feats)
	forest_importances.sort_values(ascending=False,inplace=True)
	fig, ax = plt.subplots()
	# forest_importances.plot.bar(yerr=std, ax=ax)
	forest_importances.plot.bar(ax=ax)
	ax.set_title("Feature importances")
	ax.set_ylabel("Mean decrease in impurity according to the datast we used")
	fig.tight_layout()
	st.pyplot(fig)
	st.write(text[2])
	for i in range(8):
		st.write(text[3+i])
	st.write(text[-1])