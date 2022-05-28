import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from PIL import Image
from pass1 import get_prediction
def display_cells_to_get_data():
	st.title("Early diabetes prediction")
	with st.form("diabetes_symptoms"):
		flag = True
		age = st.text_input("Enter your age here (as an integer).",help="Pease enter an integer")
		age = age.strip()
		if len(age) > 0:
			if not age.isnumeric():
				flag = False
				st.error("Age is not a number")
			else:
				age = int(age)
		gender = st.selectbox('Please select your gender?',('Male', 'Female'))
		polyuria = st.selectbox('Do you urinate excessively?',('Yes', 'No'),help="Whether the patient experienced excessive urination or not.")
		polydipsia = st.selectbox('Do you feel abnormally thirsty?',('Yes', 'No'),help="Whether the patient experienced excessive thirst/excess drinking or not.")
		sudden_weight_loss = st.selectbox('Have you experienced sudden weight loss recently?',('Yes', 'No'),help="Whether patient had an episode of sudden weight loss or not.")
		weakness = st.selectbox('Do you feeling weak or fatigued often?',('Yes', 'No'))
		polyphagia = st.selectbox('Do you eat excessively?',('Yes', 'No'),help="Whether patient had an episode of excessive/extreme hunger or not.")
		genital_thrush = st.selectbox('Do you have any infections, especially near your genitals and/or mouth?',('Yes', 'No'),help="Whether patient had a yeast infection or not.")
		visual_blurring = st.selectbox('Do you feel like your vision has blurred recently?',('Yes', 'No'),help="Whether patient had an episode of blurred vision.")
		itching = st.selectbox('Do you face localized and severe itching anywhere on your body?',('Yes', 'No'))
		irritability = st.selectbox('Do you feel like you have low mood and are irritable?',('Yes', 'No'))
		delayed_healing = st.selectbox('Do you feel that your recent wounds have healed slowly or not healed at all?',('Yes', 'No'))
		partial_paresis = st.selectbox('Have you recently experienced weakening of muscles or a group of muscles?',('Yes', 'No'),help="Whether patient had partial or mild paralysis")
		muscle_stiffness = st.selectbox('Have you recently experienced cramps, join pains or painful walking?',('Yes', 'No'))
		alopecia = st.selectbox('Have you recently experienced any hair loss?',('Yes', 'No'),help="Whether patient had an episode of hair loss.")
		obesity = st.selectbox('Are you obese?',('Yes', 'No'))
		feats = ['age', 'alopecia', 'gender', 'genital_thrush', 'irritability', 'itching', 'obesity', 'polydipsia', 'polyphagia', 'polyuria', 'delayed_healing', 'muscle_stiffness', 'partial_paresis', 'sudden_weight_loss', 'visual_blurring', 'weakness']
		data = []
		for feat in feats:
			if eval(feat)=="Yes" or eval(feat) =="Male":
				feat = 1
			elif eval(feat) =="No" or eval(feat)=="Female":
				feat = 0
			exec(f"data.append({feat})")
		if st.form_submit_button('Click here to get your diagnosis'):
			if not flag or len(str(age))==0:
				st.write("Please check your inputs again.")
			else:
				x = np.array(data)
				result = get_prediction(x)
				st.success('Your diagnosis is {}'.format(result))
				if result=="Positive":
					prevention_im = Image.open("diabetes_prevention.jpg")
					st.image(prevention_im)