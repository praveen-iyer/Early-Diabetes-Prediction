import time,requests, json, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.naive_bayes import CategoricalNB
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt

# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader

start = time.time()
rs = 69 # Random state number to be used at all places

firebase_url = 'https://dsci551-c8eb7-default-rtdb.firebaseio.com/.json'
r = requests.get(firebase_url)
json_ob = json.loads(r.text)
df = pd.DataFrame.from_dict(json_ob)
df.replace(["Yes","Positive","Male"],1,inplace=True)
df.replace(["No","Negative","Female"],0,inplace=True)
# print(df)
# print(df.isna().sum())
# print(df.describe())
# print(df['class'].value_counts())

all_x_columns = df.columns.values.tolist()
all_x_columns.remove("class")
x = df.loc[:,all_x_columns]
y = df.loc[:,"class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=rs, stratify=y)

def get_metrics_for_model(model):
	model.fit(x_train,y_train)
	train_pred = model.predict(x_train)
	test_pred = model.predict(x_test)
	train_f1 = f1_score(y_train, train_pred, average='binary')
	test_f1 = f1_score(y_test, test_pred, average='binary')
	train_acc = accuracy_score(y_train, train_pred)
	test_acc = accuracy_score(y_test, test_pred)
	return train_f1,test_f1,train_acc,test_acc

model_rf = RandomForestClassifier(n_estimators=120, max_depth=3, oob_score=False, n_jobs=-1, random_state=rs)
train_f1_rf,test_f1_rf,train_acc_rf,test_acc_rf = get_metrics_for_model(model_rf)
# print(f"\ntrain_f1_rf {train_f1_rf} , test_f1_rf {test_f1_rf}")
# print(f"train_acc_rf {train_acc_rf} , test_acc_rf {test_acc_rf}")

# model_svm = SVC(kernel='rbf', random_state=rs)
# train_f1_svm,test_f1_svm,train_acc_svm,test_acc_svm = get_metrics_for_model(model_svm)
# print(f"\ntrain_f1_svm {train_f1_svm} , test_f1_svm {test_f1_svm}")
# print(f"train_acc_svm {train_acc_svm} , test_acc_svm {test_acc_svm}")

# model_lr = LogisticRegression(random_state=rs)
# train_f1_lr,test_f1_lr,train_acc_lr,test_acc_lr = get_metrics_for_model(model_lr)
# print(f"\ntrain_f1_lr {train_f1_lr} , test_f1_lr {test_f1_lr}")
# print(f"train_acc_lr {train_acc_lr} , test_acc_lr {test_acc_lr}")

# model_nb = CategoricalNB()
# train_f1_nb,test_f1_nb,train_acc_nb,test_acc_nb = get_metrics_for_model(model_nb)
# print(f"\ntrain_f1_nb {train_f1_nb} , test_f1_nb {test_f1_nb}")
# print(f"train_acc_nb {train_acc_nb} , test_acc_nb {test_acc_nb}")

# torch.manual_seed(rs)
# np.random.seed(rs)
# random.seed(0)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"\nDevice being used is {device}\n")

# class custom_data_loader(Dataset):
# 	def __init__(self,df_x,df_y):
# 		super(custom_data_loader,self).__init__()
# 		self.df_x = df_x.values
# 		self.df_y = df_y.values
	
# 	def __len__(self):
# 		return len(self.df_x)
	
# 	def __getitem__(self,idx):
# 		x = self.df_x[idx]
# 		y = self.df_y[idx]
# 		return x,y

# train_data = custom_data_loader(x_train,y_train)
# test_data = custom_data_loader(x_test,y_test)

# bs = 32
# train_data_batches = DataLoader(train_data,batch_size=bs,shuffle=True)
# test_data_batches = DataLoader(test_data,batch_size=bs,shuffle=False)

# class my_network(nn.Module):
# 	def __init__(self,input_size,dropout_required=True):
# 		super(my_network,self).__init__()
# 		self.input_size = input_size
# 		self.dropout_required = dropout_required
# 		self.l1 = nn.Linear(input_size,128)
# 		self.a = nn.ReLU()
# 		self.l2 = nn.Linear(128,32)
# 		self.l3 = nn.Linear(32,1)
# 		self.a_final = nn.Sigmoid()
# 		if dropout_required:
# 			self.d = nn.Dropout(p=0.3)

# 	def forward(self,x):
# 		out = self.l1(x)
# 		out = self.a(out)
# 		if self.dropout_required:
# 			out = self.d(out)
# 		out = self.l2(out)
# 		out = self.a(out)
# 		if self.dropout_required:
# 			out = self.d(out)
# 		out = self.l3(out)
# 		out = self.a_final(out)
# 		return out

# model_nn = my_network(16,True).to(device)
# # print(model_nn)

# loss_function = nn.BCELoss()
# optim = torch.optim.Adam(model_nn.parameters(), lr=5e-3)

# tr_f1s,tr_accs,tr_loss = [],[],[]
# te_f1s,te_accs,te_loss = [],[],[]
# def train_net(data_batches, loss_function, optim):
# 	n_samples = len(data_batches.dataset)
# 	model_nn.train()
# 	for batch,(x,y) in enumerate(data_batches):
# 		x,y = x.to(device).float(),y.to(device).float()
# 		pred = model_nn(x)
# 		pred = pred.squeeze()
# 		loss = loss_function(pred,y)

# 		optim.zero_grad()
# 		loss.backward()
# 		optim.step()

# 		if batch%5==0:
# 			loss,done = loss.item(),batch*len(x)
# 			print(f"Loss={loss} for progress {done}/{n_samples}")

# def test_net(data_batches, loss_function,train=False):
# 	n_samples = len(data_batches.dataset)
# 	n_batches = len(data_batches)
# 	model_nn.eval()
# 	test_loss,correct = 0,0
# 	tp,tn,fp,fn = 0,0,0,0
# 	with torch.no_grad():
# 		for x,y in data_batches:
# 			x,y = x.to(device).float(),y.to(device).float()
# 			pred = model_nn(x)
# 			pred = pred.squeeze()
# 			test_loss += loss_function(pred,y)
# 			pred = (pred>=0.5).int()
# 			pred,y = pred.cpu().detach().numpy(),y.cpu().detach().numpy()
# 			cm = confusion_matrix(y, pred)
# 			tn += cm[0][0]
# 			fn += cm[1][0]
# 			tp += cm[1][1]
# 			fp += cm[0][1]
# 			correct += ((pred==y).sum()).item()
# 	test_loss = test_loss / n_batches
# 	accuracy = correct / n_samples
# 	precision = tp/(tp+fp)
# 	recall = tp/(tp+fn)
# 	f1 = (2*precision*recall)/(precision+recall)
# 	if not train:
# 		te_f1s.append(f1)
# 		te_accs.append(accuracy)
# 		te_loss.append(test_loss)
# 		print(f"Test error is {test_loss} ; Accuracy is {accuracy} ; F1 score is {f1}")
# 	else:
# 		tr_f1s.append(f1)
# 		tr_accs.append(accuracy)
# 		tr_loss.append(test_loss)
# 		print(f"Train error is {test_loss} ; Accuracy is {accuracy} ; F1 score is {f1}")

# n_epochs = 20
# for i in range(n_epochs):
# 	print(f"Starting epoch {i+1}")
# 	print("-----"*10)
# 	train_net(train_data_batches, loss_function, optim)
# 	test_net(train_data_batches, loss_function,train=True)
# 	test_net(test_data_batches, loss_function)
# 	print("-----"*10+"\n\n")
# print("All epochs have been completed!")

end = time.time()

def get_feature_importances():
	importances = model_rf.feature_importances_
	std = np.std([tree.feature_importances_ for tree in model_rf.estimators_], axis=0)
	return importances,std,all_x_columns

def get_prediction(x):
	# y = model_nn(torch.from_numpy(x).float())
	y = model_rf.predict(x.reshape(1,-1))
	# if y.item()>=0.5:
	if y>=0.5:
		return "Positive"
	else:
		return "Negative"

# print(f"\n\nTotal duration is {end-start}")

# print(tr_f1s)
# print(tr_accs)
# print(tr_loss)
# print(te_f1s)
# print(te_accs)
# print(tr_loss)