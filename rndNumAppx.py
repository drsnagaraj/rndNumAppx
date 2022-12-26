from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

base=10
window=base
sz=window*100
base = int(base)
window = int(window)
sz=int(sz)

def genIntNormal(mean,dev=0.75,siz=sz):
	s = np.random.normal(mean, dev, siz)
	s = np.maximum(0,s)
	s = np.minimum(9,s)
	s= np.round(s)
	return np.resize(s,(1,siz))
	

def genIntUnif():
	s = np.random.uniform(0,10,sz)
	s = np.maximum(0,s)
	s = np.minimum(9,s)
	s=np.round(s)
	return np.resize(s,(1,sz))

def getDigitFrequencies(digits,frequencies):
	strout=''
	if len(digits)!=len(frequencies):
		print("error len(digits!=len(frequencies))")
		return
	arr = [str(n)*frequencies[n] for n in digits]
	strout = ''.join(arr)
	return np.array([int(k) for k in strout])
	
def genIntMultinomial(unif=True, mmode=False, mean=[3,8], siz=sz):
	if unif and mmode:
		print("error genIntMultinomial: cant be unif and mmode same time")
		print("returning ")
	digits = list(range(window))
	s = []
	k = sz//window
	if unif:
		vec = [1/float(window)]*window
		for kk in range(k):
			draw = np.random.multinomial(10,vec,size=1)
			s+=[getDigitFrequencies(digits,list(draw[0]))]		
		s = np.array(s)
		s = np.resize(s,(1,siz))
	elif not mmode:
		mu = mean[0]
		vec = [1/float(2.0*window)]*window
		vec[mu] += 0.25
		if mu==0:
			vec[mu+1] += 0.25/2
			vec[mu+2] += 0.25/2
		elif mu==window-1:
			vec[mu-1] += 0.25/2
			vec[mu-2] += 0.25/2	
		else:
			vec[mu-1] += 0.25/2
			vec[mu+1] += 0.25/2
		
		for kk in range(k):
			draw = np.random.multinomial(10,vec,size=1)
			s+=[getDigitFrequencies(digits,list(draw[0]))]		
		s = np.array(s)
		s = np.resize(s,(1,siz))
	
	elif mmode:			
		mu1 = mean[0]
		mu2 = mean[1]
		vec = [1/float(2.0*window)]*window
		vec[mu1] += 0.25
		vec[mu2] += 0.25							
									
		for kk in range(k):
			draw = np.random.multinomial(10,vec,size=1)
			s+=[getDigitFrequencies(digits,list(draw[0]))]		
		s = np.array(s)
		s = np.resize(s,(1,siz))
							
	return s

def genRational(periodic=False):
	
	if not periodic:
		
		s = genIntMultinomial(unif=True, mmode=False, mean=[], siz=sz//2)
		s = np.append(s,np.zeros(sz//2))	
		if len(s)!=sz:
			print("len of s != sz not periodic")
			print(len(s))
			print("returning")
			return
		return np.resize(s,(1,sz))
	else:
		s = genIntMultinomial(unif=True, mmode=False, mean=[], siz=window)
		if not sz//window == 100:
			print('error: sz/window!=100')
			print('returning')
			return
		
		s=np.tile(s,sz//window)
		if len(s[0])!=sz:
			print("len of s != sz periodic")
			print(len(s))
			print(len(s[0]))
			print("returning")
			return		
		return np.resize(s,(1,sz))

def genRationalData(sz_data=1000):
	X = []
	for i in range(sz_data//2):
		X += [genRational(periodic=False)]
		X += [genRational(periodic=True)]
	X=np.array(X)
	X=np.resize(X,(sz_data,sz))

	if X.shape[0]!=sz_data:
		print('genRationalData X.shape[0]!=sz_data')
		print("X.shape = ", X.shape)
		print('returning')
		return		
	return X


def genDistrData(unif=True, mmode=False, mean=[3,8], sz_data=1000):
	X = []	
	if unif:
		for i in range(sz_data):
			X += [genIntMultinomial(unif=True, mmode=False, mean=[], siz=sz)]
		
		X=np.array(X)	
		if X.shape[0]!=sz_data:
			print('genDistrData X.shape[0]!=sz_data')
			print("X.shape = ", X.shape)
			print('returning')
			return				
		X=np.resize(X,(sz_data,sz))		
	elif not mmode:
		sz_per_digit = sz_data//window
		singlemode = list(range(window))
		for i in range(sz_per_digit):
			for j in singlemode:
				X += [genIntMultinomial(unif=False, mmode=False, mean= [singlemode[j]], siz=sz)]	
		
		X=np.array(X)
		if X.shape[0]!=sz_data:
			print('genDistrData X.shape[0]!=sz_data')
			print("X.shape = ", X.shape)
			print('returning')
			return				
		X=np.resize(X,(sz_data,sz))		
	elif mmode:
		sz_per_digit = sz_data//len(mean)
		for i in range(sz_per_digit):
			for j in mean:
				X += [genIntMultinomial(unif=False, mmode=False, mean= [mean[j]], siz=sz)]	
		
		X=np.array(X)
		if X.shape[0]!=sz_data:
			print('genDistrData X.shape[0]!=sz_data')
			print("X.shape = ", X.shape)
			print('returning')
			return				
		X=np.resize(X,(sz_data,sz))			
				
		
	return X



def genDistrNormalData(sz_data=1000):
	X = []
	sz_per_digit = sz_data//window
	
	for i in range(sz_per_digit):
		for j in range(window):
			X += [genIntNormal(mean=j,dev=0.75)]
	X=np.array(X)
	X=np.resize(X,(sz_data,sz))

	if X.shape[0]!=sz_data:
		print('genDistrNormalData X.shape[0]!=sz_data')
		print("X.shape = ", X.shape)
		print('returning')
		return		
	return X

def genDistrMMNormalData(sz_data=1000):
	X = []
	
	i1 = 3
	i2 = 8
	
	for i in range(sz_data):
		x1=genIntNormal(mean=i1,dev=0.5,siz=sz//2)
		x2=genIntNormal(mean=i2,dev=0.5,siz=sz//2)
		#print("x2\n",x2)
		x3=np.concatenate((x1,x2),axis=1)
		x3=np.resize(x3,(1,sz))
		#print("len of x3",len(x3[0]),"\n",x3[0])
		#x1=x1
		#x1=np.round(x1)
		X += [x3]
	X=np.array(X)
	X=np.resize(X,(sz_data,sz))

	if X.shape[0]!=sz_data:
		print('genDistrNormalData X.shape[0]!=sz_data')
		print("X.shape = ", X.shape)
		print('returning')
		return		
	return X	
	
def genDistrUnifData(sz_data=1000):
	X = []
	
	for i in range(sz_data):
		X += [genIntUnif()]
	X=np.array(X)
	X=np.resize(X,(sz_data,sz))

	if X.shape[0]!=sz_data:
		print('XgenDistrUnifData .shape[0]!=sz_data')
		print("X.shape = ", X.shape)
		print('returning')
		return		
	return X


def getAutoencoder(data_size = 10000, mode='rational'):
#
	if mode=='rational':
		X_train = genRationalData(sz_data=int(data_size*0.8))
		X_test = genRationalData(sz_data=int(data_size*0.2))
	elif mode=='distr':
		X_train = genDistrNormalData(sz_data=int(data_size*0.8))
		X_test = genDistrNormalData(sz_data=int(data_size*0.2))
	elif mode=='mmdistr':
		X_train = genDistrMMNormalData(sz_data=int(data_size*0.8))
		X_test = genDistrMMNormalData(sz_data=int(data_size*0.2))		
	elif mode=='unif':
		X_train = genDistrUnifData(sz_data=int(data_size*0.8))
		X_test = genDistrUnifData(sz_data=int(data_size*0.2))		
		
	# number of input columns
	n_inputs = X_train.shape[1]
	print(n_inputs)
	# define encoder
	visible = Input(shape=(n_inputs,))
	# encoder level 1
	e = Dense(n_inputs*2)(visible)
	e = BatchNormalization()(e)
	e = LeakyReLU()(e)
	# encoder level 2
	e = Dense(n_inputs)(e)
	e = BatchNormalization()(e)
	e = LeakyReLU()(e)
	# encoder level 3
	e = Dense(n_inputs)(e)
	e = BatchNormalization()(e)
	e = LeakyReLU()(e)
	# bottleneck
	n_bottleneck = min(window*5,n_inputs//5)
	bottleneck = Dense(n_bottleneck)(e)
	# define decoder, level 1
	d = Dense(n_inputs)(bottleneck)
	d = BatchNormalization()(d)
	d = LeakyReLU()(d)
	# decoder level 2
	d = Dense(n_inputs)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU()(d)
	# decoder level 3
	d = Dense(n_inputs*2)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU()(d)
	# output layer
	output = Dense(n_inputs, activation='linear')(d)
	# define autoencoder model
	model = Model(inputs=visible, outputs=output)
	# compile autoencoder model
	model.compile(optimizer='adam', loss='mse')
	# plot the autoencoder
	plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
	# fit the autoencoder model to reconstruct input
	history = model.fit(X_train, X_train, epochs=300, batch_size=32, verbose=0, validation_data=(X_test,X_test))

	model.save(mode+'autoencoder.h5')

def driver():
	
	# ~ getAutoencoder(data_size = 10000, mode='rational')
	# ~ getAutoencoder(data_size = 10000, mode='distr')
	# ~ getAutoencoder(data_size = 10000, mode='unif')
	getAutoencoder(data_size = 10000, mode='mmdistr')
	
	# ~ rationalautoencoder = load_model('rationalautoencoder.h5')
	# ~ distrautoencoder = load_model('distrautoencoder.h5')
	# ~ unifautoencoder = load_model('unifautoencoder.h5')
	mmdistrautoencoder = load_model('mmdistrautoencoder.h5')
	
	X_input_rational = genRationalData(sz_data=10)
	X_pred_rational_rational = rationalautoencoder.predict(X_input_rational)
	X_pred_rational_distr = distrautoencoder.predict(X_input_rational)
	X_pred_rational_unif = unifautoencoder.predict(X_input_rational)
	
	X_input_distr = genDistrNormalData(sz_data=10)
	X_pred_distr_rational = rationalautoencoder.predict(X_input_distr)
	X_pred_distr_distr = distrautoencoder.predict(X_input_distr)
	X_pred_distr_unif = unifautoencoder.predict(X_input_distr)	
	
	
	X_input_unif = genDistrUnifData(sz_data=10)
	X_pred_unif_rational = rationalautoencoder.predict(X_input_unif)
	X_pred_unif_distr = distrautoencoder.predict(X_input_unif)
	X_pred_unif_unif = unifautoencoder.predict(X_input_unif)	


rationalautoencoder = load_model('rationalautoencoder.h5')
distrautoencoder = load_model('distrautoencoder.h5')
unifautoencoder = load_model('unifautoencoder.h5')
	
X_input_rational = genRationalData(sz_data=100)
X_pred_rational_rational = np.round(rationalautoencoder.predict(X_input_rational))
X_pred_rational_distr = np.round(distrautoencoder.predict(X_input_rational))
X_pred_rational_unif = np.round(unifautoencoder.predict(X_input_rational))

X_pred_rational_rational = np.maximum(0,X_pred_rational_rational)
X_pred_rational_rational = np.minimum(9,X_pred_rational_rational)	

X_pred_rational_distr = np.maximum(0,X_pred_rational_distr)
X_pred_rational_distr = np.minimum(9,X_pred_rational_distr)

X_pred_rational_unif = np.maximum(0,X_pred_rational_unif)
X_pred_rational_unif = np.minimum(9,X_pred_rational_unif)






	
X_input_distr = genDistrMMNormalData(sz_data=100)#genDistrData(unif=False, mmode=False, mean=[3,8], sz_data=100)#
X_pred_distr_rational = np.round(rationalautoencoder.predict(X_input_distr))
X_pred_distr_distr = np.round(distrautoencoder.predict(X_input_distr))
X_pred_distr_unif = np.round(unifautoencoder.predict(X_input_distr)	)

X_pred_distr_rational = np.maximum(0,X_pred_distr_rational)
X_pred_distr_rational = np.minimum(9,X_pred_distr_rational)	

X_pred_distr_distr = np.maximum(0,X_pred_distr_distr)
X_pred_distr_distr = np.minimum(9,X_pred_distr_distr)

X_pred_distr_unif = np.maximum(0,X_pred_distr_unif)
X_pred_distr_unif = np.minimum(9,X_pred_distr_unif)


	
	
X_input_unif = genDistrUnifData(sz_data=100)#genDistrData(unif=True, mmode=False, mean=[3,8], sz_data=1000)#
X_pred_unif_rational = np.round(rationalautoencoder.predict(X_input_unif))
X_pred_unif_distr = np.round(distrautoencoder.predict(X_input_unif))
X_pred_unif_unif = np.round(unifautoencoder.predict(X_input_unif))

X_pred_unif_rational = np.maximum(0,X_pred_unif_rational)
X_pred_unif_rational = np.minimum(9,X_pred_unif_rational)	

X_pred_unif_distr = np.maximum(0,X_pred_unif_distr)
X_pred_unif_distr = np.minimum(9,X_pred_unif_distr)

X_pred_unif_unif = np.maximum(0,X_pred_unif_unif)
X_pred_unif_unif = np.minimum(9,X_pred_unif_unif)

n= 29


rirp = X_input_rational[n]
rorp = X_pred_rational_rational[n]

dirp = X_input_distr[n]
dorp = X_pred_distr_rational[n]

uirp = X_input_unif[n]
uorp = X_pred_unif_rational[n]

_ = plt.hist(rirp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram rirp")
plt.show()


_ = plt.hist(rorp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram rorp")
plt.show()


_ = plt.hist(dirp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram dirp")
plt.show()


_ = plt.hist(dorp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram dorp")
plt.show()

_ = plt.hist(uirp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram uirp")
plt.show()

_ = plt.hist(uorp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram uorp")
plt.show()





ridp = X_input_rational[n]
rodp = X_pred_rational_distr[n]

didp = X_input_distr[n]
dodp = X_pred_distr_distr[n]

uidp = X_input_unif[n]
uodp = X_pred_unif_distr[n]

_ = plt.hist(ridp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram ridp")
plt.show()


_ = plt.hist(rodp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram rodp")
plt.show()


_ = plt.hist(didp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram didp")
plt.show()


_ = plt.hist(dodp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram dodp")
plt.show()

_ = plt.hist(uidp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram uidp")
plt.show()

_ = plt.hist(uodp, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram uodp")
plt.show()








riup = X_input_rational[n]
roup = X_pred_rational_unif[n]

diup = X_input_distr[n]
doup = X_pred_distr_unif[n]

uiup = X_input_unif[n]
uoup = X_pred_unif_unif[n]

_ = plt.hist(riup, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram riup")
plt.show()


_ = plt.hist(roup, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram roup")
plt.show()


_ = plt.hist(diup, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram diup")
plt.show()


_ = plt.hist(doup, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram doup")
plt.show()

_ = plt.hist(uiup, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram uiup")
plt.show()

_ = plt.hist(uoup, bins=range(0,10))  # arguments are passed to np.histogram
plt.title("Histogram uoup")
plt.show()

def run():

	X_input_rational = genRationalData(sz_data=10)
	X_pred_rational_rational = rationalautoencoder.predict(X_input_rational)
	X_pred_rational_distr = distrautoencoder.predict(X_input_rational)
	X_pred_rational_unif = unifautoencoder.predict(X_input_rational)
	
	X_input_distr = genDistrNormalData(sz_data=10)
	X_pred_distr_rational = rationalautoencoder.predict(X_input_distr)
	X_pred_distr_distr = distrautoencoder.predict(X_input_distr)
	X_pred_distr_unif = unifautoencoder.predict(X_input_distr)	
	
	
	X_input_unif = genDistrUnifData(sz_data=10)
	X_pred_unif_rational = rationalautoencoder.predict(X_input_unif)
	X_pred_unif_distr = distrautoencoder.predict(X_input_unif)
	X_pred_unif_unif = unifautoencoder.predict(X_input_unif)	











