from random import random
import struct as st

import matplotlib.pyplot as plt

def norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [(e-_min)/(_max-_min) for e in arr]

def ema(arr, K):
	e = [arr[0]]
	for p in arr[1:]: e += [e[-1]*(1-1/(1+K)) + p*1/(1+K)]
	return e

def source(fichier="prixs/BTCUSDT_high.bin"):
	with open(fichier, "rb") as co:
		text = co.read()
		(L,) = st.unpack('I', text[:4])
		return st.unpack('f'*L, text[4:])

def montrer(prixs, i,j):
	EMA    = i
	INTERV = int(i*j)
	_ema = ema(prixs, i)
	#
	plt.plot(prixs, label="prixs")
	plt.plot( _ema, label=" ema ")
	#
	s=0; r = [s:=(s+2*random()-1) for _ in range(8)]
	depart = len(prixs) - 8 * INTERV
	#
	_min = min([_ema[-1], _ema[depart]])
	_max = max([_ema[-1], _ema[depart]])
	#
	plt.plot(
		[depart+i*INTERV for i in range(8)],
		[(r[i]-min(r))/(max(r)-min(r)) * (_max-_min) + _min for i in range(8)],
		label="f"
	)
	#
	plt.show()

montrer(source(), i=64, j=1)