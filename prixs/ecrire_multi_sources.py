#! /usr/bin/python3

from random import random
import struct as st

rnd = lambda : 2*random()-1

ema = lambda l,K,ema=0: [ema:=(ema*(1-1/K) + e/K) for e in l]

prixs      = lambda infos:     [p                  for p,vb,vu,l,h,t in infos]
_low       = lambda infos:     [l                  for p,vb,vu,l,h,t in infos]
_high      = lambda infos:     [h                  for p,vb,vu,l,h,t in infos]
median     = lambda infos:     [(h+l)/2            for p,vb,vu,l,h,t in infos]
volumes    = lambda infos,s=0: [s:=(s + (vb*p-vu)) for p,vb,vu,l,h,t in infos]
volumes_A  = lambda infos:     [vb                 for p,vb,vu,l,h,t in infos]
volumes_U  = lambda infos:     [vu                 for p,vb,vu,l,h,t in infos]
tradecount = lambda infos:     [t                  for p,vb,vu,l,h,t in infos]

def bruit_pro_normalisant(l):
	'''	Afin d'eviter que de memes valeurs se presentent en suite
		Autrement la normalisation est parfois impossible
	'''
	_min = min(l)
	_max = max(l)
	alea = abs(_max-_min) * 0.0001
	return [i+alea*rnd() for i in l]

def lire_une_source(source):
	with open(source, "r") as co:
		text = co.read().split('\n')
		del text[0]
		del text[0]
		del text[-1]
		lignes = [l.split(',') for l in text][::-1]
		#
		NOM    = lignes[0][2]
		infos  = [(float(Close), float(Volume_BTC), float(Volume_USDT), float(Low), float(High), float(int(tradecount))) for Unix,Date,Symbol,Open,High,Low,Close,Volume_BTC,Volume_USDT,tradecount in lignes]
		#
	return infos, NOM

if __name__ == "__main__":
	from sys import argv

	sources        = [argv[i] for i in range(1, len(argv))]

	sources_lignes = [lire_une_source(source) for source in sources]

	sorties = {
		f'{NOM}/{nom_extraction}.bin' : bruit_pro_normalisant(exctration(lignes))
		for lignes,NOM in sources_lignes
			for exctration, nom_extraction in [
				(prixs,      'prixs'  ),
				( _low,      'low'    ),
				(_high,      'high'   ),
				(median,     'median' ),
				(volumes,    'volumes'),
				(volumes_A,  'volumes_A'),
				(volumes_U,  'volumes_U'),
				(tradecount, 'tradecount'),
		]
	}

	for fichier, liste in sorties.items():
		with open(fichier, "wb") as co:
			print(f'LEN {fichier}   = {len(liste)}')
			co.write(st.pack('I', len(liste)))
			co.write(st.pack('f'*len(liste), *liste))