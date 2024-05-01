#! /usr/bin/python3

from mdl import *

import matplotlib.pyplot as plt

signe = lambda x: (1 if x >= 0 else -1)

plusde50 = lambda x: (x if abs(x) >= 0.0 else 0)

prixs = I__sources[0]

if __name__ == "__main__":
	mdl_normale    = Mdl("mdl.bin"           )
	mdl_validation = Mdl("mdl_validation.bin")

	print("Calcule ...")
	pred_normale    = mdl_normale   ()
	pred_validaiton = mdl_validation()
	print(pred_normale   [-20:])
	print(pred_validaiton[-20:])
	print("Fin Calcule")

	_prixs = I__sources[0][DEPART:]

	print(len(pred_normale), len(_prixs))

	plt.plot(e_norme(_prixs));
	plt.plot(pred_normale, 'o');
	#
	plt.plot([0 for _ in pred_normale], label='-')
	for i in range(len(pred_normale)): plt.plot([i for _ in pred_normale], e_norme(list(range(len(pred_normale)))), '--')

	plt.show()

	##	================ Gain ===============

	for nom, pred in {'pred_normale':pred_normale, 'pred_validaiton':pred_validaiton}.items():
		LEVIER = 25#125
		METHODES  = ["LIBRE", "SIGNE", "REDUCTION PERTES"]
		METHODE = METHODES[0]
		POURCENTS = [0.01, 0.04, 0.08, 0.15, 0.25, 1.0]
		fig, axs = plt.subplots(len(POURCENTS), len(METHODES))
		for _i,POURCENT in enumerate(POURCENTS):
			#for _j,METHODE in enumerate(METHODES):
			for _j,LEVIER in enumerate([20, 50, 125]):
				for version in (+1,-1):
					#
					u = 40
					usd = [u]
					#
					for i in range(DEPART, I_PRIXS-1):

						__pred = version*pred[i-DEPART]#+0.30

						if METHODE == "LIBRE":
							u += u * LEVIER * POURCENT * (__pred) * (prixs[i+1]/prixs[i]-1)
						elif METHODE == "SIGNE":
							u += u * LEVIER * POURCENT * signe(__pred) * (prixs[i+1]/prixs[i]-1)
						elif METHODE == "REDUCTION PERTES":
							mise = u*POURCENT*abs(__pred)
							gain = mise*LEVIER*signe(__pred)*(prixs[i+1]/prixs[i]-1)
							assert low[ i ] <= prixs[ i ] <= hight[ i ]
							assert low[i+1] <= prixs[i+1] <= hight[i+1]
							if signe(__pred) == +1:
								if mise * LEVIER * (+1) * (low  [i+1]/prixs[i]-1) <= -mise:
									gain = -mise # Dès que la mise est perdu, la prediction est stopée
							else:
								if mise * LEVIER * (-1) * (hight[i+1]/prixs[i]-1) <= -mise:
									gain = -mise # Dès que la mise est perdu, la prediction est stopée
									
							u += gain
						else:
							raise Exception(f"Pas de METHODE : {METHODE}")

						if (u <= 0): u = 0
						usd += [u]

					if version == +1:
						axs[_i][_j].plot(usd, 'b', label=f'mdt={METHODE} %={POURCENT} x{LEVIER} u={float(int(usd[-1]))}')
					else:
						axs[_i][_j].twinx().plot(usd, 'r', label=f'mdt={METHODE} %={POURCENT} x-{LEVIER} u={float(int(usd[-1]))}')
				axs[_i][_j].legend()
		plt.title(nom)
		plt.show()