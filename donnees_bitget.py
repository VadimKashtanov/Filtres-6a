import time
import datetime

import requests

import matplotlib.pyplot as plt

ARONDIRE_AU_MODULO = lambda x,mod: (x + (mod - (x%mod)) if x%mod!=0 else x)

milliseconde = lambda la: int(la * 1000   )*1
seconde      = lambda la: int(la          )*1000
heure        = lambda la: int(la / (60*60))*1000*60*60

requette_bitget = lambda de, a, SYMBOLE: eval(
	requests.get(
		f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol={SYMBOLE}_UMCBL&granularity=1H&startTime={de}&endTime={a}"
	).text
)

HEURES_PAR_REQUETTE = 100

T = (30)*24 #(2*4)*7*24

__DEPART = (15+0)*256

la = heure(time.time())
heures_voulues = [
	la - 60*60*1000*i
	for i in range(ARONDIRE_AU_MODULO(__DEPART+T, HEURES_PAR_REQUETTE))
][::-1] # <<<< ---- ??????

donnees_BTCUSDT = []
donnees_ETHUSDT = []

REQUETTES = int(len(heures_voulues) / HEURES_PAR_REQUETTE)
print(f"Extraction de {len(heures_voulues)} heures depuis api.bitget.com ...")
for i in range(REQUETTES):
	paquet_heures_btc = requette_bitget(heures_voulues[i*HEURES_PAR_REQUETTE], heures_voulues[(i+1)*HEURES_PAR_REQUETTE-1], "BTCUSDT")
	paquet_heures_eth = requette_bitget(heures_voulues[i*HEURES_PAR_REQUETTE], heures_voulues[(i+1)*HEURES_PAR_REQUETTE-1], "ETHUSDT")
	donnees_BTCUSDT += paquet_heures_btc
	donnees_ETHUSDT += paquet_heures_eth

	if i % 1 == 0:
		print(f"[{round(i*HEURES_PAR_REQUETTE/len(heures_voulues)*100)}%],   len(paquet_heures_btc)={len(paquet_heures_btc)}, len(paquet_heures_eth)={len(paquet_heures_eth)} (btc, eth)")

#donnees = donnees[::-1]

print(f"HEURES VOULUES = {len(heures_voulues)}, len(donnees_BTCUSDT)={len(donnees_BTCUSDT)}, len(donnees_ETHUSDT)={len(donnees_ETHUSDT)}")

prixs     = lambda donnees: [float(c)                       for _,o,h,l,c,vB,vU in donnees]
hight     = lambda donnees: [float(h)                       for _,o,h,l,c,vB,vU in donnees]
low       = lambda donnees: [float(l)                       for _,o,h,l,c,vB,vU in donnees]
volumes   = lambda donnees: [float(c)*float(vB) - float(vU) for _,o,h,l,c,vB,vU in donnees]
volumes_A = lambda donnees: [float(vB)                      for _,o,h,l,c,vB,vU in donnees]
volumes_U = lambda donnees: [float(vU)                      for _,o,h,l,c,vB,vU in donnees]
median    = lambda donnees: [(float(h)+float(l))/2          for _,o,h,l,c,vB,vU in donnees]

I__sources = [
	    prixs(donnees_BTCUSDT),
	    hight(donnees_BTCUSDT),
	      low(donnees_BTCUSDT),
	   median(donnees_BTCUSDT),
 	  volumes(donnees_BTCUSDT),
	volumes_A(donnees_BTCUSDT),
	volumes_U(donnees_BTCUSDT),
	# --- 
	    prixs(donnees_ETHUSDT),
	    hight(donnees_ETHUSDT),
	      low(donnees_ETHUSDT),
	   median(donnees_ETHUSDT),
 	  volumes(donnees_ETHUSDT),
	volumes_A(donnees_ETHUSDT),
	volumes_U(donnees_ETHUSDT),

]

I_PRIXS = len(I__sources[0])

print("5 dernieres heures BTC : ", I__sources[0*7+0][-5:])
print("5 dernieres heures ETH : ", I__sources[1*7+0][-5:])