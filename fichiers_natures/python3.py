prixs_BTC, haut_BTC, bas_BTC, median_BTC, volumes_BTC, volumes_A_BTC, volumes_U_BTC = "SRC_PRIXS_BTC", "SRC_HIGH_BTC", "SRC_LOW_BTC", "SRC_MEDIAN_BTC", "SRC_VOLUMES_BTC", "SRC_VOLUMES_A_BTC", "SRC_VOLUMES_U_BTC"
prixs_ETH, haut_ETH, bas_ETH, median_ETH, volumes_ETH, volumes_A_ETH, volumes_U_ETH = "SRC_PRIXS_ETH", "SRC_HIGH_ETH", "SRC_LOW_ETH", "SRC_MEDIAN_ETH", "SRC_VOLUMES_ETH", "SRC_VOLUMES_A_ETH", "SRC_VOLUMES_U_ETH"

directes = "DIRECT", """
(1,2,4,8,16,32,64,128,256,512,1024)
""", (
	prixs_BTC, haut_BTC, bas_BTC, volumes_BTC, volumes_A_BTC, volumes_U_BTC,
	#prixs_ETH, haut_ETH, bas_ETH, volumes_ETH, volumes_A_ETH, volumes_U_ETH,
	), "cree_DIRECTE"
#""", (prixs_BTC, haut_BTC, bas_BTC, volumes_BTC, prixs_ETH, haut_ETH, bas_ETH, volumes_ETH), "cree_DIRECTE"

macds = "MACD", """
{'K': 1, 'interv': 1, 'params': (1,)}
{'K': 1, 'interv': 2, 'params': (1,)}
{'K': 1, 'interv': 2, 'params': (2,)}
{'K': 1, 'interv': 8, 'params': (1,)}
{'K': 1, 'interv': 8, 'params': (2,)}
{'K': 1, 'interv': 8, 'params': (4,)}
{'K': 1, 'interv': 8, 'params': (8,)}
{'K': 2, 'interv': 1.0, 'params': (1,)}
{'K': 2, 'interv': 2, 'params': (1,)}
{'K': 2, 'interv': 2, 'params': (2,)}
{'K': 2, 'interv': 4, 'params': (1,)}
{'K': 2, 'interv': 4, 'params': (2,)}
{'K': 2, 'interv': 4, 'params': (4,)}
{'K': 2, 'interv': 16, 'params': (2,)}
{'K': 2, 'interv': 16, 'params': (4,)}
{'K': 2, 'interv': 16, 'params': (8,)}
{'K': 2, 'interv': 16, 'params': (16,)}
{'K': 4, 'interv': 2.0, 'params': (1,)}
{'K': 4, 'interv': 2.0, 'params': (2,)}
{'K': 4, 'interv': 4, 'params': (1,)}
{'K': 4, 'interv': 4, 'params': (2,)}
{'K': 4, 'interv': 4, 'params': (4,)}
{'K': 4, 'interv': 8, 'params': (1,)}
{'K': 4, 'interv': 8, 'params': (2,)}
{'K': 4, 'interv': 8, 'params': (4,)}
{'K': 4, 'interv': 8, 'params': (8,)}
{'K': 4, 'interv': 32, 'params': (4,)}
{'K': 4, 'interv': 32, 'params': (8,)}
{'K': 4, 'interv': 32, 'params': (16,)}
{'K': 4, 'interv': 32, 'params': (32,)}
{'K': 8, 'interv': 4.0, 'params': (1,)}
{'K': 8, 'interv': 4.0, 'params': (2,)}
{'K': 8, 'interv': 4.0, 'params': (4,)}
{'K': 8, 'interv': 8, 'params': (1,)}
{'K': 8, 'interv': 8, 'params': (2,)}
{'K': 8, 'interv': 8, 'params': (4,)}
{'K': 8, 'interv': 8, 'params': (8,)}
{'K': 8, 'interv': 16, 'params': (2,)}
{'K': 8, 'interv': 16, 'params': (4,)}
{'K': 8, 'interv': 16, 'params': (8,)}
{'K': 8, 'interv': 16, 'params': (16,)}
{'K': 8, 'interv': 64, 'params': (8,)}
{'K': 8, 'interv': 64, 'params': (16,)}
{'K': 8, 'interv': 64, 'params': (32,)}
{'K': 8, 'interv': 64, 'params': (64,)}
{'K': 16, 'interv': 8.0, 'params': (1,)}
{'K': 16, 'interv': 8.0, 'params': (2,)}
{'K': 16, 'interv': 8.0, 'params': (4,)}
{'K': 16, 'interv': 8.0, 'params': (8,)}
{'K': 16, 'interv': 16, 'params': (2,)}
{'K': 16, 'interv': 16, 'params': (4,)}
{'K': 16, 'interv': 16, 'params': (8,)}
{'K': 16, 'interv': 16, 'params': (16,)}
{'K': 16, 'interv': 32, 'params': (4,)}
{'K': 16, 'interv': 32, 'params': (8,)}
{'K': 16, 'interv': 32, 'params': (16,)}
{'K': 16, 'interv': 32, 'params': (32,)}
{'K': 16, 'interv': 128, 'params': (16,)}
{'K': 16, 'interv': 128, 'params': (32,)}
{'K': 16, 'interv': 128, 'params': (64,)}
{'K': 16, 'interv': 128, 'params': (128,)}
{'K': 32, 'interv': 16.0, 'params': (2,)}
{'K': 32, 'interv': 16.0, 'params': (4,)}
{'K': 32, 'interv': 16.0, 'params': (8,)}
{'K': 32, 'interv': 16.0, 'params': (16,)}
{'K': 32, 'interv': 32, 'params': (4,)}
{'K': 32, 'interv': 32, 'params': (8,)}
{'K': 32, 'interv': 32, 'params': (16,)}
{'K': 32, 'interv': 32, 'params': (32,)}
{'K': 32, 'interv': 64, 'params': (8,)}
{'K': 32, 'interv': 64, 'params': (16,)}
{'K': 32, 'interv': 64, 'params': (32,)}
{'K': 32, 'interv': 64, 'params': (64,)}
{'K': 32, 'interv': 256, 'params': (32,)}
{'K': 32, 'interv': 256, 'params': (64,)}
{'K': 32, 'interv': 256, 'params': (128,)}
{'K': 32, 'interv': 256, 'params': (256,)}
{'K': 64, 'interv': 32.0, 'params': (4,)}
{'K': 64, 'interv': 32.0, 'params': (8,)}
{'K': 64, 'interv': 32.0, 'params': (16,)}
{'K': 64, 'interv': 32.0, 'params': (32,)}
{'K': 64, 'interv': 64, 'params': (8,)}
{'K': 64, 'interv': 64, 'params': (16,)}
{'K': 64, 'interv': 64, 'params': (32,)}
{'K': 64, 'interv': 64, 'params': (64,)}
{'K': 64, 'interv': 128, 'params': (16,)}
{'K': 64, 'interv': 128, 'params': (32,)}
{'K': 64, 'interv': 128, 'params': (64,)}
{'K': 64, 'interv': 128, 'params': (128,)}
{'K': 256, 'interv': 128.0, 'params': (16,)}
{'K': 256, 'interv': 128.0, 'params': (32,)}
{'K': 256, 'interv': 128.0, 'params': (64,)}
{'K': 256, 'interv': 128.0, 'params': (128,)}
{'K': 256, 'interv': 256, 'params': (32,)}
{'K': 256, 'interv': 256, 'params': (64,)}
{'K': 256, 'interv': 256, 'params': (128,)}
{'K': 256, 'interv': 256, 'params': (256,)}
""", (prixs_BTC,# haut_BTC, bas_BTC,
	#prixs_ETH,# haut_ETH, bas_ETH,
	), "cree_MACD"
#""", (prixs_BTC, haut_BTC, bas_BTC, prixs_ETH, haut_ETH, bas_ETH), "cree_MACD"

chiffres = "CHIFFRE", """
{'K': 1, 'interv': 4, 'params': (10000,)}
{'K': 1, 'interv': 8, 'params': (10000,)}
{'K': 1, 'interv': 8, 'params': (10000,)}
{'K': 8, 'interv': 4.0, 'params': (10000,)}
{'K': 8, 'interv': 8, 'params': (10000,)}
{'K': 8, 'interv': 8, 'params': (10000,)}
{'K': 8, 'interv': 32, 'params': (10000,)}
{'K': 8, 'interv': 32, 'params': (10000,)}
{'K': 8, 'interv': 64, 'params': (10000,)}
{'K': 8, 'interv': 64, 'params': (10000,)}
{'K': 64, 'interv': 32.0, 'params': (10000,)}
{'K': 64, 'interv': 32.0, 'params': (10000,)}
{'K': 64, 'interv': 64, 'params': (10000,)}
{'K': 64, 'interv': 64, 'params': (10000,)}
{'K': 64, 'interv': 256, 'params': (10000,)}
{'K': 64, 'interv': 256, 'params': (10000,)}
{'K': 256, 'interv': 128.0, 'params': (10000,)}
{'K': 256, 'interv': 128.0, 'params': (10000,)}
{'K': 256, 'interv': 256, 'params': (10000,)}
{'K': 256, 'interv': 256, 'params': (10000,)}
""", (haut_BTC, bas_BTC), "cree_CHIFFRE"

awesome = "AWESOME", """
{'K': 1, 'interv': 1, 'params': (1,)}
{'K': 1, 'interv': 2, 'params': (1,)}
{'K': 1, 'interv': 2, 'params': (2,)}
{'K': 1, 'interv': 8, 'params': (1,)}
{'K': 1, 'interv': 8, 'params': (2,)}
{'K': 1, 'interv': 8, 'params': (4,)}
{'K': 1, 'interv': 8, 'params': (8,)}
{'K': 2, 'interv': 1.0, 'params': (1,)}
{'K': 2, 'interv': 2, 'params': (1,)}
{'K': 2, 'interv': 2, 'params': (2,)}
{'K': 2, 'interv': 4, 'params': (1,)}
{'K': 2, 'interv': 4, 'params': (2,)}
{'K': 2, 'interv': 4, 'params': (4,)}
{'K': 2, 'interv': 16, 'params': (2,)}
{'K': 2, 'interv': 16, 'params': (4,)}
{'K': 2, 'interv': 16, 'params': (8,)}
{'K': 2, 'interv': 16, 'params': (16,)}
{'K': 4, 'interv': 2.0, 'params': (1,)}
{'K': 4, 'interv': 2.0, 'params': (2,)}
{'K': 4, 'interv': 4, 'params': (1,)}
{'K': 4, 'interv': 4, 'params': (2,)}
{'K': 4, 'interv': 4, 'params': (4,)}
{'K': 4, 'interv': 8, 'params': (1,)}
{'K': 4, 'interv': 8, 'params': (2,)}
{'K': 4, 'interv': 8, 'params': (4,)}
{'K': 4, 'interv': 8, 'params': (8,)}
{'K': 4, 'interv': 32, 'params': (4,)}
{'K': 4, 'interv': 32, 'params': (8,)}
{'K': 4, 'interv': 32, 'params': (16,)}
{'K': 4, 'interv': 32, 'params': (32,)}
{'K': 8, 'interv': 4.0, 'params': (1,)}
{'K': 8, 'interv': 4.0, 'params': (2,)}
{'K': 8, 'interv': 4.0, 'params': (4,)}
{'K': 8, 'interv': 8, 'params': (1,)}
{'K': 8, 'interv': 8, 'params': (2,)}
{'K': 8, 'interv': 8, 'params': (4,)}
{'K': 8, 'interv': 8, 'params': (8,)}
{'K': 8, 'interv': 16, 'params': (2,)}
{'K': 8, 'interv': 16, 'params': (4,)}
{'K': 8, 'interv': 16, 'params': (8,)}
{'K': 8, 'interv': 16, 'params': (16,)}
{'K': 8, 'interv': 64, 'params': (8,)}
{'K': 8, 'interv': 64, 'params': (16,)}
{'K': 8, 'interv': 64, 'params': (32,)}
{'K': 8, 'interv': 64, 'params': (64,)}
{'K': 16, 'interv': 8.0, 'params': (1,)}
{'K': 16, 'interv': 8.0, 'params': (2,)}
{'K': 16, 'interv': 8.0, 'params': (4,)}
{'K': 16, 'interv': 8.0, 'params': (8,)}
{'K': 16, 'interv': 16, 'params': (2,)}
{'K': 16, 'interv': 16, 'params': (4,)}
{'K': 16, 'interv': 16, 'params': (8,)}
{'K': 16, 'interv': 16, 'params': (16,)}
{'K': 16, 'interv': 32, 'params': (4,)}
{'K': 16, 'interv': 32, 'params': (8,)}
{'K': 16, 'interv': 32, 'params': (16,)}
{'K': 16, 'interv': 32, 'params': (32,)}
{'K': 16, 'interv': 128, 'params': (16,)}
{'K': 16, 'interv': 128, 'params': (32,)}
{'K': 16, 'interv': 128, 'params': (64,)}
{'K': 16, 'interv': 128, 'params': (128,)}
{'K': 32, 'interv': 16.0, 'params': (2,)}
{'K': 32, 'interv': 16.0, 'params': (4,)}
{'K': 32, 'interv': 16.0, 'params': (8,)}
{'K': 32, 'interv': 16.0, 'params': (16,)}
{'K': 32, 'interv': 32, 'params': (4,)}
{'K': 32, 'interv': 32, 'params': (8,)}
{'K': 32, 'interv': 32, 'params': (16,)}
{'K': 32, 'interv': 32, 'params': (32,)}
{'K': 32, 'interv': 64, 'params': (8,)}
{'K': 32, 'interv': 64, 'params': (16,)}
{'K': 32, 'interv': 64, 'params': (32,)}
{'K': 32, 'interv': 64, 'params': (64,)}
{'K': 32, 'interv': 256, 'params': (32,)}
{'K': 32, 'interv': 256, 'params': (64,)}
{'K': 32, 'interv': 256, 'params': (128,)}
{'K': 32, 'interv': 256, 'params': (256,)}
{'K': 64, 'interv': 32.0, 'params': (4,)}
{'K': 64, 'interv': 32.0, 'params': (8,)}
{'K': 64, 'interv': 32.0, 'params': (16,)}
{'K': 64, 'interv': 32.0, 'params': (32,)}
{'K': 64, 'interv': 64, 'params': (8,)}
{'K': 64, 'interv': 64, 'params': (16,)}
{'K': 64, 'interv': 64, 'params': (32,)}
{'K': 64, 'interv': 64, 'params': (64,)}
{'K': 64, 'interv': 128, 'params': (16,)}
{'K': 64, 'interv': 128, 'params': (32,)}
{'K': 64, 'interv': 128, 'params': (64,)}
{'K': 64, 'interv': 128, 'params': (128,)}
{'K': 256, 'interv': 128.0, 'params': (16,)}
{'K': 256, 'interv': 128.0, 'params': (32,)}
{'K': 256, 'interv': 128.0, 'params': (64,)}
{'K': 256, 'interv': 128.0, 'params': (128,)}
{'K': 256, 'interv': 256, 'params': (32,)}
{'K': 256, 'interv': 256, 'params': (64,)}
{'K': 256, 'interv': 256, 'params': (128,)}
{'K': 256, 'interv': 256, 'params': (256,)}
""", (median_BTC,
	#prixs_ETH, median_ETH
	), "cree_AWESOME"
#""", (prixs, haut, bas, volumes, median), "cree_AWESOME"

pourcent_r = "POURCENT_R", """
{'K': 1, 'interv': 4, 'params': (4, 2)}
{'K': 1, 'interv': 8, 'params': (8, 2)}
{'K': 1, 'interv': 8, 'params': (8, 2)}
{'K': 4, 'interv': 2.0, 'params': (2.0, 2)}
{'K': 4, 'interv': 4, 'params': (4, 2)}
{'K': 4, 'interv': 16, 'params': (16, 2)}
{'K': 4, 'interv': 16, 'params': (16, 2)}
{'K': 4, 'interv': 32, 'params': (32, 2)}
{'K': 4, 'interv': 32, 'params': (32, 2)}
{'K': 16, 'interv': 8.0, 'params': (8.0, 2)}
{'K': 16, 'interv': 8.0, 'params': (8.0, 2)}
{'K': 16, 'interv': 16, 'params': (16, 2)}
{'K': 16, 'interv': 16, 'params': (16, 2)}
{'K': 16, 'interv': 64, 'params': (64, 2)}
{'K': 16, 'interv': 64, 'params': (64, 2)}
{'K': 16, 'interv': 128, 'params': (128, 2)}
{'K': 16, 'interv': 128, 'params': (128, 2)}
{'K': 32, 'interv': 16.0, 'params': (16.0, 2)}
{'K': 32, 'interv': 16.0, 'params': (16.0, 2)}
{'K': 32, 'interv': 32, 'params': (32, 2)}
{'K': 32, 'interv': 32, 'params': (32, 2)}
{'K': 32, 'interv': 128, 'params': (128, 2)}
{'K': 32, 'interv': 128, 'params': (128, 2)}
{'K': 32, 'interv': 256, 'params': (256, 2)}
{'K': 32, 'interv': 256, 'params': (256, 2)}
{'K': 64, 'interv': 32.0, 'params': (32.0, 2)}
{'K': 64, 'interv': 32.0, 'params': (32.0, 2)}
{'K': 64, 'interv': 64, 'params': (64, 2)}
{'K': 64, 'interv': 64, 'params': (64, 2)}
{'K': 64, 'interv': 256, 'params': (256, 2)}
{'K': 64, 'interv': 256, 'params': (256, 2)}
{'K': 256, 'interv': 128.0, 'params': (128.0, 2)}
{'K': 256, 'interv': 128.0, 'params': (128.0, 2)}
{'K': 256, 'interv': 256, 'params': (256, 2)}
{'K': 256, 'interv': 256, 'params': (256, 2)}
""", (prixs_BTC,
	#prixs_ETH, haut_ETH, bas_ETH, volumes_ETH
	), "cree_POURCENT_R"

rsi = "RSI", """
{'K': 1, 'interv': 4, 'params': (4,)}
{'K': 1, 'interv': 8, 'params': (8,)}
{'K': 1, 'interv': 8, 'params': (8,)}
{'K': 4, 'interv': 2.0, 'params': (2.0,)}
{'K': 4, 'interv': 4, 'params': (4,)}
{'K': 4, 'interv': 16, 'params': (16,)}
{'K': 4, 'interv': 16, 'params': (16,)}
{'K': 4, 'interv': 32, 'params': (32,)}
{'K': 4, 'interv': 32, 'params': (32,)}
{'K': 16, 'interv': 8.0, 'params': (8.0,)}
{'K': 16, 'interv': 8.0, 'params': (8.0,)}
{'K': 16, 'interv': 16, 'params': (16,)}
{'K': 16, 'interv': 16, 'params': (16,)}
{'K': 16, 'interv': 64, 'params': (64,)}
{'K': 16, 'interv': 64, 'params': (64,)}
{'K': 16, 'interv': 128, 'params': (128,)}
{'K': 16, 'interv': 128, 'params': (128,)}
{'K': 32, 'interv': 16.0, 'params': (16.0,)}
{'K': 32, 'interv': 16.0, 'params': (16.0,)}
{'K': 32, 'interv': 32, 'params': (32,)}
{'K': 32, 'interv': 32, 'params': (32,)}
{'K': 32, 'interv': 128, 'params': (128,)}
{'K': 32, 'interv': 128, 'params': (128,)}
{'K': 32, 'interv': 256, 'params': (256,)}
{'K': 32, 'interv': 256, 'params': (256,)}
{'K': 64, 'interv': 32.0, 'params': (32.0,)}
{'K': 64, 'interv': 32.0, 'params': (32.0,)}
{'K': 64, 'interv': 64, 'params': (64,)}
{'K': 64, 'interv': 64, 'params': (64,)}
{'K': 64, 'interv': 256, 'params': (256,)}
{'K': 64, 'interv': 256, 'params': (256,)}
{'K': 256, 'interv': 128.0, 'params': (128.0,)}
{'K': 256, 'interv': 128.0, 'params': (128.0,)}
{'K': 256, 'interv': 256, 'params': (256,)}
{'K': 256, 'interv': 256, 'params': (256,)}
""", (prixs_BTC, 
	#prixs_ETH
	), "cree_RSI"

ANALYSES = (directes, macds)#, awesome, rsi, chiffres, pourcent_r)

N = sum(1 for n,l,s,f in ANALYSES for src in s for i in list(map(eval, l.strip('\n').split('\n'))))

k = 0
for nom, lignes, sources, fonc_params in ANALYSES:
	lignes = list(map(eval, lignes.strip('\n').split('\n')))
	#print("\t// -------")
	for src in sources:
		for i in lignes:
			print(f"\t\tcree_ligne({src}, {nom}, {i['K']}, {i['interv']}, {fonc_params}{str(i['params']).replace(',)',')')}),")
			k += 1
			K = 64#int(N/8)
			#if (k % K) == 0: print(f"// ----------- bloque ---------- ({k})")

print(f"\nlignes = {k}")