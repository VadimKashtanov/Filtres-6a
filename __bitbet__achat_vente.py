from os import system
from pybitget import Client

client = Client()

argent = lambda : float(client.mix_get_account('BTCUSDT_UMCBL', 'USDT')['data']['usdtEquity'])

print(f"Le compte a {argent()} $")

SYMBOLE     = 'BTCUSDT_UMCBL'
productType = 'umcbl'
marginCoin  = 'USDT'

from def_futures_bitget import fermer_les_positions, executer_future, derniere_prediction

POURCENT = 0.10

fermer_les_positions(client)

#   === Achat / Vente ===

REFAIRE_CALCULE = True

__pred = derniere_prediction(REFAIRE_CALCULE)
print(f"Prediction : {__pred}")
#executer_future(client, argent(), POURCENT*abs(__pred), __pred/abs(__pred))
