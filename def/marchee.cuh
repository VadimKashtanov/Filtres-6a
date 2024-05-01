#pragma once

#include "etc.cuh"
#include "cuda_math.cuh"

#define PRIXS 37865
//	16*16 = 256 ~= 10.0j
//	8*8   = 64  ~=  2.5j
#define GRAND_T (8*8*1)//(8*8*1)
#define MEGA_T 12//24//1

#define N_FLTR  8
#define N       N_FLTR

#define MAX_INTERVALLE 256
#define MAX_MULTPLE_INTERV_NATURES 15
#define MULTIPLE MAX2(N_FLTR,MAX_MULTPLE_INTERV_NATURES)

#define VALIDATION (GRAND_T*MEGA_T)

#define DEPART (MULTIPLE*MAX_INTERVALLE)
#define FIN    (PRIXS-1-VALIDATION)

#define DEPART_VALIDATION (FIN)
#define FIN_VALIDATION    (PRIXS-1)

//	--- Sources ---

//	Marchés : BTCUSDT, ETHUSDT

#define SOURCES (7+7)

extern char * nom_sources[SOURCES];

#define SRC_PRIXS_BTC     0
#define SRC_HIGH_BTC      1
#define SRC_LOW_BTC       2
#define SRC_MEDIAN_BTC    3
#define SRC_VOLUMES_BTC   4
#define SRC_VOLUMES_A_BTC 5
#define SRC_VOLUMES_U_BTC 6
// --------------------- //
#define SRC_PRIXS_ETH     7
#define SRC_HIGH_ETH      8
#define SRC_LOW_ETH       9
#define SRC_MEDIAN_ETH    10
#define SRC_VOLUMES_ETH   11
#define SRC_VOLUMES_A_ETH 12
#define SRC_VOLUMES_U_ETH 13

//	Sources en CPU
extern float     prixs_btc[PRIXS];	//  prixs.bin
extern float      high_btc[PRIXS];	//   high.bin
extern float       low_btc[PRIXS];	//    low.bin
extern float    median_btc[PRIXS];	//  media.bin
extern float   volumes_btc[PRIXS];	// volume.bin
extern float volumes_A_btc[PRIXS];	// volume.bin
extern float volumes_U_btc[PRIXS];	// volume.bin
//
extern float     prixs_eth[PRIXS];	//  prixs.bin
extern float      high_eth[PRIXS];	//   high.bin
extern float       low_eth[PRIXS];	//    low.bin
extern float    median_eth[PRIXS];	//  media.bin
extern float   volumes_eth[PRIXS];	// volume.bin
extern float volumes_A_eth[PRIXS];	// volume.bin
extern float volumes_U_eth[PRIXS];	// volume.bin

extern float * sources[SOURCES];

//	Sources en GPU
extern float *    prixs_btc__d;	//	nVidia
extern float *     high_btc__d;	//	nVidia
extern float *      low_btc__d;	//	nVidia
extern float *   median_btc__d;	//	nVidia
extern float *  volumes_btc__d;	//	nVidia
extern float * volumes_A_btc__d;//	nVidia
extern float * volumes_U_btc__d;//	nVidia
//
extern float  *   prixs_eth__d;	//	nVidia
extern float  *    high_eth__d;	//	nVidia
extern float  *     low_eth__d;	//	nVidia
extern float  *  median_eth__d;	//	nVidia
extern float  * volumes_eth__d;	//	nVidia
extern float *volumes_A_eth__d;	//	nVidia
extern float *volumes_U_eth__d;	//	nVidia

extern float * sources__d[SOURCES];

#define      MARCHEE_DE_TRADE prixs_btc //prixs_eth
#define cuda_MARCHEE_DE_TRADE prixs_btc__d //prixs_eth__d

void   charger_les_prixs();
void charger_vram_nvidia();
//
void  liberer_cudamalloc();
//
void charger_tout();
void liberer_tout();

//	---	analyse des sources ---

#define MAX_PARAMS 4
#define    NATURES 6

//	-- Analysateurs --
#define     DIRECT 0
#define       MACD 1
#define    CHIFFRE 2
#define    AWESOME 3
#define POURCENT_R 4
#define        RSI 5

uint * cree_DIRECTE();					//classique
uint * cree_MACD(uint k);				//relative
uint * cree_CHIFFRE(uint chiffre);		//theorique
uint * cree_AWESOME(uint k);			//relative
uint * cree_POURCENT_R(uint interv, uint ema_K_post_r);	//theorique
uint * cree_RSI(uint interv);			//theorique

extern uint nature_multiple_interv[NATURES];

extern uint min_param[NATURES][MAX_PARAMS];
extern uint max_param[NATURES][MAX_PARAMS];

extern uint NATURE_PARAMS[NATURES];

extern char * nom_natures[NATURES];

#define MAX_EMA          1024
#define MAX_PLUS         500
#define MAX_COEF_MACD    500
#define MAX_COEF_AWESOME 500
#define MAX_CHIFFRE      10000

typedef struct {
	//	Intervalle
	uint      K_ema;	//ASSERT(1 <=      ema   <= inf           )
	uint intervalle;	//ASSERT(1 <= intervalle <= MAX_INTERVALLE)

	//	Nature
	uint nature;
	/*	Natures: ema-K, macd-k, chiffre-M, dx, dxdx, dxdxdx
			directe : {}							// Juste le Ema_int
			macd    : {coef }   					// le macd sera ema(9*c)-ema(26*c) sur ema(prixs,k)
			chiffre : {cible}						// Peut importe la cible, mais des chiffres comme 50, 100, 1.000 ... sont bien
	*/
	uint params[MAX_PARAMS];

	//	Valeurs
	float   ema[PRIXS];
	float brute[PRIXS];

	//	Gestion des Normes
#define NORME_CLASSIQUE 0 	//[f NORME] r = [(l[i]-min(l))/(max(l) - min(l))]
#define NORME_THEORIQUE 1 	//[f BORNE] r = [(l[i]-min_t)/(max_t-min_t)]
#define NORME_RELATIVE  2   //[f BORNE] r = [(l[i]--max|l]) / (max|l|--max|l|)] (ca devrait etre entre -1;1, mais en fait ca change rien)
	uint  type_de_norme;
	float min_theorique, max_theorique;

	/*
Norme classique : 
	- Des prixs qui peuvent prendre toutes valeurs
	Ex : les prixs du marchee, les volumes ...

Norme Theorique :
	- Des valeurs bornées.
	Ex : Des pourcentages%, rsi, des 'chiffres' ...

Norme Relative :
	- Des valeurs non bornée mais en pratique limités et ou le signe est important
	Ex : macd, awesome ...

	*/

	/*	Note : dans `normalisee` et `dif_normalisee`
	les intervalles sont deja calculee. Donc tout
	ce qui est avant DEPART n'est pas initialisee (car pas utilisee).
	*/
	uint source;
} ema_int_t;

extern char * nom_type_de_norme[3];

void ema_int_calc_ema(ema_int_t * ema_int);

//	Outils qui composent les natures
void _outil_ema(float * y, float * x, uint K);
void _outil_macd(float * y, float * x, float coef);
void _outil_chiffre(float * y, float * x, float chiffre);
void _outil_awesome(float * y, float * x, float coef);
void _outil_pourcent_r(float * y, float * x, uint interv, uint ema);
void _outil_rsi(float * y, float * x, uint interv);

//	Les natures
void nature0__direct    (ema_int_t * ema_int);
void nature1__macd      (ema_int_t * ema_int);
void nature2__chiffre   (ema_int_t * ema_int);
void nature3__awesome   (ema_int_t * ema_int);
void nature4__pourcent_r(ema_int_t * ema_int);
void nature5__rsi       (ema_int_t * ema_int);

typedef void (*nature_f)(ema_int_t*);
extern nature_f fonctions_nature[NATURES];

//	Mem
ema_int_t * cree_ligne(uint source, uint nature, uint K_ema, uint intervalle, uint params[MAX_PARAMS]);
void     liberer_ligne(ema_int_t * ema_int);

//	IO
ema_int_t * lire_ema_int(FILE * fp);
void      ecrire_ema_int(ema_int_t * ema_int, FILE * fp);

//	Visualisation simple Matplotlib
void visualiser_ema_int(
	uint source,
	uint nature,
	uint K_ema, uint intervalle,
	uint params[MAX_PARAMS]);