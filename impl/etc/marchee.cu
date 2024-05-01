#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

char * nom_sources[SOURCES] = {
	" prixs    BTC",
	"  haut    BTC",
	"  bas     BTC",
	" median   BTC",
	"volumes   BTC",
	"volumes A BTC",
	"volumes U BTC",
	//
	" prixs    ETH",
	"  haut    ETH",
	"  bas     ETH",
	" median   ETH"
	"volumes   ETH",
	"volumes A ETH",
	"volumes U ETH",
};

//	Sources
float     prixs_btc[PRIXS] = {};
float      high_btc[PRIXS] = {};
float       low_btc[PRIXS] = {};
float    median_btc[PRIXS] = {};
float   volumes_btc[PRIXS] = {};
float volumes_A_btc[PRIXS] = {};
float volumes_U_btc[PRIXS] = {};
//
float     prixs_eth[PRIXS] = {};
float      high_eth[PRIXS] = {};
float       low_eth[PRIXS] = {};
float    median_eth[PRIXS] = {};
float   volumes_eth[PRIXS] = {};
float volumes_A_eth[PRIXS] = {};
float volumes_U_eth[PRIXS] = {};


float *     prixs_btc__d = 0x0;
float *      high_btc__d = 0x0;
float *       low_btc__d = 0x0;
float *    median_btc__d = 0x0;
float *   volumes_btc__d = 0x0;
float * volumes_A_btc__d = 0x0;
float * volumes_U_btc__d = 0x0;
//
float *     prixs_eth__d = 0x0;
float *      high_eth__d = 0x0;
float *       low_eth__d = 0x0;
float *    median_eth__d = 0x0;
float *   volumes_eth__d = 0x0;
float * volumes_A_eth__d = 0x0;
float * volumes_U_eth__d = 0x0;


float * sources[SOURCES] = {
	prixs_btc, high_btc, low_btc, median_btc, volumes_btc, volumes_A_btc, volumes_U_btc,
	prixs_eth, high_eth, low_eth, median_eth, volumes_eth, volumes_A_eth, volumes_U_eth
};

float * sources__d[SOURCES] = {
	prixs_btc__d, high_btc__d, low_btc__d, median_btc__d, volumes_btc__d, volumes_A_btc__d, volumes_U_btc__d,
	prixs_eth__d, high_eth__d, low_eth__d, median_eth__d, volumes_eth__d, volumes_A_eth__d, volumes_U_eth__d
};

static void charger_une_source(float * ou, char * fichier) {
	FILE * fp = fopen(fichier, "rb");
	ASSERT(fp != 0);
	uint __PRIXS;
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(ou, sizeof(float), PRIXS, fp);
	fclose(fp);
};

void charger_les_prixs() {
	//	--- BTC ---
	charger_une_source(prixs_btc,     "prixs/BTCUSDT/prixs.bin"    );
	charger_une_source(high_btc,      "prixs/BTCUSDT/high.bin"     );
	charger_une_source(low_btc,       "prixs/BTCUSDT/median.bin"   );
	charger_une_source(median_btc,    "prixs/BTCUSDT/median.bin"   );
	charger_une_source(volumes_btc,   "prixs/BTCUSDT/volumes.bin"  );
	charger_une_source(volumes_A_btc, "prixs/BTCUSDT/volumes_A.bin");
	charger_une_source(volumes_U_btc, "prixs/BTCUSDT/volumes_U.bin");
	
	//	--- ETH ---
	charger_une_source(prixs_eth,     "prixs/ETHUSDT/prixs.bin"    );
	charger_une_source(high_eth,      "prixs/ETHUSDT/high.bin"     );
	charger_une_source(low_eth,       "prixs/ETHUSDT/median.bin"   );
	charger_une_source(median_eth,    "prixs/ETHUSDT/median.bin"   );
	charger_une_source(volumes_eth,   "prixs/ETHUSDT/volumes.bin"  );
	charger_une_source(volumes_A_eth, "prixs/ETHUSDT/volumes_A.bin");
	charger_une_source(volumes_U_eth, "prixs/ETHUSDT/volumes_U.bin");
};

//	===========================================================

void ema_int_calc_ema(ema_int_t * ema_int) {
	//			-- Parametres --
	uint K = ema_int->K_ema;
	float _K = 1.0 / ((float)K);
	//	EMA
	ema_int->ema[0] = sources[ema_int->source][0];
	FOR(1, i, PRIXS) {
		ema_int->ema[i] = ema_int->ema[i-1] * (1.0 - _K) + sources[ema_int->source][i]*_K;
	}
};

//	===========================================================

uint nature_multiple_interv[NATURES] = {
	0,
	0,
	0,
	14,
	14
};

nature_f fonctions_nature[NATURES] = {
	nature0__direct,
	nature1__macd,
	nature2__chiffre,
	nature3__awesome,
	nature4__pourcent_r,
	nature5__rsi,
};

uint NATURE_PARAMS[NATURES] = {
	0,
	1,
	1,
	1,
	2,
	2
};

uint min_param[NATURES][MAX_PARAMS] = {
	{0,0,0,0},
	{1,0,0,0},
	{1,0,0,0},
	{1,0,0,0},
	{1,1,0,0},
	{1,1,0,0}
};

uint max_param[NATURES][MAX_PARAMS] = {
	{0,                0,       0,        0      }, 
	{MAX_COEF_MACD,    0,       0,        0      },
	{MAX_CHIFFRE,      0,       0,        0      },
	{MAX_COEF_AWESOME, 0,       0,        0      },
	{MAX_INTERVALLE,   MAX_EMA, 0,        0      },
	{MAX_INTERVALLE,   MAX_EMA, 0,        0      } 
};

char * nom_natures[NATURES] {
	"directe",
	"  macd ",
	"chiffre",
	"awesome",
	"  %R   ",
	"  RSI  "
};

ema_int_t * cree_ligne(uint source, uint nature, uint K_ema, uint intervalle, uint params[MAX_PARAMS]) {
	ema_int_t * ret = alloc<ema_int_t>(1);
	//
	ret->source = source;
	ret->nature = nature;
	ret->K_ema  = K_ema;
	ret->intervalle = intervalle;
	//
	ASSERT(intervalle <= MAX_INTERVALLE);
	ASSERT(K_ema      <= MAX_EMA);
	//
	memcpy(ret->params, params, sizeof(uint) * MAX_PARAMS);
	//
	ema_int_calc_ema(ret);
	fonctions_nature[nature](ret);
	//
	return ret;
};

void liberer_ligne(ema_int_t * ema_int) {

};

void charger_vram_nvidia() {
	prixs_btc__d     = cpu_vers_gpu<float>(prixs_btc,     PRIXS);
	high_btc__d      = cpu_vers_gpu<float>(high_btc,      PRIXS);
	low_btc__d       = cpu_vers_gpu<float>(low_btc,       PRIXS);
	median_btc__d    = cpu_vers_gpu<float>(median_btc,    PRIXS);
	volumes_btc__d   = cpu_vers_gpu<float>(volumes_btc,   PRIXS);
	volumes_A_btc__d = cpu_vers_gpu<float>(volumes_A_btc, PRIXS);
	volumes_U_btc__d = cpu_vers_gpu<float>(volumes_U_btc, PRIXS);
	//
	prixs_eth__d     = cpu_vers_gpu<float>(prixs_eth,     PRIXS);
	high_eth__d      = cpu_vers_gpu<float>(high_eth,      PRIXS);
	low_eth__d       = cpu_vers_gpu<float>(low_eth,       PRIXS);
	median_eth__d    = cpu_vers_gpu<float>(median_eth,    PRIXS);
	volumes_eth__d   = cpu_vers_gpu<float>(volumes_eth,   PRIXS);
	volumes_A_eth__d = cpu_vers_gpu<float>(volumes_A_eth, PRIXS);
	volumes_U_eth__d = cpu_vers_gpu<float>(volumes_U_eth, PRIXS);
};

void     liberer_cudamalloc() {
	CONTROLE_CUDA(cudaFree(  prixs_btc__d));
	CONTROLE_CUDA(cudaFree(   high_btc__d));
	CONTROLE_CUDA(cudaFree(    low_btc__d));
	CONTROLE_CUDA(cudaFree( median_btc__d));
	CONTROLE_CUDA(cudaFree(volumes_btc__d));
	CONTROLE_CUDA(cudaFree(volumes_A_btc__d));
	CONTROLE_CUDA(cudaFree(volumes_U_btc__d));
	//
	CONTROLE_CUDA(cudaFree(  prixs_eth__d));
	CONTROLE_CUDA(cudaFree(   high_eth__d));
	CONTROLE_CUDA(cudaFree(    low_eth__d));
	CONTROLE_CUDA(cudaFree( median_eth__d));
	CONTROLE_CUDA(cudaFree(volumes_eth__d));
	CONTROLE_CUDA(cudaFree(volumes_A_eth__d));
	CONTROLE_CUDA(cudaFree(volumes_U_eth__d));
};

void charger_tout() {
	//	Assertions
	FOR(0, i, NATURES) ASSERT(nature_multiple_interv[i] <= MAX_MULTPLE_INTERV_NATURES);
	//
	printf("charger_les_prixs : ");    MESURER(charger_les_prixs());
	printf("charger_vram_nvidia : ");  MESURER(charger_vram_nvidia());
};

void liberer_tout() {
	titre("Liberer tout");
	liberer_cudamalloc();
};

ema_int_t * lire_ema_int(FILE * fp) {
	uint source, nature, K_ema, intervalle;
	uint params[MAX_PARAMS];
	FREAD(&source,     sizeof(uint), 1, fp);
	FREAD(&nature,     sizeof(uint), 1, fp);
	FREAD(&K_ema,      sizeof(uint), 1, fp);
	FREAD(&intervalle, sizeof(uint), 1, fp);
	//
	FREAD(&params,     sizeof(uint), MAX_PARAMS, fp);
	//
	return cree_ligne(source, nature, K_ema, intervalle, params);
};

void      ecrire_ema_int(ema_int_t * ema_int, FILE * fp) {
	FWRITE(&ema_int->source,     sizeof(uint), 1, fp);
	FWRITE(&ema_int->nature,     sizeof(uint), 1, fp);
	FWRITE(&ema_int->K_ema,      sizeof(uint), 1, fp);
	FWRITE(&ema_int->intervalle, sizeof(uint), 1, fp);
	//
	FWRITE(&ema_int->params,     sizeof(uint), MAX_PARAMS, fp);
};

char * nom_type_de_norme[3] = {
	"NORME_CLASSIQUE",
	"NORME_THEORIQUE",
	"NORME_RELATIVE "
};