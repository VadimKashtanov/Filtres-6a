#include "main.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

//static float _pourcent_masque_nulle[C] = {0};
//static float _alpha[C] = {0.01};

__global__
static void kerd_p1e5(float * p, uint i, float _1E5) {
	p[i] += _1E5;
};

static void p1e5(Mdl_t * mdl, uint c, uint i, float _1E5, uint _MODE) {
	kerd_p1e5<<<1,1>>>(mdl->p__d[c], i, _1E5);
	ATTENDRE_CUDA();
};

static void __performance() {
	/*ASSERT(C == 11);
	titre("Performance");
	//
	uint Y[C] = {
		512,
		256,
		256,
		256,
		128,
		64,
		32,
		16,
		8,
		4,
		P
	};
	uint insts[C] = {
		FILTRES_PRIXS,
		LSTM1D,
		LSTM1D,
		LSTM1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D
	};
	uint lignes[BLOQUES] = {0};
	FOR(0, i, BLOQUES) lignes[i] = rand() % EMA_INTS;
	Mdl_t * mdl = cree_mdl(Y, insts, lignes);
	plumer_mdl(mdl);
	//
	uint plus_T = 16*16*25;
	//
	mdl_plume_grad(mdl, DEPART, DEPART+plus_T);
	//
	printf("TEMPS MODEL = ");
	MESURER(mdl_aller_retour(mdl, DEPART, DEPART+plus_T, 3));
	//
	liberer_mdl(mdl);*/
};

static void __verif_mdl_1e5() {
	/*ASSERT(SCORE_Y_COEF_BRUIT == 0.0);
	ASSERT(C == 3);
	titre("Comparer MODEL 1e-5");
	//
	uint Y[C];
	uint insts[C];
	//
	uint st[C][2] = {
		{64, FILTRES_PRIXS},
		{64, LSTM1D_PEEPHOLE},
		//{64, LSTM1D_PEEPHOLE},
		{1,  DOT1D_TANH},
	};
	FOR(0, i, C) {
		    Y[i] = st[i][0];
		insts[i] = st[i][1];
	}
	ema_int_t * bloque[BLOQUES] = {
	//			  Source,      Nature,  K_ema, Intervalle,    {params}
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 4, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 4, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 1, cree_DIRECTE())
	};
	ASSERT(F_PAR_BLOQUES == 8);
	ASSERT(BLOQUES == 8);
	//
	//
	uint plus_T = 8*8*1;
	//
	//
	Mdl_t * mdl = cree_mdl(plus_T, Y, insts, bloque);
	plumer_mdl(mdl);
	//
	//comportement(mdl, DEPART, DEPART+plus_T);
	//
	srand(0);
	uint MODE_t_MODE  = t_CONTINUE;//t_PSEUDO_ALEA;//t_CONTINUE;
	uint grain_t_MODE = rand() % 10000;
	//
#define MODE 1 //MODE_NAIF, MODE_MAXIMALE
	//
	printf("aller_retour T=%i : ", plus_T);
	MESURER(mdl_aller_retour(mdl, DEPART, DEPART+plus_T, MODE, MODE_t_MODE, grain_t_MODE));
	//
	//	1e-5
	//
	INIT_CHRONO(temps_execution);
	DEPART_CHRONO(temps_execution);
	//
	mdl_zero_gpu(mdl);
	float _f = mdl_score(mdl, DEPART, DEPART+plus_T, MODE, MODE_t_MODE, grain_t_MODE);
	float _1E5 = 1e-2f;
	FOR(0, c, C) {
		printf("###############################################################\n");
		printf("                       C = %2.i (%s)    \n", c, nom_inst[mdl->insts[c]]);
		printf("#######################vvvvvvvvvvvvvv##########################\n");
		//
		float * dp = gpu_vers_cpu<float>(mdl->dp__d[c], mdl->inst_POIDS[c]);
		//
		FOR(0, i, mdl->inst_POIDS[c]) {
			p1e5(mdl, c, i, +_1E5, MODE);
			float grad_1e5 = (mdl_score(mdl, DEPART, DEPART+plus_T, MODE, MODE_t_MODE, grain_t_MODE) - _f)/_1E5;
			p1e5(mdl, c, i, -_1E5, MODE);
			//
			float a=grad_1e5, b=dp[i];
			plume_separateur_p(mdl, c, i);
			printf("%i| ", i);
			PLUME_CMP(a, b);
			if (b != 0) printf(" (x%f) ", a/b);
			printf("\n");
		}
		//
		free(dp);
	};
	printf("  1e5 === df(x)  \n");

	printf("Temps total = %f\n", VALEUR_CHRONO(temps_execution));

	//
	liberer_mdl(mdl);*/
};

void verif_mdl_1e5() {
	__performance();
	__verif_mdl_1e5();
};