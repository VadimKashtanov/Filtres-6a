#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static uint mise_a_jour_fichier(char * fichier, float score)
{
	FILE * fp = fopen(fichier, "rb");
	//
	uint meilleur = true;
	//
	if (fp == 0) {
		fp = fopen(fichier, "wb");
		ecrire_un<uint>(fp, 1);
		ecrire_un<float>(fp, score);
		fclose(fp);
	} else {
		uint I = lire_un<uint>(fp);
		float * scores = lire<float>(fp, I);
		fclose(fp);
		//
		meilleur = score PLUS_PETIT_QUE scores[I-1];
		//
		fp = fopen(fichier, "wb");
		ecrire_un<uint>(fp, I+1);
		ecrire<float>(fp, scores, I);
		ecrire_un<float>(fp, score);
		fclose(fp);
		//
		free(scores);
	};
	//
	return meilleur;
};

static void visualiser() {
	uint source     = SRC_PRIXS_BTC;
	uint nature     = POURCENT_R;
	uint K_ema      = 64;
	uint intervalle = 128;
	uint * params   = cree_POURCENT_R(128, 2);
	visualiser_ema_int(
		source,
		nature,
		K_ema, intervalle,
		params);
};

static void plume_pred(Mdl_t * mdl, uint t0, uint t1) {
	Stats_t * stats = statistiques(mdl, t0, t1);
	//
	printf("PRED GENERALE = %f%% | LES GAINS^2 = %f%% | LES GAINS^4 = %f%% | LES GAINS^8 = %f%%\n",
		100*stats->pred,
		100*stats->les_gains__2,
		100*stats->les_gains__4,
		100*stats->les_gains__8
	);
	//
	mise_a_jour_fichier("mdl", stats->score);
	//
	free(stats);
};

static void enregistrer_par_validation(Mdl_t * mdl) {
	Stats_t * stats = statistiques(mdl, DEPART_VALIDATION, FIN_VALIDATION);
	//
	printf("Validation = %f%% (score=%f) | Les gains^2 = %f%% | Les gains^4 = %f%% | Les gains^8 = %f%%\n", 
		100*stats->pred,
		    stats->score,
		100*stats->les_gains__2,
		100*stats->les_gains__4,
		100*stats->les_gains__8
	);
	//
	if (mise_a_jour_fichier("mdl_validation", stats->score) == 1) {
		ecrire_mdl(mdl, "mdl_validation.bin");
	};
	//
	free(stats);
	printf("Fin validation\n");
};

float * pourcent_masque_nulle = de_a(0.0, 0.0, C);
float * pourcent_masque_opti_nulle = de_a(0.0, 0.0, C);

float * pourcent_masque = de_a(0.00, 0.00, C);			//	Des poids nulls
float * pourcent_masque_opti = de_a(0.30, 0.00, C);		//	Des poids qui ne s'optimiseront pas

float * alpha = de_a(1e-5, 1e-5, C);//de_a(1e-5, 1e-5, C);

uint * optimiser_tous_les = UNIFORME_C(1);

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	pourcent_masque[C-1] = 0.0;
	alpha[0] = 1e-2;

	/*	
	/!\ /!\ /!\	/!\ /!\ /!\	/!\ /!\ /!\	/!\ /!\ /!\	/!\ /!\ /!\
	- Cloner edt.uvsq.fr et faire son emploie du temps
ou page statique sur telephone
	- Entree bruit
	- sans ou avec regression normalisation
	- !!! Pouvoire verifier gain usd de validation sans internet et tester_mdl.py
		/!\ /!\ /!\	/!\ /!\ /!\	/!\ /!\ /!\	/!\ /!\ /!\	/!\ /!\ /!\
	- ajouter gain_usd pour la selection de la validation
	*/

	//// - Score gain USD 24H (MEGA_T toujours 12h ou 24h)
	//// - Chiffre "Par le haut", "Par le bas". Tres precis. 90%=0.5, 99%=0.8
	//// - Tout est en MEGA_T. GRAND_T est uniquement pour le parrallelisme.
	//// Mega_t pour les score en chaine u+=u*f(x)*(p1/p0-1)
	//// - Filtres fausses erreures x0.1

	//pourcent_masque_nulle[0] = 0.00;//0.20
	//pourcent_masque[0] = 0.01;

	//	----- Lien constants ------

	/*
		* Attention Mechanisme (softmax)
		* f(ax+bx+c)
		* f(ax+b) * f(ax+b) + f(ax+b)
		* f(ax+b) * g(ax+b) + f(ax+b)
		* x + f(ax+b)
		* f(f(f(f(x,mem),mem),mem),mem) un gros reseau mem qui est une reflexion
	*/
	
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");   charger_tout();

	//	-- Verification --
	titre("Verifier MDL");     verif_mdl_1e5();

	//===============
	titre("  Programme Generale  ");
	ecrire_structure_generale("structure_generale.bin");

	//visualiser();

	/*uint     Y[C];
	uint insts[C];
	//
	uint st[C][2] = {
	//	{4096, DOT1D},
	//
	//    Y  ,   inst
		{2048, FILTRES_PRIXS},
		{128,  DOT1D_TANH},
		//
		{512,  DOT1D_TANH_ELMAN},
		{128,  DOT1D_TANH},
		{256,  DOT1D_TANH},
		{256,  DOT1D_TANH},
		{256,  DOT1D_TANH},
		//
		{1,    DOT1D_TANH},
	};
	FOR(0, i, C) {
		    Y[i] = st[i][0];
		insts[i] = st[i][1];
	}
	//
	//	Assurances :
	ema_int_t * bloque[BLOQUES] = {
	//			    Source,      Nature,  K_ema, Intervalle,     {params}
	// -------
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, DIRECT, 512, 256.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_HIGH_BTC, DIRECT, 512, 256.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_LOW_BTC, DIRECT, 512, 256.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_BTC, DIRECT, 512, 256.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_A_BTC, DIRECT, 512, 256.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 1, 1, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 1, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 2, 1.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 2, 2, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 2, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 4, 2.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 4, 4, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 8, 4.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 8, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 16, 8.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 16, 16, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 16, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 32, 16.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 32, 32, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 32, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 64, 32.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 64, 64, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 64, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 128, 64.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 128, 128, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 128, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 256, 128.0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 256, 256, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES_U_BTC, DIRECT, 512, 256.0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS_BTC, MACD, 1, 1, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 1, 2, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 1, 2, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 1, 8, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 1, 8, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 1, 8, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 1.0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 2, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 2, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 4, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 4, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 4, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 16, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 16, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 16, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 2, 16, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 2.0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 2.0, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 4, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 4, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 4, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 8, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 8, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 8, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 32, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 32, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 32, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 4, 32, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 4.0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 4.0, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 4.0, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 8, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 8, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 8, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 16, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 16, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 16, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 16, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 64, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 64, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 64, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 8, 64, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 8.0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 8.0, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 8.0, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 8.0, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 16, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 16, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 16, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 16, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 32, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 32, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 32, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 32, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 128, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 128, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 128, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 16, 128, cree_MACD(128)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 16.0, cree_MACD(2)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 16.0, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 16.0, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 16.0, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 32, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 32, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 32, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 32, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 64, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 64, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 64, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 64, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 256, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 256, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 256, cree_MACD(128)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 32, 256, cree_MACD(256)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 32.0, cree_MACD(4)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 32.0, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 32.0, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 32.0, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 64, cree_MACD(8)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 64, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 64, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 64, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 128, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 128, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 128, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 64, 128, cree_MACD(128)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 128.0, cree_MACD(16)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 128.0, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 128.0, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 128.0, cree_MACD(128)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 256, cree_MACD(32)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 256, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 256, cree_MACD(128)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 256, cree_MACD(256)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 256, cree_MACD(64)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 256, cree_MACD(128)),
		cree_ligne(SRC_PRIXS_BTC, MACD, 256, 256, cree_MACD(256)),
	};
	//
	system("rm mdl");system("rm mdl_validation");
	Mdl_t * mdl = cree_mdl(GRAND_T, Y, insts, bloque);*/

	Mdl_t * mdl = ouvrire_mdl(GRAND_T, "mdl.bin");

	//mdl_re_cree_poids(mdl);

	//uint c=5, nouveau_Y=64;
	//mdl_changer_couche_Y(mdl, c, nouveau_Y);

	enregistrer_les_lignes_brute(mdl, "lignes_brute.bin");

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = FIN;
	//
	plume_pred(mdl, t0, t1);
	//comportement(mdl, t0, t0+GRAND_T);
	//
	srand(time(NULL));
#define PERTURBATIONS 0
	//
	printf("Parametres : alpha[0]=%e GRAND_T=%i P_S=%f\n", alpha[0], GRAND_T, P_S);
	//
	uint MODE_t_MODE  = 0;
	uint grain_t_MODE = 0;
	//
	uint rep = 0;
	while ( 1 ) {
		//
		//MODE_t_MODE = t_CONTINUE;
		MODE_t_MODE = t_PSEUDO_ALEA;
		//
		grain_t_MODE = rand() % 10000;
		//
		/*mdl_poids_gpu_vers_cpu(mdl);
		FOR(0, c, C) regulariser_inst[mdl->insts[c]](mdl, c);
		mdl_poids_cpu_vers_gpu(mdl);*/
		//
		optimisation_mini_packet(
			mdl,
			t0, t1,
			alpha, 1.0,
			ADAM, 50,
			//
			pourcent_masque,
			//pourcent_masque_nulle,
			//
			//pourcent_masque_opti,
			pourcent_masque_opti_nulle,
			//
			PERTURBATIONS,
			optimiser_tous_les,
			MODE_t_MODE, grain_t_MODE);
		//
		mdl_poids_gpu_vers_cpu(mdl);
		//
		ecrire_mdl(mdl, "mdl.bin");
		//
		enregistrer_par_validation(mdl);
		//
		if (rep % 10 == 0) plume_pred(mdl, t0, t1);
		//
		printf("===================================================\n");
		printf("================= TERMINE %i ======================\n", rep++);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};