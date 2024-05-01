#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

static uint * tout_zeroiser = UNIFORME_C(1);

PAS_OPTIMISER()
uint __SI_NAN(Mdl_t * mdl)
{
	float * _gpu = gpu_vers_cpu<float>(mdl->p__d[C-1], mdl->inst_POIDS[C-1]);
	//
	uint vrai = 0;
	FOR(0, i, mdl->inst_POIDS[C-1]) {
		if (_gpu[i] != _gpu[i]) {
			vrai = 1;
			break;
		}
	}
	//
	free(_gpu);
	//
	return vrai;
};

PAS_OPTIMISER()
void __interne_optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	uint ** masque,
	uint ** masque_opti,
	uint PERTURBATIONS,
	uint zero_accumulation_tous_les[C],
	uint _t_MODE, uint GRAINE)
{
	mdl_zero_deriv_gpu(mdl, tout_zeroiser);
	//
	//	Cree les listes pour les `hist` si un opti en a besoin 
	Opti_classe_t opti_classe;
	if      (methode == SGD)     opti_classe.sgd     = (uint)NULL;
	else if (methode == RMSPROP) opti_classe.rmsprop = cree_rmsprop(mdl);
	else if (methode == ADAM)    opti_classe.adam    = cree_adam(mdl);
	else ERR("Pas de methode %i d'optimisation", methode);
	
	//	Plumer grad pour mieux y voire
	mdl_plume_grad(mdl, t0, t1, _t_MODE, GRAINE);
	
	/* ------- Optimisation ----------- */
	uint zeroiser[C];
	FOR(0, i, I) {
		//
		FOR(0, j, C) {
			if (i % zero_accumulation_tous_les[j] == 0)
				zeroiser[j] = 1;
			else
				zeroiser[j] = 0;
		}
		// !!! i=0 ne sera pas optimisé !!!
		//if (i != 0)	//on optimise pas au premiere, pour voire le res avant opti
		//{
		//perturber(mdl, PERTURBATIONS);
		mdl_aller_retour(mdl, t0, t1, MODE_MAXIMALE, _t_MODE, GRAINE);
		//}

		if (__SI_NAN(mdl)) ERR("Des nan dans mdl");
		
		//	--------- * Optimisation * -------------
#define optimiser_la_couche zeroiser
		if (i != 0) {
			if (methode == SGD)     opti_simple (zero_accumulation_tous_les, optimiser_la_couche, mdl, alpha, div, masque, masque_opti);
			if (methode == RMSPROP) opti_rmsprop(zero_accumulation_tous_les, optimiser_la_couche, mdl, opti_classe.rmsprop, alpha, div, masque, masque_opti);
			if (methode == ADAM)    opti_adam   (zero_accumulation_tous_les, optimiser_la_couche, mdl, opti_classe.adam,    alpha, div, masque, masque_opti);
		}
		//
		mdl_zero_deriv_gpu(mdl, zeroiser);
		//
		mdl_normer_les_filtres(mdl);
		//
		/*	mdl_poids_gpu_vers_cpu(mdl);
		FOR(0, c, C) regulariser_inst[mdl->insts[c]](mdl, c);
			mdl_poids_cpu_vers_gpu(mdl);*/
		//
		if (i % /*100*/50 == 0 || i == 1 || i == I-1) {
			//mdl_plume_grad(mdl, t0, t1, _t_MODE, GRAINE);
			//
			float __pred = mdl_pred (mdl, t0, t1, MODE_MAXIMALE, _t_MODE, GRAINE);
			float _score = mdl_score(mdl, t0, t1, MODE_MAXIMALE, _t_MODE, GRAINE);
			//
			float les_gains__2 = mdl_les_gains(mdl, t0, t1, MODE_MAXIMALE,   2.0, _t_MODE, GRAINE);
			float les_gains__4 = mdl_les_gains(mdl, t0, t1, MODE_MAXIMALE,   4.0, _t_MODE, GRAINE);
			//
			printf("%3.i/%3.i| perf=%f%%", i, I, 100*__pred);
			printf(" score=\033[93m%+f\033[0m (%%.potentiel^2=%+f, %%.potentiel^4=%+f)\n",
				_score,
				les_gains__2,
				les_gains__4
			);
			if (fabs(_score) < 0.00001) {
				printf("Score < 0.00001 => Fin d'optimisation\n");
				break;
			}
		}
	}

	//	Liberer
	if      (methode == SGD)     opti_classe.sgd = 0;
	else if (methode == RMSPROP) liberer_rmsprop(opti_classe.rmsprop);
	else if (methode == ADAM)    liberer_adam   (opti_classe.adam   );
};

void optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	float * pourcent_masque,
	float * pourcent_opti_masque,
	uint PERTURBATIONS,
	uint zero_accumulation_tous_les[C],
	uint _t_MODE, uint GRAINE)
{
	Masque_t * masque = cree_masque(mdl, pourcent_masque, pourcent_opti_masque);
	//
	__interne_optimiser(
		mdl,
		t0, t1,
		alpha, div,
		methode, I,
		masque->masque,
		masque->masque_opti,
		PERTURBATIONS,
		zero_accumulation_tous_les,
		_t_MODE, GRAINE);
	//
	sortire_masque(mdl, masque);
};