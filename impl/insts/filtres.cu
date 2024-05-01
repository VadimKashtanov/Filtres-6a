#include "filtres_prixs.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_filtres_prixs(Mdl_t * mdl, uint c)
{
	mdl->inst_POIDS        [c] = BLOQUES*F_PAR_BLOQUES*N;
	mdl->inst_VARS         [c] = mdl->Y[c];
	mdl->inst_LOCDS        [c] = 2*mdl->Y[c];
	mdl->inst_SORTIES      [c] = mdl->Y[c];
	mdl->inst_DEPART_SORTIE[c] = mdl->Y[c] - mdl->Y[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);
	FOR(0, i, mdl->inst_POIDS[c])
		mdl->p[c][i] = (2*rnd()-1) * 1.0;
	//
	mdl->inst_p_separateurs[c] = 0;
};

void plume_filtres_prixs(Mdl_t * mdl, uint c)
{
	printf("POIDS FILTRES: \n");
	FOR(0, b, BLOQUES) {
		FOR(0, f, F_PAR_BLOQUES) {
			printf("bloque=%i f=%i :", b, f);
			FOR(0, i, N)
				printf("%+f, ", mdl->p[c][b*F_PAR_BLOQUES*N + f*N + i]);
			printf("\n");
		}
	}
};

//	=========================================================

void regulariser_filtres_prixs(Mdl_t * mdl, uint c) {

};

//	=========================================================

void f_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t) {
	uint depart = t0;
	uint X_vars=0, Y_vars=mdl->inst_VARS[inst];
	//
	if (mode == MODE_NAIF) {
		nvidia_filtres_prixs___naive(
			mega_t,
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			depart, mdl->T,
			BLOQUES, F_PAR_BLOQUES,
			mdl->normalisee__d, mdl->dif_normalisee__d,
			mdl->p__d[inst],
			mdl->y__d[inst],
			mdl->l__d[inst]);
	} else if (mode == MODE_MAXIMALE) {
		nvidia_filtres_prixs___shared(
			mega_t,
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			depart, mdl->T,
			BLOQUES, F_PAR_BLOQUES,
			mdl->normalisee__d, mdl->dif_normalisee__d,
			mdl->p__d[inst],
			mdl->y__d[inst],
			mdl->l__d[inst]);
	} else {
		ERR("Pas de mode %i pour mes filtres", mode);
	};
};

void df_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t) {
	uint depart = t0;
	uint X_vars=0, Y_vars=mdl->inst_VARS[inst];
	//
	if (mode == MODE_NAIF) {
		d_nvidia_filtres_prixs___naive(
			mega_t,
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			depart, mdl->T,
			BLOQUES, F_PAR_BLOQUES,
			mdl->normalisee__d, mdl->dif_normalisee__d,
			mdl->p__d[inst],
			mdl->y__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dp__d[inst]);
	} else if (mode == MODE_MAXIMALE) {
		d_nvidia_filtres_prixs___shared(
			mega_t,
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			depart, mdl->T,
			BLOQUES, F_PAR_BLOQUES,
			mdl->normalisee__d, mdl->dif_normalisee__d,
			mdl->p__d[inst],
			mdl->y__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dp__d[inst]);
	} else {
		ERR("Pas de mode %i pour mes filtres", mode);
	}
};