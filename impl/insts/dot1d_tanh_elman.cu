#include "dot1d_tanh_elman.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_dot1d_tanh_elman(Mdl_t * mdl, uint c)
{
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	mdl->inst_POIDS        [c] = (X*Y + Y*Y + Y);
	mdl->inst_VARS         [c] = Y;
	mdl->inst_LOCDS        [c] = Y;
	mdl->inst_SORTIES      [c] = Y;
	mdl->inst_DEPART_SORTIE[c] = mdl->inst_VARS[c] - mdl->Y[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);

	FOR(0, i, mdl->inst_POIDS[c]) {
		mdl->p[c][i] = (2*rnd()-1) * sqrtf(/*10.0*/ 8.0 / (X+Y));
	}
	//
	mdl->inst_p_separateurs[c] = 0;
};

void plume_dot1d_tanh_elman(Mdl_t * mdl, uint c)
{
	printf("POIDS dot1d_tanh: \n");
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	TODO();
};

//	=========================================================

void regulariser_dot1d_tanh_elman(Mdl_t * mdl, uint c) {

};

//	=========================================================

void f_dot1d_tanh_elman(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	//
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == MODE_NAIF) {
		nvidia_dot1d_tanh_elman_naive(
			mega_t,
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, mdl->T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else if (mode == MODE_MAXIMALE) {
		nvidia_dot1d_tanh_elman_shared_2_16(
			mega_t,
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, mdl->T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}

//	----------------------------

void df_dot1d_tanh_elman(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	//
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	//
	if (mode == MODE_NAIF) {
		d_nvidia_dot1d_tanh_elman_naive(
			mega_t,
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, mdl->T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else if (mode == MODE_MAXIMALE) {
		d_nvidia_dot1d_tanh_elman_shared_2_16(
			mega_t,
			_t_MODE, GRAINE,
			X_vars, Y_vars,
			X, Y,
			depart, mdl->T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else {
		ERR("Pas de mode %i pour cuda df(x)", mode);
	}
}