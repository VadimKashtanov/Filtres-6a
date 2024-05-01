#include "dot1d_tanh.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_dot1d_tanh(Mdl_t * mdl, uint c)
{
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	mdl->inst_POIDS        [c] = (mdl->Y[c-1]+1)*mdl->Y[c];
	mdl->inst_VARS         [c] = mdl->Y[c];
	mdl->inst_LOCDS        [c] = mdl->Y[c];
	mdl->inst_SORTIES      [c] = mdl->Y[c];
	mdl->inst_DEPART_SORTIE[c] = mdl->Y[c] - mdl->Y[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);

	FOR(0, y, Y) {
		FOR(0, x, X+1) {
			mdl->p[c][y*(X+1)+x] = (2*rnd()-1) * sqrtf(/*10.0*/ 8.0 / (X+Y));
		}
	}
	//
	mdl->inst_p_separateurs[c] = 0;
};

void plume_dot1d_tanh(Mdl_t * mdl, uint c)
{
	printf("POIDS dot1d_tanh: \n");
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	FOR(0, y, Y) {
		printf("y=%i : ", y);
		FOR(0, x, X) {
			printf("%+f,", mdl->p[c][y*(X+1)+x]);
		}
		printf(" biais=%+f\n", mdl->p[c][y*(X+1)+X+1-1]);
	}
};

//	=========================================================

void regulariser_dot1d_tanh(Mdl_t * mdl, uint c) {
	/*uint X=mdl->Y[c-1], Y=mdl->Y[c];
	//
	//float na = somme_absolue_horizontale<float>(mdl->p);
	float na;
	FOR(0, y, Y) {
		na = 0;
		FOR(0, x, X+1) na += fabs(mdl->p[c][y*(X+1) + x]);
		float je_veux = 15.0;
		float coef = je_veux / na;
		FOR(0, x, X+1) mdl->p[c][y*(X+1) + x] *= coef;
	}*/
};

//	=========================================================

void f_dot1d_tanh(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	//
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == MODE_NAIF) {
		nvidia_dot1d_tanh_naive(
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
		nvidia_dot1d_tanh_shared_2_16(
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

void df_dot1d_tanh(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	//
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	//
	if (mode == MODE_NAIF) {
		d_nvidia_dot1d_tanh_naive(
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
		d_nvidia_dot1d_tanh_shared_2_16(
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