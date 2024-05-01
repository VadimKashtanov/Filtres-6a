#include "lstm1d_peephole.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_lstm1d_peephole(Mdl_t * mdl, uint c)
{
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	//
	mdl->inst_POIDS        [c] = lstm_POIDS(X,Y);
	mdl->inst_VARS         [c] = lstm_VARS(X,Y);
	mdl->inst_LOCDS        [c] = lstm_LOCDS(X,Y);
	mdl->inst_SORTIES      [c] = Y;
	mdl->inst_DEPART_SORTIE[c] = mdl->inst_VARS[c] - mdl->inst_SORTIES[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);
	//
	FOR(0, i, mdl->inst_POIDS[c]) {
		mdl->p[c][i] = (2*rnd()-1) * sqrtf(/*10.0*/ 6.0 / (X+Y));
	}
	//
	mdl->inst_p_separateurs[c] = 15;
	mdl->char_inst_p_separateurs[c] = alloc<char*>(mdl->inst_p_separateurs[c]);
	mdl->depart_inst_p_separateurs[c] = alloc<uint>(mdl->inst_p_separateurs[c]);
	//
	//	F
	mdl->char_inst_p_separateurs  [c][0] = "Fx";
	mdl->depart_inst_p_separateurs[c][0] = depart_poids_f(X,Y)+0;
	mdl->char_inst_p_separateurs  [c][1] = "Fh";
	mdl->depart_inst_p_separateurs[c][1] = depart_poids_f(X,Y)+Fx(X,Y);
	mdl->char_inst_p_separateurs  [c][2] = "Fc";
	mdl->depart_inst_p_separateurs[c][2] = depart_poids_f(X,Y)+Fx(X,Y)+Fh(X,Y);
	mdl->char_inst_p_separateurs  [c][3] = "Fb";
	mdl->depart_inst_p_separateurs[c][3] = depart_poids_f(X,Y)+Fx(X,Y)+Fh(X,Y)+Fc(X,Y);
	//	I
	mdl->char_inst_p_separateurs  [c][4] = "Ix";
	mdl->depart_inst_p_separateurs[c][4] = depart_poids_i(X,Y)+0;
	mdl->char_inst_p_separateurs  [c][5] = "Ih";
	mdl->depart_inst_p_separateurs[c][5] = depart_poids_i(X,Y)+Ix(X,Y);
	mdl->char_inst_p_separateurs  [c][6] = "Ic";
	mdl->depart_inst_p_separateurs[c][6] = depart_poids_i(X,Y)+Ix(X,Y)+Ih(X,Y);
	mdl->char_inst_p_separateurs  [c][7] = "Ib";
	mdl->depart_inst_p_separateurs[c][7] = depart_poids_i(X,Y)+Ix(X,Y)+Ih(X,Y)+Ic(X,Y);
	//	U
	mdl->char_inst_p_separateurs  [c][8] = "Ux";
	mdl->depart_inst_p_separateurs[c][8] = depart_poids_u(X,Y)+0;
	mdl->char_inst_p_separateurs  [c][9] = "Uh";
	mdl->depart_inst_p_separateurs[c][9] = depart_poids_u(X,Y)+Ux(X,Y);
	mdl->char_inst_p_separateurs  [c][10] = "Ub";
	mdl->depart_inst_p_separateurs[c][10] = depart_poids_u(X,Y)+Ux(X,Y)+Uh(X,Y);
	//	O
	mdl->char_inst_p_separateurs  [c][11] = "Ox";
	mdl->depart_inst_p_separateurs[c][11] = depart_poids_o(X,Y)+0;
	mdl->char_inst_p_separateurs  [c][12] = "Oh";
	mdl->depart_inst_p_separateurs[c][12] = depart_poids_o(X,Y)+Ix(X,Y);
	mdl->char_inst_p_separateurs  [c][13] = "Oc";
	mdl->depart_inst_p_separateurs[c][13] = depart_poids_o(X,Y)+Ix(X,Y)+Ih(X,Y);
	mdl->char_inst_p_separateurs  [c][14] = "Ob";
	mdl->depart_inst_p_separateurs[c][14] = depart_poids_o(X,Y)+Ix(X,Y)+Ih(X,Y)+Ic(X,Y);
};

void plume_lstm1d_peephole(Mdl_t * mdl, uint c)
{
	printf("POIDS lstm1d_peephole: \n");
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
};

//	=========================================================

static void equilibrer_vecteur(float * p, uint X) {
	float na = 0;
	FOR(0, x, X) na += fabs(p[x]);
	float je_veux = 15.0;
	float coef = je_veux / na;
	FOR(0, x, X) p[x] *= coef;
} 

void regulariser_lstm1d_peephole(Mdl_t * mdl, uint c) {
	/*uint X=mdl->Y[c-1], Y=mdl->Y[c];
	//
	FOR(0, y, Y) {	//Pas les Biais
		//	f
		equilibrer_vecteur(mdl->p[c] + depart_poids_f(X,Y)+y*X,                 X);
		equilibrer_vecteur(mdl->p[c] + depart_poids_f(X,Y)+Fx(X,Y)+y*Y,         Y);
		equilibrer_vecteur(mdl->p[c] + depart_poids_f(X,Y)+Fx(X,Y)+Fh(X,Y)+y*Y, Y);
		//	i
		equilibrer_vecteur(mdl->p[c] + depart_poids_i(X,Y)+y*X,                 X);
		equilibrer_vecteur(mdl->p[c] + depart_poids_i(X,Y)+Ix(X,Y)+y*Y,         Y);
		equilibrer_vecteur(mdl->p[c] + depart_poids_i(X,Y)+Ix(X,Y)+Ih(X,Y)+y*Y, Y);
		//	u
		equilibrer_vecteur(mdl->p[c] + depart_poids_u(X,Y)+y*X,                 X);
		equilibrer_vecteur(mdl->p[c] + depart_poids_u(X,Y)+Ux(X,Y)+y*Y,         Y);
		//	o
		equilibrer_vecteur(mdl->p[c] + depart_poids_o(X,Y)+y*X,                 X);
		equilibrer_vecteur(mdl->p[c] + depart_poids_o(X,Y)+Ox(X,Y)+y*Y,         Y);
		equilibrer_vecteur(mdl->p[c] + depart_poids_o(X,Y)+Ox(X,Y)+Oh(X,Y)+y*Y, Y);
	};*/
};

//	=========================================================

void f_lstm1d_peephole(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	//
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	//
	if (mode == MODE_NAIF) {
		nvidia_lstm1d_peephole_naive(
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
		nvidia_lstm1d_peephole_shared_16_2(
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

void df_lstm1d_peephole(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	//
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	//
	if (mode == MODE_NAIF) {
		d_nvidia_lstm1d_peephole_naive(
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
	} else if (mode = MODE_MAXIMALE) {
		d_nvidia_lstm1d_peephole_shared_16_2(
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