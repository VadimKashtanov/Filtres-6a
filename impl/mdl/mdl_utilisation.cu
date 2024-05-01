#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float mdl_les_gains(Mdl_t * mdl, uint t0, uint t1, uint mode, float GRAND_COEF, uint _t_MODE, uint GRAINE) {
	ASSERT(GRAND_COEF >= 2);
	//
	float * _y = gpu_vers_cpu<float>(mdl->y__d[C-1], mdl->T*MEGA_T*1);
	//
	float somme     = 0;
	float potentiel = 0;
	//
	FOR(0, t, mdl->T) {
		FOR(0, mega_t, MEGA_T) {
			uint depart_plus_t = t_MODE(
				_t_MODE, GRAINE,
				t0, t0+mdl->T*MEGA_T,
				t, mega_t,
				mdl->T, MEGA_T
			);
			//
			float p1p0 = (MARCHEE_DE_TRADE[depart_plus_t+1]/MARCHEE_DE_TRADE[depart_plus_t]-1);
			//
			uint a_t_il_predit = (signe(p1p0) == signe(_y[(mega_t*mdl->T + t)*1 + 0]));
			//
			somme     += powf(fabs(p1p0),GRAND_COEF) * a_t_il_predit;
			potentiel += powf(fabs(p1p0),GRAND_COEF) * true         ;
		}
	}
	free(_y);
	return somme / potentiel;
};

float mdl_score(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE) {
	mdl_zero_gpu(mdl);
	//
	mdl_f(mdl, t0, t1, mode, _t_MODE, GRAINE);
	//
	float somme_score = nvidia_somme_score(mdl->y__d[C-1], t0, mdl->T, _t_MODE, GRAINE);
	//
	return nvidia_score_finale(somme_score, mdl->T, _t_MODE, GRAINE);
};

float mdl_pred(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE) {
	mdl_zero_gpu(mdl);
	//
	mdl_f(mdl, t0, t1, mode, _t_MODE, GRAINE);
	return nvidia_prediction(mdl->y__d[C-1], t0, mdl->T, _t_MODE, GRAINE);
};

void mdl_aller_retour(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE) {
	mdl_zero_gpu(mdl);
	mdl_f(mdl, t0, t1, mode, _t_MODE, GRAINE);
	//
	//
	float somme_score = nvidia_somme_score(mdl->y__d[C-1], t0, mdl->T, _t_MODE, GRAINE);
	//
	//
	float d_score = d_nvidia_score_finale(somme_score, mdl->T, _t_MODE, GRAINE);
	//
	//
	d_nvidia_somme_score(d_score, mdl->y__d[C-1], mdl->dy__d[C-1], t0, mdl->T, _t_MODE, GRAINE);
	mdl_df(mdl, t0, t1, mode, _t_MODE, GRAINE);
};