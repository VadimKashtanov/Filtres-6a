#include "statistiques.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

Stats_t * statistiques(Mdl_t * mdl, uint t0, uint t1) {
	Stats_t * stats = alloc<Stats_t>(1);
	//
	uint _t_MODE = t_CONTINUE;
	uint grain_t_MODE = 0;
	//
	t1 -= ((t1-t0) % (mdl->T*MEGA_T));
	//
	uint fois = (t1-t0)/(mdl->T*MEGA_T);
	//
	ASSERT(fois != 0);
	//
	stats->score = 0;
	stats->pred  = 0;
	stats->les_gains__2 = 0; float coef__2 = 2.0;
	stats->les_gains__4 = 0; float coef__4 = 4.0;
	stats->les_gains__8 = 0; float coef__8 = 8.0;
	//
	FOR(0, i, fois) {
		float _t0 = t0 + (i+0)*mdl->T*MEGA_T;
		float _t1 = t0 + (i+1)*mdl->T*MEGA_T;
		//
		stats->score += mdl_score(mdl, _t0, _t1, MODE_MAXIMALE, _t_MODE, grain_t_MODE);
		stats->pred  += mdl_pred (mdl, _t0, _t1, MODE_MAXIMALE, _t_MODE, grain_t_MODE);
		//
		stats->les_gains__2 += mdl_les_gains(mdl, _t0, _t1, MODE_MAXIMALE, coef__2, _t_MODE, grain_t_MODE);
		stats->les_gains__4 += mdl_les_gains(mdl, _t0, _t1, MODE_MAXIMALE, coef__4, _t_MODE, grain_t_MODE);
		stats->les_gains__8 += mdl_les_gains(mdl, _t0, _t1, MODE_MAXIMALE, coef__8, _t_MODE, grain_t_MODE);
	}
	//
	stats->score        /= (float)(fois);
	stats->pred         /= (float)(fois);
	stats->les_gains__2 /= (float)(fois);
	stats->les_gains__4 /= (float)(fois);
	stats->les_gains__8 /= (float)(fois);
	//
	return stats;
};