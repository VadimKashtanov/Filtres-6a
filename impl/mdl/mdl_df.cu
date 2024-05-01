#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

//	===================================================================

void mdl_df(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE) {
	RETRO_FOR(0, mega_t, MEGA_T) {
		RETRO_FOR(0, c, C) {
			inst_df[mdl->insts[c]](mdl, c, mode, t0, t1, _t_MODE, GRAINE, mega_t);
		};
	}
};