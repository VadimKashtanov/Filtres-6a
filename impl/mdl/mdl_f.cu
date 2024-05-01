#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void mdl_f(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE) {
	FOR(0, mega_t, MEGA_T) {
		FOR(0, c, C) {
			inst_f[mdl->insts[c]](mdl, c, mode, t0, t1, _t_MODE, GRAINE, mega_t);
		};
	};
};