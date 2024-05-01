#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void plume_separateur_p(Mdl_t * mdl, uint c, uint p) {
	FOR(0, i, mdl->inst_p_separateurs[c]) {
		if (p == mdl->depart_inst_p_separateurs[c][i]) {
			printf(" = \033[93m%s\033[0m =\n", mdl->char_inst_p_separateurs[c][i]);
		}
	}
};

void mdl_plume_poids(Mdl_t * mdl) {
	FOR(0, c, C) {
		printf("=========== C= %i ==========\n", c);
		plume_inst[mdl->insts[c]](mdl, c);
	}
};

void plumer_mdl(Mdl_t * mdl) {
	printf("\033[4m Plume mdl\033[0m\n");
	printf("Filtres : %i\n", mdl->Y[0]);
	printf("Lignes des bloques (pour filtres_prixs): \n");
	FOR(0, i, BLOQUES) {
		printf("%3.i| %s  source=%s intervalle=%3.i K_ema=%3.i ",
			i,
			nom_natures[mdl->bloque[i]->nature],
			nom_sources[mdl->bloque[i]->source],
			mdl->intervalles[i],
			mdl->bloque[i]->K_ema
		);
		FOR(0, j, MAX_PARAMS) printf("P%i=%3.i ", j, mdl->bloque[i]->params[j]);
		printf("\n");
	}
	printf("Instructions : \n");
	uint POIDS = 0;
	FOR(0, c, C) {
		POIDS += mdl->inst_POIDS[c];
		printf("%2.i| %s:<Y=%4.i> [poids=%7.i]\n", c, nom_inst[mdl->insts[c]], mdl->Y[c], mdl->inst_POIDS[c]);
	}
	printf("Quantitée poids = %s\n", scientifique(POIDS));
	printf(" --- fin plume mdl ---\n");
};

void comportement(Mdl_t * mdl, uint t0, uint t1) {
	uint MODE_t_MODE = t_CONTINUE;
	uint grain_t_MODE = 0;
	//
	mdl_f(mdl, t0, t1, MODE_MAXIMALE, MODE_t_MODE, grain_t_MODE);
	mdl_gpu_vers_cpu(mdl);
	FOR(0, c, C) {
		float * y = gpu_vers_cpu<float>(mdl->y__d[c], mdl->inst_VARS[c]*mdl->T*MEGA_T);
		printf("###=== c=%i ===###\n", c);
		FOR(0, i, mdl->inst_VARS[c]) {
			printf("%3.i| ", i);
			//FOR(t0, t, t1)
			uint t = t0;
			FOR(0, mega_t, MEGA_T)
				printf("%+f ", y[(mega_t*mdl->T + t-t0)*mdl->inst_VARS[c] + i]);
			printf("\n");
		}
		free(y);
	}
};

static uint * zeroiser_nulle = UNIFORME_C(1);

void mdl_plume_grad(Mdl_t * mdl, uint t0, uint t1, uint _t_MODE, uint GRAINE) {
	//uint * zeroiser_nulle = UNIFORME_C(1);
	//
	mdl_zero_deriv_gpu(mdl, zeroiser_nulle);
	//
	mdl_aller_retour(mdl, t0, t1, MODE_MAXIMALE, _t_MODE, GRAINE);
	//
	FOR(0, c, C) {
		printf(" Couche %2.i \033[93mY=%5.i poids=%7.i\033[0m", c, mdl->Y[c], mdl->inst_POIDS[c]);
		uint POIDS = mdl->inst_POIDS[c];
		float * _grad = gpu_vers_cpu<float>(mdl->dp__d[c], POIDS);
		float * ____p = gpu_vers_cpu<float>(mdl->p__d[c], POIDS);
		//
		//	Grad
		//
		float moyenne = 0;
		float min=_grad[0], max=_grad[0];
		float abs_min=fabs(____p[0]);
		FOR(0, i, POIDS) {
			if (_grad[i] > max) max = _grad[i];
			if (_grad[i] < min) min = _grad[i];
			moyenne += fabs(_grad[i]);
			if (fabs(_grad[i]) < abs_min) abs_min = fabs(_grad[i]);
		}
		printf("[Grad:Min=%+f;Max=%+f;Moyenne=(-/+)%f]",
			min, max, moyenne / POIDS
		);
		free(_grad);
		//
		//	P
		//
		moyenne = 0;
		min=____p[0], max=____p[0];
		abs_min=fabs(____p[0]);
		FOR(0, i, POIDS) {
			if (____p[i] > max) max = ____p[i];
			if (____p[i] < min) min = ____p[i];
			moyenne += fabs(____p[i]);
			if (fabs(____p[i]) < abs_min) abs_min = fabs(____p[i]);
		}
		printf("[P:Min=%+f;Max=%+f;Moyenne=(-/+)%f]",
			min, max, moyenne / POIDS
		);
		free(____p);
		printf("  %s\n", nom_inst[mdl->insts[c]]);
	}
};