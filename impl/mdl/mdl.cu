#include "mdl.cuh"

#include "filtres_prixs.cuh"
#include "dot1d_tanh.cuh"
#include "lstm1d_peephole.cuh"
#include "dot1d_tanh_elman.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

mdl_inst_f cree_inst[INSTS] = {
	cree_filtres_prixs,
	cree_dot1d_tanh,
	cree_lstm1d_peephole,
	cree_dot1d_tanh_elman
};

mdl_f_f inst_f [INSTS] = {
	f_filtres_prixs,
	f_dot1d_tanh,
	f_lstm1d_peephole,
	f_dot1d_tanh_elman
};

mdl_f_f inst_df[INSTS] = {
	df_filtres_prixs,
	df_dot1d_tanh,
	df_lstm1d_peephole,
	df_dot1d_tanh_elman
};

char * nom_inst[INSTS] = {
	"filtres_prixs      ",
	"dot1d tanh(x)      ",
	"lstmd1d_peephole   ",
	"dot1d tanh(x) elman"
};

mdl_inst_f plume_inst[INSTS] = {
	plume_filtres_prixs,
	plume_dot1d_tanh,
	plume_lstm1d_peephole,
	plume_dot1d_tanh_elman
};

mdl_inst_f regulariser_inst[INSTS] = {
	regulariser_filtres_prixs,
	regulariser_dot1d_tanh,
	regulariser_lstm1d_peephole,
	regulariser_dot1d_tanh_elman
};

uint * UNIFORME_C(uint x) {
	uint * ret = alloc<uint>(C);
	FOR(0, i, C) ret[i] = x;
	return ret;
};

static void calculer_normalisee__et__dif_normalisee(Mdl_t * mdl) { 
	FOR(0, b, BLOQUES) {
		FOR(DEPART, t, FIN) {
			//	_max & _min pour ce filtre-8
			float _max, _min;
			
			if (mdl->type_de_norme[b] == NORME_CLASSIQUE) {
				_max = mdl->bloque[b]->brute[t - 0*mdl->bloque[b]->intervalle];
				_min = mdl->bloque[b]->brute[t - 0*mdl->bloque[b]->intervalle];
				FOR(1, i, N_FLTR) {
					float xi = mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle];
					if (_max < xi)
						_max = xi;
					if (_min > xi)
						_min = xi;
				}


			} else if (mdl->type_de_norme[b] == NORME_RELATIVE) {
				_max = fabs(mdl->bloque[b]->brute[t - 0*mdl->bloque[b]->intervalle]);
				FOR(1, i, N_FLTR) {
					float xi = fabs(mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle]);
					if (_max < xi)
						_max = xi;
				}
				_max = +_max;
				_min = -_max;

			} else if (mdl->type_de_norme[b] == NORME_THEORIQUE) {
				_min = /*1.0;//*/mdl->min_theorique[b];
				_max = /*0.0;//*/mdl->max_theorique[b];
				FOR(0, i, N_FLTR) {
					float xi = mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle];
					if (!(_min <= xi && xi <= _max))
						printf("%f  %f %f\n", xi, _min, _max);
					ASSERT(_min <= xi && xi <= _max);
				}

			} else {
				ERR("Norme ni classique, ni theorique, mdl->type_de_norme[b] == %i", mdl->type_de_norme[b]);
			};

			if (_min == _max) {
				FOR(0, i, N_FLTR) printf("%f\n", mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle]);

				ERR("_mi==_max (%f!=%f) b=%i type=%i", _min, _max, b, mdl->type_de_norme[b]);
				//ASSERT(_min != _max);
			}

			//
			FOR(0, i, N_FLTR) {
				mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] = ( mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle] - _min)/( _max - _min );
				//if (mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i]==0) printf("%i %i %i\n", b,t,i);
			}
		};

		//#pragma omp parallel
		//#pragma omp for
		FOR(DEPART, t, FIN) {
			FOR(1, i, N_FLTR)
				mdl->dif_normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] = mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] - mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i-1];
			mdl->dif_normalisee[b*PRIXS*N_FLTR+t*N_FLTR+N_FLTR+0] = 0.f;
		}
	}

	mdl->normalisee__d     = cpu_vers_gpu<float>(mdl->normalisee,     BLOQUES * PRIXS * N_FLTR);
	mdl->dif_normalisee__d = cpu_vers_gpu<float>(mdl->dif_normalisee, BLOQUES * PRIXS * N_FLTR);
};

static uint * tout_zeroiser = UNIFORME_C(1);

Mdl_t * cree_mdl(
	uint T,
	uint Y[C], uint insts[C],
	ema_int_t * bloque[BLOQUES]
) {
	ASSERT(Y[C-1] == 1);
	ASSERT(Y[ 0 ] == BLOQUES * F_PAR_BLOQUES);
	ASSERT(insts[C-1] != FILTRES_PRIXS);			//	Afin d'assurer un Y=inst_VARS
	
	Mdl_t * mdl = alloc<Mdl_t>(1);

	mdl->T = T;

	//
	FOR(0, i, BLOQUES) {
		mdl->bloque[i]  = bloque[i];
		mdl->intervalles[i] = bloque[i]->intervalle;
		//
		mdl->type_de_norme[i] = bloque[i]->type_de_norme;
		mdl->min_theorique[i] = bloque[i]->min_theorique;
		mdl->max_theorique[i] = bloque[i]->max_theorique;
	};

	mdl->intervalles__d = cpu_vers_gpu<uint>(mdl->intervalles, BLOQUES);

	//
	calculer_normalisee__et__dif_normalisee(mdl);
	//raise(SIGINT);

	//	Architecture
	memcpy(mdl->insts,                 insts, sizeof(uint) * C);
	memcpy(mdl->Y,                         Y, sizeof(uint) * C);

	//	Allocation
	mdl->total_POIDS = 0;
	FOR(0, c, C) {
		if (c>0) ASSERT(insts[c] != 0);
		if (c==0) ASSERT(Y[c]==BLOQUES*F_PAR_BLOQUES);
		ASSERT(Y[c] <= MAX_Y);
		//
		cree_inst[insts[c]](mdl, c);
		//
		//mdl->p [c] = alloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->p__d [c] = cpu_vers_gpu<float>(mdl->p[c], mdl->inst_POIDS[c]);
		mdl->y__d [c] = cudalloc<float>(mdl->inst_VARS [c] * T * MEGA_T);
		mdl->l__d [c] = cudalloc<float>(mdl->inst_LOCDS[c] * T * MEGA_T);
		mdl->dy__d[c] = cudalloc<float>(mdl->inst_VARS [c] * T * MEGA_T);
		mdl->dp__d[c] = cudalloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->total_POIDS += mdl->inst_POIDS[c];
	}
	ASSERT(mdl->inst_DEPART_SORTIE[C-1] == 0);
	//
	mdl_normer_les_filtres(mdl);
	//
	//uint * tout_zeroiser = UNIFORME_C(1);
	//
	mdl_zero_deriv_gpu(mdl, tout_zeroiser);
	//
	return mdl;
};

PAS_OPTIMISER()
void mdl_verif(Mdl_t * mdl) {
	uint nombre;

	//	---
	nombre=0;
	FOR(0, i, BLOQUES * PRIXS * N_FLTR) if (isnan(mdl->normalisee[i])) nombre++;
	printf("mdl->normalisee a     \033[93m%i\033[0m / %i nan\n", nombre, BLOQUES * PRIXS * N_FLTR);

	//	---
	nombre=0;
	FOR(0, i, BLOQUES * PRIXS * N_FLTR) if (isnan(mdl->dif_normalisee[i])) nombre++;
	printf("mdl->dif_normalisee a \033[93m%i\033[0m / %i nan\n", nombre, BLOQUES * PRIXS * N_FLTR);

	FOR(0, c, C) {
		printf("c = %i :\n", c);

		//	---
		nombre=0;
		FOR(0, i, mdl->inst_POIDS[c]) if (isnan(mdl->p[c][i])) nombre++;
		printf("  mdl->p a     \033[93m%i\033[0m / %i nan\n", nombre, mdl->inst_POIDS[c]);

		//	---
		float * y = gpu_vers_cpu<float>(mdl->y__d[c], mdl->inst_VARS[c]);
		nombre=0;
		FOR(0, i, mdl->inst_VARS[c]) if (isnan(y[i])) nombre++;
		printf("  mdl->y__d a  \033[93m%i\033[0m / %i nan\n", nombre, mdl->inst_VARS[c]);
		free(y);

		//	---
		float * l = gpu_vers_cpu<float>(mdl->l__d[c], mdl->inst_LOCDS[c]);
		nombre=0;
		FOR(0, i, mdl->inst_LOCDS[c]) if (isnan(l[i])) nombre++;
		printf("  mdl->l__d a  \033[93m%i\033[0m / %i nan\n", nombre, mdl->inst_LOCDS[c]);
		free(l);

		//	---
		float * dy = gpu_vers_cpu<float>(mdl->dy__d[c], mdl->inst_VARS[c]);
		nombre=0;
		FOR(0, i, mdl->inst_VARS[c]) if (isnan(dy[i])) nombre++;
		printf("  mdl->dy__d a \033[93m%i\033[0m / %i nan\n", nombre, mdl->inst_VARS[c]);
		free(dy);

		//	---
		float * dp = gpu_vers_cpu<float>(mdl->dp__d[c], mdl->inst_POIDS[c]);
		nombre=0;
		FOR(0, i, mdl->inst_POIDS[c]) if (isnan(dp[i])) nombre++;
		printf("  mdl->dp__d a \033[93m%i\033[0m / %i nan\n", nombre, mdl->inst_POIDS[c]);
		free(dp);
	}
};

PAS_OPTIMISER()
void mdl_re_cree_poids(Mdl_t * mdl) {
	//	Allocation
	mdl->total_POIDS = 0;
	FOR(0, c, C) {
		if (c>0) ASSERT(mdl->insts[c] != 0);
		ASSERT(mdl->Y[c] <= MAX_Y);
		//
		free(mdl->p[c]);
		CONTROLE_CUDA(cudaFree(mdl->p__d[c]));
		//
		cree_inst[mdl->insts[c]](mdl, c);
		//
		//mdl->p [c] = alloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->p__d [c] = cpu_vers_gpu<float>(mdl->p[c], mdl->inst_POIDS[c]);
		//
		mdl->total_POIDS += mdl->inst_POIDS[c];
	}
};

PAS_OPTIMISER()
void mdl_changer_couche_Y(Mdl_t * mdl, uint c, uint nouveau_Y) {
	mdl->total_POIDS -= mdl->inst_POIDS[c];
	{
		if (c>0) ASSERT(mdl->insts[c] != 0);
		mdl->Y[c] = nouveau_Y;
		ASSERT(mdl->Y[c] <= MAX_Y);
		//
		free(mdl->p[c]);
		CONTROLE_CUDA(cudaFree(mdl->p__d[c]));
		//
		cree_inst[mdl->insts[c]](mdl, c);
		//
		//mdl->p [c] = alloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->p__d [c] = cpu_vers_gpu<float>(mdl->p[c], mdl->inst_POIDS[c]);
		//
		mdl->total_POIDS += mdl->inst_POIDS[c];
	}
};

void mdl_normer_les_filtres(Mdl_t * mdl) {
	FOR(0, b, BLOQUES) {
		FOR(0, f, F_PAR_BLOQUES) {
			float min, max;
			uint type_norme = mdl->type_de_norme[b];
			if (type_norme == NORME_CLASSIQUE) {
				max=mdl->p[0][b*F_PAR_BLOQUES*N + f*N+0];
				min=mdl->p[0][b*F_PAR_BLOQUES*N + f*N+0];
				FOR(1, i, N) {
					if (max < mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]) max = mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i];
					if (min > mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]) min = mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i];
				}
			} else if (type_norme == NORME_THEORIQUE || type_norme == NORME_RELATIVE) {
				min = 0.0;	//se sont des filtres
				max = 1.0;	//pas les natures

				//	Borne au cas ou
				FOR(0, i, N) {
					float f_val = mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i];
					mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i] = MAX2(MIN2(f_val, max), min);
				}
			} else {
				ERR("mdl->type_de_norme[b]=%i", mdl->type_de_norme[b]);
			}
			//
			FOR(0, i, N) mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i] = (mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]-min)/(max-min);
		};
	}
	CONTROLE_CUDA(cudaMemcpy(mdl->p__d[0], mdl->p[0], sizeof(float)*BLOQUES*F_PAR_BLOQUES*N, cudaMemcpyHostToDevice))
};

PAS_OPTIMISER()
void mdl_poids_cpu_vers_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p__d[c],  mdl->p[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
	}
};

PAS_OPTIMISER()
void mdl_poids_gpu_vers_cpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p[c],  mdl->p__d[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyDeviceToHost));
	}
};

PAS_OPTIMISER()
void mdl_gpu_vers_cpu(Mdl_t * mdl) {
	mdl_poids_gpu_vers_cpu(mdl);
}

PAS_OPTIMISER()
void mdl_cpu_vers_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p__d[c],  mdl->p[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
	}
};

PAS_OPTIMISER()
void liberer_mdl(Mdl_t * mdl) {
	FOR(0, c, C) {
		free(mdl->p [c]);
		//
		CONTROLE_CUDA(cudaFree(mdl->p__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->y__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->l__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->dy__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->dp__d[c]));
		//
		//FOR(0, i, mdl->inst_p_separateurs[c]) free(mdl->char_inst_p_separateurs[c][i]);
		//free(mdl->mdl->char_inst_p_separateurs[c]);
	}
};

PAS_OPTIMISER()
void mdl_zero_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemset(mdl->y__d [c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T * MEGA_T));
	}
};

PAS_OPTIMISER()
void mdl_zero_deriv_gpu(Mdl_t * mdl, uint zeroiser[C]) {
	FOR(0, c, C) {
		if (zeroiser[c]) {
			CONTROLE_CUDA(cudaMemset(mdl->dy__d[c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T * MEGA_T));
			CONTROLE_CUDA(cudaMemset(mdl->dp__d[c], 0, sizeof(float) * mdl->inst_POIDS[c]));
		}
	}
};