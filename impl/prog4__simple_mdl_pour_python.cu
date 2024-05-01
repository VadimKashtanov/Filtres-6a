#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

#include "marchee.cuh"

#include "filtres_prixs.cuh"
#include "dot1d_tanh.cuh"
#include "lstm1d_peephole.cuh"
#include "dot1d_tanh_elman.cuh"

static float filtre(
	uint depart,
	float * x, float * f,
	uint intervalle,
	uint type_norme, float _min, float _max)
{
	float normer_x[N];
	//
	FOR(0, i, N) normer_x[i] = x[depart - (i)*intervalle];
	//
	if (type_norme == NORME_CLASSIQUE) {
		_min=normer_x[0];
		_max=normer_x[0];
		//
		FOR(1, i, N) {
			float a = normer_x[i];
			if (a > _max) _max = a;
			if (a < _min) _min = a;
		}
	} else if (type_norme == NORME_THEORIQUE) {
		// rien
	} else if (type_norme == NORME_RELATIVE) {
		_max=fabs(normer_x[0]);
		//
		FOR(1, i, N) {
			float a = fabs(normer_x[i]);
			if (a > _max) _max = a;
		}
		_max = +_max;
		_min = -_max;
	} else {
		ERR("type_norme == %i", type_norme);
	}
	//
	FOR(0, i, N) normer_x[i] = (normer_x[i]-_min)/(_max-_min);
	
	float s = 0, d = 0;
	float f_nouveau = f[0];
	float x_nouveau = normer_x[0];
	//
	float Ps = (0.5+0/N*0.5);
	s += powf(1 + fabs(x_nouveau - f_nouveau), Ps);
	//
	float f_avant = f_nouveau;
	float x_avant = x_nouveau;
	FOR(1, i, N) {
		f_nouveau = f[i];
		x_nouveau = normer_x[i];
		//
		float Ps = (0.5+i/N*0.5);
		float Pd = (1.0+i/N*1.0);
		//
		s += powf(1 + fabs(  x_nouveau   -   f_nouveau  ), Ps);
		d += powf(1 + fabs((x_nouveau-x_avant) - (f_nouveau-f_avant)), Pd);
		f_avant   = f_nouveau;
		x_avant   = x_nouveau;
	};
	
	s = (s/(float)N) - 1;
	d = (d/(float)(N-1))-1;

	return fp_normalisation(expf(-2*s*s-2*d*d));
};

int main(int argc, char ** argv) {
	srand(0);
	//
	FILE * fp = fopen(argv[1], "rb");
	//
	uint * inst_Y = lire<uint>(fp, C);
	uint * _VARS  = lire<uint>(fp, C);
	uint * insts  = lire<uint>(fp, C);
	//
	//
	uint PRIXS_bitget  = lire_un<uint>(fp);
	//
	uint * intervalles = lire<uint>(fp, BLOQUES);
	//
	uint * type_norme  = lire<uint>(fp, BLOQUES);
	uint * _min        = lire<uint>(fp, BLOQUES);
	uint * _max        = lire<uint>(fp, BLOQUES);
	//
	float * lignes     = lire<float>(fp, PRIXS_bitget*BLOQUES);
	//
	float * poids[C];
	FOR(0, c, C) {
		uint POIDS = lire_un<uint>(fp);
		poids[c]   = lire<float>(fp, POIDS);
	}
	//
	fclose(fp);

	//	------------- Espace ----------------
	uint T = (PRIXS_bitget-DEPART);
	//
	float * espace_y[C];
	FOR(0, c, C) {
		espace_y[c] = alloc<float>(T * _VARS[c]);
		memset(espace_y[c], 0, sizeof(float) * T * _VARS[c]);
	}

	uint DEPART_X[C];
	FOR(0, c, C) {
		DEPART_X[c] = _VARS[c] - inst_Y[c];
		//printf("Depart_x = %i\n", DEPART_X[c]);
	}

	//	------------- Calcule ----------------
	FOR(0, t, T) {
		FOR(0, c, C) {
			if (insts[c] == FILTRES_PRIXS) {
				ASSERT(c == 0);
				//
				float * x = NULL;
				float * y = espace_y[0];
				//
				FOR(0, f, BLOQUES*F_PAR_BLOQUES) {
					uint b = (f - (f % F_PAR_BLOQUES)) / F_PAR_BLOQUES;
					y[t*_VARS[c] + f] = filtre(
						b*PRIXS_bitget + DEPART + t,	//depart
						lignes,
						poids[0] + f*N,
						intervalles[b],
						type_norme[b],
						_min[b], _max[b]
					);
				};
			} else if (insts[c] == DOT1D_TANH) {
				float * x = espace_y[c-1];
				float * y = espace_y[ c ];
				//
				uint X = inst_Y[c-1];
				uint Y = inst_Y[ c ];
				//
				FOR(0, i, Y) {
					float s = poids[c][(X+1)*i + X-1+1];
					FOR(0, j, X) {
						s += poids[c][(X+1)*i + j] * x[t*_VARS[c-1] + DEPART_X[c-1] + j];
					}
					y[t*_VARS[c] + i] = dot1d_tanh_ACTIV(i,s);
				};
			} else if (insts[c] == DOT1D_TANH_ELMAN) {
#define CYCLE_DOT1D_TANH_ELMAN 24
				uint condition = (t>0 && !(t % CYCLE_DOT1D_TANH_ELMAN==0));
				float * x = espace_y[c-1];
				float * y = espace_y[ c ];
				//
				uint X = inst_Y[c-1];
				uint Y = inst_Y[ c ];
				//
				FOR(0, _y, Y) {
					float s = poids[c][elman_depart_poid_Ub(X,Y) + _y];
					FOR(0, ix, X) {
						s += poids[c][elman_depart_poid_Ux(X,Y) + _y*X + ix] * x[t*_VARS[c-1] + DEPART_X[c-1] + ix];
					}
					if (condition) {
						FOR(0, iy, Y) {
							s += condition * y[(t-1)*Y + iy] * poids[c][elman_depart_poid_Uy(X,Y) + _y*Y + iy];
						}
					}
					y[t*_VARS[c] + _y] = tanh(s);
				};
			} else if (insts[c] == LSTM1D_PEEPHOLE) {
#define CYCLE_LSTM1D_PEEPHOLE 12
				uint condition = (t>0 && !(t % CYCLE_LSTM1D_PEEPHOLE==0));
				/*
	//	--- Partie fiu ---
	f = logistique(sF = Fx@x + Fh@h + Fc@c[-1] + Fb)
	i = logistique(sI = Ix@x + Ih@h + Ic@c[-1] + Ib)
	u =       tanh(sU = Ux@x + Uh@h +          + Ub)
	//	--- Partie cch ---
	c = f*c[-1] + i*u
	ch = tanh(c)
	//	--- Partie o ---
	o = logistique(sO = Ox@x + Oh@h + Oc@c    + Ob)
	//	--- Partie h ---
	h = o * ch
				*/
				//
				float * x = espace_y[c-1];
				float * y = espace_y[ c ];
				//
				uint X = inst_Y[c-1];
				uint Y = inst_Y[ c ];
				//
				float f[Y], i[Y], u[Y], o[Y];
				//
				memset(f, 0, sizeof(float)*Y);
				memset(i, 0, sizeof(float)*Y);
				memset(u, 0, sizeof(float)*Y);
				memset(o, 0, sizeof(float)*Y);
				//
				FOR(0, _y, Y) {
					//	@x
					FOR(0, ix, X) {
						float val_x = x[t*_VARS[c-1] + DEPART_X[c-1] + ix];
						f[_y] += val_x * poids[c][depart_poids_f(X,Y)+(_y*X+ix)];
						i[_y] += val_x * poids[c][depart_poids_i(X,Y)+(_y*X+ix)];
						u[_y] += val_x * poids[c][depart_poids_u(X,Y)+(_y*X+ix)];
						o[_y] += val_x * poids[c][depart_poids_o(X,Y)+(_y*X+ix)];
					};
						
					//	@h
					FOR(0, ih, Y) {
						float val_h = (condition ? y[(t-1)*_VARS[c] + depart_h(X,Y) + ih] : 0.0);
						f[_y] += val_h * poids[c][depart_poids_f(X,Y)+Fx(X,Y)+(_y*Y+ih)];
						i[_y] += val_h * poids[c][depart_poids_i(X,Y)+Ix(X,Y)+(_y*Y+ih)];
						u[_y] += val_h * poids[c][depart_poids_u(X,Y)+Ux(X,Y)+(_y*Y+ih)];
						o[_y] += val_h * poids[c][depart_poids_o(X,Y)+Ox(X,Y)+(_y*Y+ih)];
					};
						
					//	@c[-1]
					FOR(0, ic, Y) {
						float val_c = (condition ? y[(t-1)*_VARS[c] + depart_c(X,Y) + ic] : 0.0);
						f[_y] += val_c * poids[c][depart_poids_f(X,Y)+Fx(X,Y)+Fh(X,Y)+(_y*Y+ic)];
						i[_y] += val_c * poids[c][depart_poids_i(X,Y)+Ix(X,Y)+Ih(X,Y)+(_y*Y+ic)];
					}
						
					//	+b
					f[_y] += poids[c][depart_poids_f(X,Y)+Fx(X,Y)+Fh(X,Y)+Fc(X,Y)+_y];
					i[_y] += poids[c][depart_poids_i(X,Y)+Ix(X,Y)+Ih(X,Y)+Ic(X,Y)+_y];
					u[_y] += poids[c][depart_poids_u(X,Y)+Ux(X,Y)+Uh(X,Y)        +_y];
					o[_y] += poids[c][depart_poids_o(X,Y)+Ox(X,Y)+Oc(X,Y)+Oc(X,Y)+_y];

					//	()
					f[_y] = logistic(f[_y]);
					i[_y] = logistic(i[_y]);
					u[_y] =     tanh(u[_y]);

					//c = f*c[-1] + i*u
					float c1 = (condition ? y[(t-1)*_VARS[c] + depart_c(X,Y) + _y] : 0.0);
					y[t*_VARS[c] + depart_c(X,Y)  + _y] = f[_y]*c1 + i[_y]*u[_y];
					y[t*_VARS[c] + depart_ch(X,Y) + _y] = lstmpeephole_activ_CH(y[t*_VARS[c] + depart_c(X,Y) + _y]);
				}

				FOR(0, _y, Y) {
					//	+Oc@c
					FOR(0, ic, Y) {
						float val_c = y[t*_VARS[c] + depart_c(X,Y) + ic];
						o[_y] += val_c * poids[c][depart_poids_o(X,Y)+Ox(X,Y)+Oh(X,Y)+(_y*Y+ic)];
					}

					//	()
					o[_y] = logistic(o[_y]);

					//	h
					y[t*_VARS[c] + depart_h(X,Y) + _y] = o[_y] * y[t*_VARS[c] + depart_ch(X,Y) + _y];
				}
			} else {
				ERR("Inst = %i", insts[c]);
			}
		}
	};

	//	---------- Ecrire Resultat ----------
	fp = fopen(argv[1], "wb");
	//
	float res[T];
	FOR(0, t, T) {
		res[t] = espace_y[C-1][t*_VARS[C-1] + DEPART_X[C-1] + 0];
	}
	FWRITE(res, sizeof(float), T, fp);
	//
	fclose(fp);
}