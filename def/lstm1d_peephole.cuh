#pragma once

#include "insts.cuh"
#include "cuda_math.cuh"
#include "mdl.cuh"

//	+	Attention Spatio-Temporelle

/*	--- Peephole LSTM ---

f = logistique(sF = Fx@x + Fh@h[-1] + Fc@c[-1] + Fb)
i = logistique(sI = Ix@x + Ih@h[-1] + Ic@c[-1] + Ib)
u =       tanh(sU = Ux@x + Uh@h[-1] +          + Ub)
c = f*c[-1] + i*u
o = logistique(sO = Ox@x + Oh@h[-1] + Oc@c     + Ob)
ch = tanh(c)
h = o * ch

*/

#define   logistic(x)   (1.0 / (1+expf(-x)))
#define d_logistic(x,a) (a   * ( 1.0  -  a))

#define d_tanh(x,a) (1-a*a)

#define lstmpeephole_activ_CH(x) x//tanh(x)
#define d_lstmpeephole_activ_CH(x,a) 1//(1-a*a)

// 		-- Vars --
#define lstm_VARS(X,Y) (7*Y)
#define depart_f(X,Y)  (0*Y)
#define depart_i(X,Y)  (1*Y)
#define depart_u(X,Y)  (2*Y)
#define depart_c(X,Y)  (3*Y)
#define depart_o(X,Y)  (4*Y)
#define depart_ch(X,Y) (5*Y)
#define depart_h(X,Y)  (6*Y)

//	-- Dérivés Locales --
#define lstm_LOCDS(X,Y) (4*Y)
#define depart_dsF(X,Y) (0*Y)
#define depart_dsI(X,Y) (1*Y)
#define depart_dsU(X,Y) (2*Y)
#define depart_dsO(X,Y) (3*Y)

//	f = logistique(Fx@x + Fh@h + Fc@c + Fb)
#define depart_poids_f(X,Y) (0)
#define Fx(X,Y) (X*Y)
#define Fh(X,Y) (Y*Y)
#define Fc(X,Y) (Y*Y)
#define Fb(X,Y) ( Y )
//	i = logistique(Ix@x + Ih@h + Ic@c + Ib)
#define depart_poids_i(X,Y) (1*X*Y + 2*Y*Y + Y)
#define Ix(X,Y) (X*Y)
#define Ih(X,Y) (Y*Y)
#define Ic(X,Y) (Y*Y)
#define Ib(X,Y) ( Y )
//	u =       tanh(Ux@x + Uh@h +      + Ub)
#define depart_poids_u(X,Y) (2*X*Y + 4*Y*Y + 2*Y)
#define Ux(X,Y) (X*Y)
#define Uh(X,Y) (Y*Y)
#define Ub(X,Y) ( Y )
//	o = logistique(Ox@x + Oh@h + Oc@c + Ob)
#define depart_poids_o(X,Y) (3*X*Y + 5*Y*Y + 3*Y)
#define Ox(X,Y) (X*Y)
#define Oh(X,Y) (Y*Y)
#define Oc(X,Y) (Y*Y)
#define Ob(X,Y) ( Y )
//	--
#define lstm_POIDS(X,Y) (4*X*Y + 7*Y*Y + 4*Y)

//	------------- Departs --------------

void cree_lstm1d_peephole(Mdl_t * mdl, uint inst);
void plume_lstm1d_peephole(Mdl_t * mdl, uint c);

//	============================================

void regulariser_lstm1d_peephole(Mdl_t * mdl, uint c);

//	============================================

void nvidia_lstm1d_peephole_naive(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_lstm1d_peephole_shared_16_2(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_lstm1d_peephole(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);

//	============================================

void d_nvidia_lstm1d_peephole_naive(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_lstm1d_peephole_shared_16_2(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void df_lstm1d_peephole(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);