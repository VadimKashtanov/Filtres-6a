#pragma once

#include "insts.cuh"

#define _______tanh_f(s)    (tanh(s))
#define ______tanh_df(s,a)  (1 - a*a)

#define _logistique_f(s)   	(1.f/(1.f + expf(-s)))
#define logistique_df(s,a) 	(a * (1.0 - a))

#define ______gauss_f(s)	(expf(-s*s))
#define _____gauss_df(s,a)	(2*s*a)

#define _______ReLu_f(s)	(s>=0?s:0)
#define ______ReLu_df(s,a)	((float)(s>=0))

#define dot1d_tanh_ACTIV(yid, s) _______tanh_f(s)

#define dot1d_tanh_dACTIV(yid, s, a) ______tanh_df(s,a)

#include "mdl.cuh"

//	============================================

void cree_dot1d_tanh (Mdl_t * mdl, uint c);
void plume_dot1d_tanh(Mdl_t * mdl, uint c);

//	============================================

void regulariser_dot1d_tanh(Mdl_t * mdl, uint c);

//	============================================

void nvidia_dot1d_tanh_naive(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_tanh_shared_2_16(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_dot1d_tanh(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);

//	============================================

void d_nvidia_dot1d_tanh_naive(
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

void d_nvidia_dot1d_tanh_shared_2_16(
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

void df_dot1d_tanh(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);