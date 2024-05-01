#pragma once

#include "insts.cuh"

#include "cuda_math.cuh"

#define d_tanh(s,a)  (1 - a*a)

#include "mdl.cuh"

/*
	y = g(Ux@x + Uy@y[-1] + Ub)
*/

#define elman_depart_poid_Ux(X,Y) (0)
#define elman_depart_poid_Uy(X,Y) (X*Y)
#define elman_depart_poid_Ub(X,Y) (X*Y+Y*Y)

//	============================================

void cree_dot1d_tanh_elman(Mdl_t * mdl, uint inst);
void plume_dot1d_tanh_elman(Mdl_t * mdl, uint c);

//	============================================

void regulariser_dot1d_tanh_elman(Mdl_t * mdl, uint c);

//	============================================

void nvidia_dot1d_tanh_elman_naive(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_tanh_elman_shared_2_16(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_dot1d_tanh_elman(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);

//	============================================

void d_nvidia_dot1d_tanh_elman_naive(
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

void d_nvidia_dot1d_tanh_elman_shared_2_16(
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

void df_dot1d_tanh_elman(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);