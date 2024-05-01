#pragma once

#include "etc.cuh"

//__device__ float  activation_f(uint act, float x);
//__device__ float activation_df(uint act, float x, float a);

#define cuda_math_LOGISTIC 0
#define cuda_math_TANH     1

/*	---------------------------------------------
				f(ax + b)
	---------------------------------------------  */

void nvidia_F_AX__shared_16(
	uint T,			//KERD(T)
	//
	float * x0, uint X0_vars, uint X0, uint depart_x0, uint x0_depart__t,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * a0, uint depart_a0,
	float * b , uint depart__b,
	//
	uint activation);

void d_nvidia_F_AX__shared_16(
	uint T,
	//
	float * x0, uint X0_vars, uint X0, uint depart_x0, float * dx0, uint x0_depart__t,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, float * dy, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * a0, uint depart_a0,	float * da0,
	float * b , uint depart__b, float * db,
	//
	uint activation);

/*	---------------------------------------------
				f(ax + bx + c)
	---------------------------------------------  */

void nvidia_F_AX_AX__shared_16(
	uint T,			//KERD(T)
	//
	float * x0, uint X0_vars, uint X0, uint depart_x0, uint x0_depart__t,
	float * x1, uint X1_vars, uint X1, uint depart_x1, uint x1_depart__t,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * a0, uint depart_a0,
	float * a1, uint depart_a1,
	float * b , uint depart__b,
	//
	uint activation);

void d_nvidia_F_AX_AX__shared_16(
	uint T,
	//
	float * x0, uint X0_vars, uint X0, uint depart_x0, float * dx0, uint x0_depart__t,
	float * x1, uint X1_vars, uint X1, uint depart_x1, float * dx1, uint x1_depart__t,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, float * dy, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * a0, uint depart_a0,	float * da0,
	float * a1, uint depart_a1, float * da1,
	float * b , uint depart__b, float * db,
	//
	uint activation);

/*	---------------------------------------------
				f(ax + bx + cx + d)
	---------------------------------------------  */

void nvidia_F_AX_AX_AX__shared_16(
	uint T,			//KERD(T)
	//
	float * x0, uint X0_vars, uint X0, uint depart_x0, uint x0_depart__t,
	float * x1, uint X1_vars, uint X1, uint depart_x1, uint x1_depart__t,
	float * x2, uint X2_vars, uint X2, uint depart_x2, uint x2_depart__t,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * a0, uint depart_a0,
	float * a1, uint depart_a1,
	float * a2, uint depart_a2,
	float * b , uint depart__b,
	//
	uint activation);

void d_nvidia_F_AX_AX_AX__shared_16(
	uint T,
	//
	float * x0, uint X0_vars, uint X0, uint depart_x0, float * dx0, uint x0_depart__t,
	float * x1, uint X1_vars, uint X1, uint depart_x1, float * dx1, uint x1_depart__t,
	float * x2, uint X2_vars, uint X2, uint depart_x2, float * dx2, uint x2_depart__t,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, float * dy, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * a0, uint depart_a0,	float * da0,
	float * a1, uint depart_a1, float * da1,
	float * a2, uint depart_a2, float * da2,
	float * b , uint depart__b, float * db,
	//
	uint activation);