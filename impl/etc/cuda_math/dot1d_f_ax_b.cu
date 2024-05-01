#include "cuda_math.cuh"

#define BLOQUE     16
#define BLOQUE_MAX BLOQUE//16

#include "./activations.cu"

//y = activation(a0@x0 + a1@x1 + a2@x2 + b)

static __global__ void kerd_f_ax_b__shared_16__t(
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
	uint activation)
{
	// <KERD(T, BLOQUE), KERD(Y,BLOQUE)>
	// <         BLOQUE,         BLOQUE>

	__shared__ float __partage__x[BLOQUE][BLOQUE];
	__shared__ float __partage__p[BLOQUE][BLOQUE];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _y = thx + blockIdx.x * blockDim.x;
	uint _t = thy + blockIdx.y * blockDim.y;

	float s = 0;

	//	+a0@x0
	FOR(0, d, X0/BLOQUE) {
		__partage__x[thy][thx] = x0[(x0_depart__t+_t)*X0_vars + depart_x0 + (d*BLOQUE + thx)];
		__partage__p[thy][thx] = a0[   depart_a0             + _y*X0     + (d*BLOQUE + thy)];
		__syncthreads();
#pragma unroll
		FOR(0, i, BLOQUE) s += __partage__x[thy][i] * __partage__p[i][thx];
		__syncthreads();
	};

	//	+b
#define __partage__b __partage__x[0]
	if (thy == 0) __partage__b[thx] = b[depart__b + _y];
	__syncthreads();

	s = (s + __partage__b[thx]);
	float a = activation_f(activation, s);
	y[(y__depart__t+_t)*Y__vars + depart__y + _y] = a;
	l[(y__depart__t+_t)*L__vars + depart__l + _y] = activation_df(activation, s,a);
};

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
	uint activation)
{
	ASSERT(T%BLOQUE==0);
	ASSERT(X0%BLOQUE==0);
	ASSERT(Y%BLOQUE==0);
	//
	kerd_f_ax_b__shared_16__t<<<dim3(KERD(Y, BLOQUE), KERD(T, BLOQUE)), dim3(BLOQUE, BLOQUE)>>>(
		T,
		//
		x0, X0_vars, X0, depart_x0, x0_depart__t,
		//
		y, Y__vars, Y, depart__y, y__depart__t,
		l, L__vars,    depart__l,
		//
		a0, depart_a0,
		b , depart__b,
		//
		activation);
	//ATTENDRE_CUDA();
}

//	======================================================================================
//	======================================================================================
//	======================================================================================

static __global__ void d_kerd_f_ax_b__shared_16___dX( //pour a b c
	uint T,
	//
	float * x, uint X_vars, uint X, uint depart_x, float * dx,  uint x_depart__t,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, float * dy, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * a, uint depart_a,	float * da,
	//
	uint activation)
{
	//dx = (p @ ((y-_y)*dtanh(x@p)).T).T

	__shared__ float __partage__x[BLOQUE][BLOQUE];
	__shared__ float __partage__p[BLOQUE][BLOQUE];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _x = thx + blockIdx.x * blockDim.x;
	uint _t = thy + blockIdx.y * blockDim.y;

	float s = 0;

	FOR(0, d, Y/BLOQUE) {
		float _l  = l[(y__depart__t+_t)*L__vars + depart__l + (d*BLOQUE+thx)];
		float _dy =   dy[(y__depart__t+_t)*Y__vars + depart__y + (d*BLOQUE+thx)];
		__partage__x[thy][thx] =  _l * _dy;
		__partage__p[thy][thx] = a[depart_a + (d*BLOQUE+thy)*X + _x];
		__syncthreads();
#pragma unroll
		FOR(0, i, BLOQUE) s += __partage__x[thy][i] * __partage__p[i][thx];
		__syncthreads();
	};

	atomicAdd(&dx[(x_depart__t+_t)*X_vars + depart_x + _x], s);
};

static __global__ void d_kerd_f_ax_b__shared_16___dA(
	uint T,
	//
	float * x, uint X_vars, uint X, uint depart_x, float * dx, uint x_depart__t,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, float * dy, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * a, uint depart_a,	float * da,
	//
	uint activation)
{
	//dp = x.T @ ((y-_y)*dtanh(x@p))

	__shared__ float __partage__x[BLOQUE_MAX][BLOQUE_MAX];
	__shared__ float __partage__p[BLOQUE_MAX][BLOQUE_MAX];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _x = thx + blockIdx.x * blockDim.x;
	uint _y = thy + blockIdx.y * blockDim.y;

	float s = 0;

	uint d = blockIdx.z;
	//FOR(0, d, T/BLOQUE) {
		float __l =  l[(y__depart__t+d*BLOQUE_MAX+thx)*L__vars + depart__l + _y];
		float _dy = dy[(y__depart__t+d*BLOQUE_MAX+thx)*Y__vars + depart__y + _y];
		__partage__x[thy][thx] = __l * _dy; 
		__partage__p[thy][thx] = x[(x_depart__t+(d*BLOQUE_MAX+thy))*X_vars + depart_x + _x];
		__syncthreads();

	#pragma unroll
		FOR(0, i, BLOQUE_MAX) {
			s += __partage__x[thy][i] * __partage__p[i][thx];
		}
		__syncthreads();
	//};

	atomicAdd(&da[depart_a + _y*X + _x], s);
};

static __global__ void d_kerd_f_ax_b__shared_16___db(
	uint T,
	//
	float *  y, uint Y__vars, uint  Y, uint depart__y, float * dy, uint y__depart__t,
	float *  l, uint L__vars,          uint depart__l,
	//
	float * b , uint depart__b, float * db,
	//
	uint activation)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	uint _t = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		float _l  =  l[(y__depart__t+_t)*L__vars + depart__l + _y];
		float _dy = dy[(y__depart__t+_t)*Y__vars + depart__y + _y];
		atomicAdd(&db[depart__b + _y], _l * _dy);
	}
};

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
	uint activation)
{
	ASSERT(T%BLOQUE==0);
	ASSERT(X0%BLOQUE==0);
	ASSERT(Y%BLOQUE==0);
	
	//dx0 = (a0 @ (dy*dtanh(x@p)).T).T
	d_kerd_f_ax_b__shared_16___dX<<<dim3(KERD(X0, BLOQUE), KERD(T, BLOQUE)), dim3(BLOQUE, BLOQUE)>>>(
		T,
		//
		x0, X0_vars, X0, depart_x0, dx0, x0_depart__t,
		//
		y, Y__vars, Y, depart__y, dy, y__depart__t,
		l, L__vars,    depart__l,
		//
		a0, depart_a0, da0,
		//
		activation);

	// ==============================================================================

	d_kerd_f_ax_b__shared_16___dA<<<dim3(KERD(X0, BLOQUE_MAX), KERD(Y, BLOQUE_MAX), DIV(T,BLOQUE_MAX)), dim3(BLOQUE_MAX, BLOQUE_MAX,1)>>>(
		T,
		//
		x0, X0_vars, X0, depart_x0, dx0, x0_depart__t,
		//
		y, Y__vars, Y, depart__y, dy, y__depart__t,
		l, L__vars,    depart__l,
		//
		a0, depart_a0, da0,
		//
		activation);

	//	=============================================================================

	d_kerd_f_ax_b__shared_16___db<<<dim3(KERD(Y, BLOQUE_MAX), KERD(T,BLOQUE_MAX)), dim3(BLOQUE_MAX, BLOQUE_MAX)>>>(
		T,
		//
		y, Y__vars, Y, depart__y, dy, y__depart__t,
		l, L__vars,    depart__l,
		//
		b , depart__b, db,
		//
		activation);
}