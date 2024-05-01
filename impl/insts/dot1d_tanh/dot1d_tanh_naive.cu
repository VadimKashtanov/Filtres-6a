#include "dot1d_tanh.cuh"

#define BLOQUE_T 16//32
#define BLOQUE_Y 16//32

static __global__ void kerd_dot1d_tanh_naive(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		float s = p[_y*(X+1) + (X+1-1)];
		FOR(0, i, X) s += x[(mega_t*T+0+_t)*X_vars + DEPART_x + i] * p[_y*(X+1) + i];
		float a = dot1d_tanh_ACTIV(_y, s);
		y[/*(depart+_t)*/(mega_t*T+0+_t)*Y + _y] = a;
		locd[/*(depart+_t)*/(mega_t*T+0+_t)*Y + _y] = dot1d_tanh_dACTIV(_y, s,a);
	}
};

void nvidia_dot1d_tanh_naive(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	kerd_dot1d_tanh_naive<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
}

//	============================= Derivation ==============================

static __global__ void kerd_deriv_dot1d_tanh_naive(
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
	float * dp)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		float _locd = locd[/*(depart+_t)*/(mega_t*T+0+_t)*Y + _y] * dy[/*(depart+_t)*/(mega_t*T+0+_t)*Y + _y];
		atomicAdd(&dp[_y*(X+1) + (X+1-1)], _locd);
		FOR(0, i, X) {
			atomicAdd(&dx[(mega_t*T+0+_t)*X_vars + DEPART_x + i], _locd * p[_y*(X+1) + i]);
			atomicAdd(&dp[_y*(X+1) + i], _locd * x[(mega_t*T+0+_t)*X_vars + DEPART_x + i]);
		}
	}
};

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
	float * dp)
{
	kerd_deriv_dot1d_tanh_naive<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();
};