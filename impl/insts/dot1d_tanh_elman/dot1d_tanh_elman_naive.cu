#include "dot1d_tanh_elman.cuh"

#define BLOQUE_T 16//32
#define BLOQUE_Y 16//32

static __global__ void kerd_dot1d_tanh_elman_naive(
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
		float s = 0;
		//
		FOR(0, ix, X) {
			s += x[(mega_t*T+0+_t)*X_vars + DEPART_x + ix] * p[elman_depart_poid_Ux(X,Y) + _y*X + ix];
		}
		if (mega_t != 0) {
			FOR(0, iy, Y) {
				s += y[( (mega_t-1)*T+0+_t )*Y + iy] * p[elman_depart_poid_Uy(X,Y) + _y*Y + iy];
			}
		}
		s += p[elman_depart_poid_Ub(X,Y) + _y];
		//
		float a = tanh(s);
		   y[(mega_t*T+0+_t)*Y + _y] = a;
		locd[(mega_t*T+0+_t)*Y + _y] = d_tanh(s,a);
	}
};

void nvidia_dot1d_tanh_elman_naive(
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
	kerd_dot1d_tanh_elman_naive<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
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

static __global__ void kerd_deriv_dot1d_tanh_elman_naive(
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
		float _locd = locd[(mega_t*T+0+_t)*Y + _y] * dy[(mega_t*T+0+_t)*Y + _y];
		atomicAdd(&dp[elman_depart_poid_Ub(X,Y) + _y], _locd);
		//
		FOR(0, ix, X) {
			//s += x[(mega_t*T+0+_t)*X_vars + DEPART_x + ix] * p[elman_depart_poid_Ux(X,Y) + _y*X + ix];
			atomicAdd(&dx[(mega_t*T+0+_t)*X_vars + DEPART_x + ix], _locd * p[elman_depart_poid_Ux(X,Y) + _y*X + ix]);
			atomicAdd(&dp[elman_depart_poid_Ux(X,Y) + _y*X + ix], _locd * x[(mega_t*T+0+_t)*X_vars + DEPART_x + ix]);
		}
		if (mega_t != 0) {
			FOR(0, iy, Y) {
				//s += y[( (mega_t-1)*T+0+_t )*Y + iy] * p[elman_depart_poid_Uy(X,Y) + _y*X + iy];
				atomicAdd(&dy[( (mega_t-1)*T+0+_t )*Y + iy], _locd * p[elman_depart_poid_Uy(X,Y) + _y*Y + iy]);
				atomicAdd(&dp[elman_depart_poid_Uy(X,Y) + _y*Y + iy], _locd * y[( (mega_t-1)*T+0+_t )*Y + iy]);
			}
		}
	}
};

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
	float * dp)
{
	kerd_deriv_dot1d_tanh_elman_naive<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
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