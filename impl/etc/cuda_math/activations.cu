#include "cuda_math.cuh"

static __device__ float  activation_f(uint act, float x) {
	if (act == 0) return 1 / (1 + expf(-x));
	else if (act == 1) return tanh(x);
	else assert(0);
};

static __device__ float activation_df(uint act, float x, float a) {
	if (act == 0) return a*(1-a);
	else if (act == 1) return 1 - a*a;
	else assert(0);
	return 0;
};