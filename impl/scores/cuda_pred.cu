#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

static __global__ void kerd_nvidia_prediction_somme(
	uint _t_MODE, uint GRAINE,
	float * y, uint t0, uint T,
	float * pred, float * _PRIXS)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thx < T) {
		FOR(0, mega_t, MEGA_T) {
			uint depart_plus_t = t_MODE(
				_t_MODE, GRAINE,
				t0, t0+T*MEGA_T,
				thx, mega_t,
				T, MEGA_T
			);
			//
			float p1 = _PRIXS[depart_plus_t+1];
			float p0 = _PRIXS[depart_plus_t];
			atomicAdd(
				pred,
				1.0*(uint)(cuda_signe((y[(mega_t*T*1 + 0+thx)*1+0])) == cuda_signe((p1/p0-1)))
			);
		}
	};
};

float nvidia_prediction(float * y, uint depart, uint T, uint _t_MODE, uint GRAINE) {
	float * pred__d = cudalloc<float>(1);
	CONTROLE_CUDA(cudaMemset(pred__d, 0, 1*sizeof(float)));
	kerd_nvidia_prediction_somme<<<dim3(KERD(T,1024)),dim3(1024)>>>(
		_t_MODE, GRAINE,
		y, depart, T,
		pred__d, cuda_MARCHEE_DE_TRADE
	);
	ATTENDRE_CUDA();
	float _pred;
	CONTROLE_CUDA(cudaMemcpy(&_pred, pred__d, sizeof(float)*1, cudaMemcpyDeviceToHost));
	cudafree<float>(pred__d);
	return _pred / (float)(T*MEGA_T);
};