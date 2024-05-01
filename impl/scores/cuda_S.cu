#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define pseudo_alea_d_une_grain(i) ((float)((121+(i%1234))*31 % 1001 ) / 1001.0)

//	===============================================================

static __global__ void kerd_nvidia_score_somme(
	uint _t_MODE, uint GRAINE,
	float * y, uint t0, uint T,
	float * score, float * _PRIXS)
{
	uint t = threadIdx.x + blockIdx.x + blockDim.x;
	//
	if (t < T) {
		float s = 0;
		//
		FOR(0, mega_t, MEGA_T) {
			uint depart_plus_t = t_MODE(
				_t_MODE, GRAINE,
				t0, t0+T*MEGA_T,
				t, mega_t,
				T, MEGA_T
			);

			float _y = y[(mega_t*T*1 + 0 + t)*1 + 0];
			float alea = 2*(PSEUDO_ALEA_cuda((t + ((uint)_y % 10001)))%1000)/1000.0-1;
			_y += alea * SCORE_Y_COEF_BRUIT;
			s += cuda_SCORE(
				_y, _PRIXS[depart_plus_t+1], _PRIXS[depart_plus_t],
				alea * SCORE_Y_COEF_BRUIT
			);
		}
		//
		score[t] = s;
		//atomicAdd(&score[0], s);
	}
};

#define HORIZON 32

static __global__ void kerd_addition_horizontale(
	float * vecteur, uint T, float * somme_finale)
{
	uint thx = threadIdx.x;
	uint t = threadIdx.x + blockIdx.x * blockDim.x;
	//
	uint __BLOQUE = blockDim.x;
	//
	if (t < T) {
		uint depart_bloque = 2*(t - (t% __BLOQUE));
		//
		for (uint mul=1; mul <= HORIZON;) {
			if (thx % mul == 0) {
				// a = b + c
				uint a = depart_bloque + 2*thx;
				uint b = depart_bloque + 2*thx;
				uint c = depart_bloque + 2*thx + 2*(mul)/2;
				//
				if (!(a < T)) assert(0);
				if (!(b < T)) assert(0);
				if (!(c < T)) assert(0);
				//
				vecteur[a] = vecteur[b] + vecteur[c];
			}
			__syncthreads();
			mul *= 2;
		}
		//
		if (thx == 0) atomicAdd(&somme_finale[0], vecteur[depart_bloque+0]);
	};
};

float nvidia_somme_score(float * y, uint depart, uint T, uint _t_MODE, uint GRAINE)
{
	ASSERT(T % (HORIZON*2) == 0);
	//
	float * somme_score__d = cudalloc<float>(T);
	float * somme_score_finale__d = cudalloc<float>(1);
	CONTROLE_CUDA(cudaMemset(somme_score_finale__d, 0, sizeof(float)*1));
	CONTROLE_CUDA(cudaMemset(somme_score__d, 0, sizeof(float)*T));
	
	//	--- Calcule du Score ---
	kerd_nvidia_score_somme<<<dim3(KERD(T,1)),dim3(1)>>>(
		_t_MODE, GRAINE,
		y, depart, T,
		somme_score__d, cuda_MARCHEE_DE_TRADE
	);
	ATTENDRE_CUDA();

	//	--- Somme Horizontale ---
	kerd_addition_horizontale<<<dim3(KERD(T/2,HORIZON)),dim3(HORIZON)>>>(
		somme_score__d,
		T, somme_score_finale__d
	);
	ATTENDRE_CUDA();

	//	Gpu vers Cpu
	float * somme_score = gpu_vers_cpu<float>(somme_score_finale__d, 1);
	float somme = somme_score[0];
	//
	CONTROLE_CUDA(cudaFree(somme_score__d));
	CONTROLE_CUDA(cudaFree(somme_score_finale__d));
	free(somme_score);
	//
	return somme;
};

float  nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE) {
	return APRES_SCORE(somme / (float)(1 * T * MEGA_T));
};

/*	Regularisation L2
	Pas oublier le Attention mechanisme
*/

//	===============================================================

float d_nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE) {
	return dAPRES_SCORE(somme / (float)(1 * T * MEGA_T)) / (float)(1 * T * MEGA_T);
};

//	===============================================================

static __global__ void kerd_nvidia_score_dpowf(
	uint _t_MODE, uint GRAINE,
	float _dy, float * y, float * dy,
	uint t0, uint T,
	float * _PRIXS)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;

	if (_t < T) {
		FOR(0, mega_t, MEGA_T) {
			uint depart_plus_t = t_MODE(
				_t_MODE, GRAINE,
				t0, t0+T*MEGA_T,
				_t, mega_t,
				T, MEGA_T
			);
			float _y = y[(mega_t*T*1 + 0+_t)*1+0];
			float alea = 2*(PSEUDO_ALEA_cuda((_t + ((uint)_y % 10001)))%1000)/1000.0-1;
			atomicAdd(&dy[(mega_t*T*1 + 0+_t)*1+0], _dy * cuda_dSCORE(
				y[(mega_t*T*1 + 0+_t)*1+0]+alea*SCORE_Y_COEF_BRUIT, _PRIXS[depart_plus_t+1], _PRIXS[depart_plus_t], alea*SCORE_Y_COEF_BRUIT
			));	//atomicAdd car certaines fonction prennent y[-1] comme fin
		}
	}
};

void d_nvidia_somme_score(float d_score, float * y, float * dy, uint depart, uint T, uint _t_MODE, uint GRAINE) {
	kerd_nvidia_score_dpowf<<<dim3(KERD(T,1024)), dim3(1024)>>>(
		_t_MODE, GRAINE,
		d_score,
		y, dy,
		depart, T,
		cuda_MARCHEE_DE_TRADE
	);
	ATTENDRE_CUDA();
};