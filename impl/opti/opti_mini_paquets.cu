#include "opti.cuh"

void optimisation_mini_packet(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	float * pourcent_masque,
	float * pourcent_opti_masque,
	uint PERTURBATIONS,
	uint zero_accumulation_tous_les[C],
	uint _t_MODE, uint GRAINE)
{
	uint _T_mini_paquet = mdl->T;
	//
	uint _t0 = t0 + (rand()%(t1-_T_mini_paquet-t0));
	//
	printf("mini paquet entre %i et %i (delta=%i*%i=%i)\n",
		_t0, _t0+_T_mini_paquet,
		_T_mini_paquet, MEGA_T, _T_mini_paquet*MEGA_T);
	//
	optimiser(
		mdl,
		_t0, _t0+_T_mini_paquet,
		alpha, div,
		RMSPROP, I,
		pourcent_masque, pourcent_opti_masque,
		PERTURBATIONS,
		zero_accumulation_tous_les,
		_t_MODE, GRAINE);
};