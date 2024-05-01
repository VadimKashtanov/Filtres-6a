#pragma once
#include "insts.cuh"

#include "mdl.cuh"

#define fp_normalisation(x) /*(x)*/(2*x-1) //x
#define fp_d_normalisation(x) /*1*/2

//	=====================================

void cree_filtres_prixs(Mdl_t * mdl, uint inst);
void plume_filtres_prixs(Mdl_t * mdl, uint c);

//	=====================================

void regulariser_filtres_prixs(Mdl_t * mdl, uint c);

//	=====================================

void nvidia_filtres_prixs___naive(			//	mode == 1
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd);

void nvidia_filtres_prixs___shared(			//	mode == 2
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd);

void f_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);

//	----------------------------

void d_nvidia_filtres_prixs___naive(		//	mode == 1
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df);

void d_nvidia_filtres_prixs___shared(		//	mode == 2
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df);

void df_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);