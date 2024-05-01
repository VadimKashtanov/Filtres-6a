#pragma once

#include "marchee.cuh"

#include "insts.cuh"

#define L2_regularisation 0.0001 //0.00001

#define SCORE_Y_COEF_BRUIT 0.0//0.01
#define ENTREE_COEF_BRUIT 0.1

#define P_S      2.00
#define P_somme  1.00
#define K_fonc(x) ((powf(x,0.25)*1.0+powf(x,1.0)*0.0)/1)

#define sng(x)	((x>=0) ? 1.0 : -1.0)

#define PUISS(diff,P)  (powf(diff,P)/P)
#define dPUISS(diff,P) (powf(diff,P-1))

#define COEF_PERTE 1.0//1.0

#define S(y,w)  ((sng(y)==sng(w)) ?  PUISS(y-w,P_S) : ( PUISS(y-w,P_S)*COEF_PERTE))
#define dS(y,w) ((sng(y)==sng(w)) ? dPUISS(y-w,P_S) : (dPUISS(y-w,P_S)*COEF_PERTE))

#define K(p1,p0,alea) K_fonc(fabs((100+5*alea/(SCORE_Y_COEF_BRUIT>0?SCORE_Y_COEF_BRUIT:1.0))*(p1/p0 - 1)))

#define __SCORE(y,p1,p0,alea)  (K(p1,p0,alea) *  S(y, sng(p1/p0 - 1)))
#define __dSCORE(y,p1,p0,alea) (K(p1,p0,alea) * dS(y, sng(p1/p0 - 1)))

//	----

static float SCORE(float y, float p1, float p0, float alea) {
	return __SCORE(y,p1,p0,alea);
};

static float APRES_SCORE(float somme) {
	return powf(somme, P_somme) / P_somme;
};

static float dAPRES_SCORE(float somme) {
	return powf(somme, P_somme - 1);
};

static float dSCORE(float y, float p1, float p0, float alea) {
	return __dSCORE(y,p1,p0,alea);
};

//	----

static __device__ float cuda_SCORE(float y, float p1, float p0, float alea) {
	return __SCORE(y,p1,p0,alea);
};

static __device__ float cuda_dSCORE(float y, float p1, float p0, float alea) {
	return __dSCORE(y,p1,p0,alea);
};

//	S(x) --- Score ---

float nvidia_somme_score(float * y, uint depart, uint T, uint _t_MODE, uint GRAINE);

float nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE);

//	dx

float d_nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE);

void d_nvidia_somme_score(float d_somme, float * y, float * dy, uint depart, uint T, uint _t_MODE, uint GRAINE);

//	%% Prediction

float nvidia_prediction(float * y, uint depart, uint T, uint _t_MODE, uint GRAINE);