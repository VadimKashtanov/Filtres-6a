#pragma once

#include "etc.cuh"
#include "marchee.cuh"

/* /!\ /!\
	Ceci ne s'applique qu'aux filtres et au score
	car c'est lié au temps reel.
	dot1d ... sont dans T * MEGA_T pas dans (FIN-t0)
*/
//	Si MODE==t_CONTINUE alors    t = depart+i
//	Si MODE==t_PSEUDO_ALEA alors t = pseudo(grain, i)

#define t_CONTINUE        0 //t0+0,t0+1,t0+2,...
#define t_PSEUDO_ALEA     1 //rnd(),rnd(),rnd()...

#define t_MODE(MODE, GRAINE, t0, t1, t, mega_t, T, MEGA_T) (                       \
	MODE == t_CONTINUE    ? (t0 + t*MEGA_T + mega_t)  :                        \
	MODE == t_PSEUDO_ALEA ? (t0 + PSEUDO_ALEA((GRAINE+t)) % (t1-t0-MEGA_T)) : \
	NULL)

//	--- Calcule Cuda ---

#define MODE_NAIF     0
#define MODE_MAXIMALE 1

#define MODE_CALCULE MODE_MAXIMAL