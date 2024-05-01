#pragma once

#include "mdl.cuh"

typedef struct {
	float score;
	float pred;
	float les_gains__2;
	float les_gains__4;
	float les_gains__8;
} Stats_t;

Stats_t * statistiques(Mdl_t * mdl, uint t0, uint t1);