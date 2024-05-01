#include "main.cuh"

void ecrire_structure_generale(char * file) {
	FILE * fp = fopen(file, "wb");
	//
	uint CONSTATES = 14;
	uint constantes[CONSTATES] = {
		DEPART,
		N,
		MAX_INTERVALLE,
		SOURCES,
		MAX_PARAMS, NATURES,
		MAX_EMA, MAX_PLUS, MAX_COEF_MACD,
		C, MAX_Y, BLOQUES, F_PAR_BLOQUES,
		INSTS
	};
	//
	FWRITE(&CONSTATES, sizeof(uint), 1, fp);
	FWRITE(constantes, sizeof(uint), 18, fp);
	//
	FWRITE(min_param,  sizeof(uint), NATURES*MAX_PARAMS, fp);
	FWRITE(max_param,  sizeof(uint), NATURES*MAX_PARAMS, fp);
	//
	FWRITE(NATURE_PARAMS,  sizeof(uint), NATURES, fp);
	//
	fclose(fp);
};