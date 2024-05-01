#pragma once

#include "marchee.cuh"

#include "S.cuh"

#define L_DEPART     142
#define L_FIN        151
#define COMMENTAIRES (2)

#define             C ((1+L_FIN-L_DEPART-COMMENTAIRES))
#define         MAX_Y 4096
#define       BLOQUES 256
#define F_PAR_BLOQUES 8

#define      INSTS 4

#define    FILTRES_PRIXS 0
#define       DOT1D_TANH 1
#define  LSTM1D_PEEPHOLE 2
#define DOT1D_TANH_ELMAN 3
#define            GRU1D 4

uint * UNIFORME_C(uint x);

// inst0 - "filtre_prixs"	// defaut. 0 pour bcp de fonctions 
// inst1 - "dot1d"
// inst2 - "dot1d bloque"

// mdl.quelconque  [        inst0            ][     inst1     ][ inst2 ][                   inst3                ]  T fois a la suite
//   ESPACE = somme( inst[i] )
// inst[i] :
//    [                   inst[i]                   ]
//    [           AUTRES           ][    SORTIES    ]
//
//	apres chaque optimisation les espaces doivent etre reinitialisés.
//		filtre a par exemple besoin de normaliser (mais on va pas le faire, juste pour voire)

typedef struct {
	//
	uint T;

	//	Filtre prixs
	ema_int_t * bloque[BLOQUES];

	//
	uint intervalles[BLOQUES];
	uint * intervalles__d;

	uint  type_de_norme[BLOQUES];
	float min_theorique[BLOQUES], max_theorique[BLOQUES];

	//	c'est bien * PRIXS
	//
	float          normalisee[BLOQUES * PRIXS * N_FLTR];
	float      dif_normalisee[BLOQUES * PRIXS * N_FLTR];
	//
	float     * normalisee__d;
	float * dif_normalisee__d;

	//
	uint insts[C];	//ASSERT(inst[0] == 0) && ASSERT(inst[i>0] != FILTRES_PRIXS)
	uint     Y[C];	//ASSERT(Y[i] < MAX_Y) && ASSERT(Y[C-1] == P)
	//
	uint inst_POIDS  [C];
	uint inst_VARS   [C];
	uint inst_LOCDS  [C];	//	infos de f(x) a ne pas re-calculer pendant df(x)
	uint inst_SORTIES[C];
	uint inst_DEPART_SORTIE[C];
	//
	uint inst_p_separateurs[C];
	char** char_inst_p_separateurs[C];
	uint* depart_inst_p_separateurs[C];

	uint total_POIDS;


	//	* mdl->T
	//
	float *  p[C];

	//	* mdl->T
	//
	float *  p__d[C];
	float *  y__d[C];
	float *  l__d[C];
	float * dy__d[C];
	float * dp__d[C];
} Mdl_t;

//	Memoire ram & vram
typedef void (*mdl_inst_f)(Mdl_t * mdl, uint inst);
extern mdl_inst_f cree_inst[INSTS];
//
Mdl_t * cree_mdl(
	uint T,
	uint Y[C], uint inst[C],
	ema_int_t * bloques[BLOQUES]
);
void mdl_verif(Mdl_t * mdl);
//
void enregistrer_les_lignes_brute(Mdl_t * mdl, char * fichier);
//
void        liberer_mdl(Mdl_t * mdl);
//
void       mdl_gpu_vers_cpu(Mdl_t * mdl);
void       mdl_cpu_vers_gpu(Mdl_t * mdl);
void mdl_poids_gpu_vers_cpu(Mdl_t * mdl);
void mdl_poids_cpu_vers_gpu(Mdl_t * mdl);
//
void       mdl_zero_gpu(Mdl_t * mdl);
//
void mdl_zero_deriv_gpu(Mdl_t * mdl, uint zeroiser[C]);
//
void mdl_normer_les_filtres(Mdl_t * mdl);
//
void mdl_changer_couche_Y(Mdl_t * mdl, uint c, uint nouveau_Y);
void mdl_re_cree_poids(Mdl_t * mdl);

//	Regulariser
extern mdl_inst_f regulariser_inst[INSTS];

//	Perturbations
void perturber_filtres(Mdl_t * mdl, uint L);
void perturber        (Mdl_t * mdl, uint L);

//	I/O
Mdl_t * ouvrire_mdl(uint T,      char * fichier);
void     ecrire_mdl(Mdl_t * mdl, char * fichier);

//	Plume
extern char * nom_inst[INSTS];
extern mdl_inst_f plume_inst[INSTS];
void plumer_mdl        (Mdl_t * mdl);
void plume_separateur_p(Mdl_t * mdl, uint c, uint p);
//
void  comportement   (Mdl_t * mdl, uint t0, uint t1);
void  mdl_plume_poids(Mdl_t * mdl);
void  mdl_plume_grad (Mdl_t * mdl, uint t0, uint t1, uint _t_MODE, uint GRAINE);
float mdl_moy_dp     (Mdl_t * mdl, uint c);

//	F & F'
typedef void (*mdl_f_f)(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE, uint mega_t);
extern mdl_f_f inst_f [INSTS];
extern mdl_f_f inst_df[INSTS];
//
void  mdl_f(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE);
void mdl_df(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE);

//	Utilisation
float mdl_score(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE);
float mdl_pred(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE);
//
void mdl_aller_retour(Mdl_t * mdl, uint t0, uint t1, uint mode, uint _t_MODE, uint GRAINE);
//
float mdl_les_gains(Mdl_t * mdl, uint t0, uint t1, uint mode, float GRAND_COEF, uint _t_MODE, uint GRAINE);