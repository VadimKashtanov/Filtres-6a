#include "lstm1d_peephole.cuh"

#define BLOQUE_T 16//32
#define BLOQUE_Y 16//32

/*
//	--- Partie fiu ---
f = logistique(sF = Fx@x + Fh@h + Fc@c[-1] + Fb)
i = logistique(sI = Ix@x + Ih@h + Ic@c[-1] + Ib)
u =       tanh(sU = Ux@x + Uh@h +          + Ub)
//	--- Partie cch ---
c = f*c[-1] + i*u
ch = tanh(c)
//	--- Partie o ---
o = logistique(sO = Ox@x + Oh@h + Oc@c    + Ob)
//	--- Partie h ---
h = o * ch
*/

static __global__ void kerd_lstm1d_peephole_naive___partie_fiu(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	//	--- Partie fiu ---
	//f = logistique(sF = Fx@x + Fh@h + Fc@c[-1] + Fb)
	//i = logistique(sI = Ix@x + Ih@h + Ic@c[-1] + Ib)
	//u =       tanh(sU = Ux@x + Uh@h +          + Ub)

	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		uint _lstm_VARS  = lstm_VARS (X,Y);
		uint _lstm_LOCDS = lstm_LOCDS(X,Y);
		//
		float sf = 0;
		float si = 0;
		float su = 0;
		//
		FOR(0, i_x, X) {	//x
			float _x = x[(mega_t*T+0+_t)*X_vars + DEPART_x + i_x];
			sf += _x * p[depart_poids_f(X,Y) + _y*X + i_x];
			si += _x * p[depart_poids_i(X,Y) + _y*X + i_x];
			su += _x * p[depart_poids_u(X,Y) + _y*X + i_x];
		};
		FOR(0, i_y, Y) {	//h & c
			float _c = (mega_t==0 ? 0.0 : y[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_c(X,Y) + i_y]);
			float _h = (mega_t==0 ? 0.0 : y[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_h(X,Y) + i_y]);
			//
			sf += _h * p[depart_poids_f(X,Y) + Fx(X,Y) + _y*Y + i_y];
			si += _h * p[depart_poids_i(X,Y) + Ix(X,Y) + _y*Y + i_y];
			su += _h * p[depart_poids_u(X,Y) + Ux(X,Y) + _y*Y + i_y];
			//
			sf += _c * p[depart_poids_f(X,Y) + Fx(X,Y) + Fh(X,Y) + _y*Y + i_y];
			si += _c * p[depart_poids_i(X,Y) + Ix(X,Y) + Ih(X,Y) + _y*Y + i_y];
			//su += _c * p[depart_poids_u(X,Y) + Ux(X,Y) + Uh(X,Y) + _y*Y + i_y];
		};
		//
		sf += p[depart_poids_f(X,Y) + Fx(X,Y) + Fh(X,Y) + Fc(X,Y) + _y];
		si += p[depart_poids_i(X,Y) + Ix(X,Y) + Ih(X,Y) + Ic(X,Y) + _y];
		su += p[depart_poids_u(X,Y) + Ux(X,Y) + Uh(X,Y) +         + _y];
		//
		float a_sf = logistic(sf);
		float a_si = logistic(si);
		float a_su =     tanh(su);
		//
		y[(mega_t*T+0+_t)*_lstm_VARS + depart_f(X,Y) + _y] = a_sf;
		y[(mega_t*T+0+_t)*_lstm_VARS + depart_i(X,Y) + _y] = a_si;
		y[(mega_t*T+0+_t)*_lstm_VARS + depart_u(X,Y) + _y] = a_su;
		//
		locd[(mega_t*T+0+_t)*_lstm_LOCDS + depart_dsF(X,Y) + _y] = d_logistic(sf,a_sf);
		locd[(mega_t*T+0+_t)*_lstm_LOCDS + depart_dsI(X,Y) + _y] = d_logistic(si,a_si);
		locd[(mega_t*T+0+_t)*_lstm_LOCDS + depart_dsU(X,Y) + _y] =     d_tanh(su,a_su);
	}
};

static __global__ void kerd_lstm1d_peephole_naive___partie_cch(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	//	--- Partie c ---
	//c = f*c[-1] + i*u

	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		uint _lstm_VARS  = lstm_VARS (X,Y);
		uint _lstm_LOCDS = lstm_LOCDS(X,Y);
		//
		float _f  =                    y[( mega_t   *T+0+_t)*_lstm_VARS + depart_f(X,Y) + _y];
		float _c1 = (mega_t==0 ? 0.0 : y[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_c(X,Y) + _y]);
		float _i  =                    y[( mega_t   *T+0+_t)*_lstm_VARS + depart_i(X,Y) + _y];
		float _u  =                    y[( mega_t   *T+0+_t)*_lstm_VARS + depart_u(X,Y) + _y];
		//
		float _c = _f*_c1 + _i*_u;
		//
		y[(mega_t*T+0+_t)*_lstm_VARS + depart_c (X,Y) + _y] = _c;
		y[(mega_t*T+0+_t)*_lstm_VARS + depart_ch(X,Y) + _y] = lstmpeephole_activ_CH(_c);
	}
}

static __global__ void kerd_lstm1d_peephole_naive___partie_o(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	//	--- Partie och ---
	//o = logistique(sO = Ox@x + Oh@h + Oc@c + Ob)

	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		uint _lstm_VARS  = lstm_VARS (X,Y);
		uint _lstm_LOCDS = lstm_LOCDS(X,Y);
		//
		float so = 0;
		//
		FOR(0, i_x, X) {	//x
			float _x = x[(mega_t*T+0+_t)*X_vars + DEPART_x + i_x];
			so += _x * p[depart_poids_o(X,Y) + _y*X + i_x];
		};
		FOR(0, i_y, Y) {	//h & c
			float _c = (mega_t==0 ? 0.0 : y[(mega_t    *T+0+_t)*_lstm_VARS + depart_c(X,Y) + i_y]);
			float _h = (mega_t==0 ? 0.0 : y[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_h(X,Y) + i_y]);
			//
			so += _h * p[depart_poids_o(X,Y) + Ox(X,Y)           + _y*Y + i_y];
			//
			so += _c * p[depart_poids_o(X,Y) + Ox(X,Y) + Oh(X,Y) + _y*Y + i_y];
		};
		//
		so += p[depart_poids_o(X,Y) + Ox(X,Y) + Oh(X,Y) + Oc(X,Y) + _y];
		//
		float a_so = logistic(so);
		//
		   y[(mega_t*T+0+_t)*_lstm_VARS  + depart_o(X,Y)   + _y] = a_so;
		//
		locd[(mega_t*T+0+_t)*_lstm_LOCDS + depart_dsO(X,Y) + _y] = d_logistic(so,a_so);
	}
}

static __global__ void kerd_lstm1d_peephole_naive___partie_h(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	//	--- Partie h ---
	//h = o * ch

	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		uint _lstm_VARS  = lstm_VARS (X,Y);
		uint _lstm_LOCDS = lstm_LOCDS(X,Y);
		//
		float _o  = y[(mega_t*T+0+_t)*_lstm_VARS + depart_o (X,Y) + _y];
		float _ch = y[(mega_t*T+0+_t)*_lstm_VARS + depart_ch(X,Y) + _y];
		//
		y[(mega_t*T+0+_t)*_lstm_VARS + depart_h(X,Y) + _y] = _o * _ch;
	}
}

void nvidia_lstm1d_peephole_naive(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	//	--- Partie fiu ---
	kerd_lstm1d_peephole_naive___partie_fiu<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
	//	--- Partie c ---
	kerd_lstm1d_peephole_naive___partie_cch<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
	//	--- Partie och ---
	kerd_lstm1d_peephole_naive___partie_o<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
	//	--- Partie h ---
	kerd_lstm1d_peephole_naive___partie_h<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
}