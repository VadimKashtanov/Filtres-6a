#include "lstm1d_peephole.cuh"

#define BLOQUE_T 16
#define BLOQUE_Y 16

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

static void lstm_peephole_f(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * l)
{
	float * x0=x; uint X0_vars=X_vars;         uint X0=X; uint depart_x0=DEPART_x;      uint x0_depart__t=mega_t*T;
	float * x1=y; uint X1_vars=lstm_VARS(X,Y); uint X1=Y; uint depart_x1=depart_h(X,Y); uint x1_depart__t=(mega_t-1)*T;
	float * x2=y; uint X2_vars=lstm_VARS(X,Y); uint X2=Y; uint depart_x2=depart_c(X,Y); uint x2_depart__t=(mega_t-1)*T;
	//
	uint Y__vars=lstm_VARS (X,Y); uint depart__y=depart_f  (X,Y); uint y__depart__t=mega_t*T;
	uint L__vars=lstm_LOCDS(X,Y); uint depart__l=depart_dsF(X,Y);
	//
	float * a0=p; uint depart_a0=depart_poids_f(X,Y);
	float * a1=p; uint depart_a1=depart_poids_f(X,Y)+Fx(X,Y);
	float * a2=p; uint depart_a2=depart_poids_f(X,Y)+Fx(X,Y)+Fh(X,Y);
	float * b =p; uint depart__b=depart_poids_f(X,Y)+Fx(X,Y)+Fh(X,Y)+Fc(X,Y);
	//
	uint activation = cuda_math_LOGISTIC;
	//
	if (mega_t == 0) {
		//f = logistique(sF = Fx@x + Fh@h +          + Fb)
		nvidia_F_AX__shared_16(
			T,			//KERD(T)
			//
			x0, X0_vars, X0, depart_x0, x0_depart__t,
			//
			y, Y__vars, Y, depart__y, y__depart__t,
			l, L__vars,    depart__l,
			//
			a0, depart_a0,
			b , depart__b,
			//
			activation);
	} else {
		//f = logistique(sF = Fx@x + Fh@h + Fc@c[-1] + Fb)
		nvidia_F_AX_AX_AX__shared_16(
			T,			//KERD(T)
			//
			x0, X0_vars, X0, depart_x0, x0_depart__t,
			x1, X1_vars, X1, depart_x1, x1_depart__t,
			x2, X2_vars, X2, depart_x2, x2_depart__t,
			//
			y, Y__vars, Y, depart__y, y__depart__t,
			l, L__vars,    depart__l,
			//
			a0, depart_a0,
			a1, depart_a1,
			a2, depart_a2,
			b , depart__b,
			//
			activation);
	}
}

static void lstm_peephole_i(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * l)
{
	float * x0=x; uint X0_vars=X_vars;         uint X0=X; uint depart_x0=DEPART_x;      uint x0_depart__t=mega_t*T;
	float * x1=y; uint X1_vars=lstm_VARS(X,Y); uint X1=Y; uint depart_x1=depart_h(X,Y); uint x1_depart__t=(mega_t-1)*T;
	float * x2=y; uint X2_vars=lstm_VARS(X,Y); uint X2=Y; uint depart_x2=depart_c(X,Y); uint x2_depart__t=(mega_t-1)*T;
	//
	uint Y__vars=lstm_VARS (X,Y); uint depart__y=depart_i  (X,Y); uint y__depart__t=mega_t*T;
	uint L__vars=lstm_LOCDS(X,Y); uint depart__l=depart_dsI(X,Y);
	//
	float * a0=p; uint depart_a0=depart_poids_i(X,Y);
	float * a1=p; uint depart_a1=depart_poids_i(X,Y)+Ix(X,Y);
	float * a2=p; uint depart_a2=depart_poids_i(X,Y)+Ix(X,Y)+Ih(X,Y);
	float * b =p; uint depart__b=depart_poids_i(X,Y)+Ix(X,Y)+Ih(X,Y)+Ic(X,Y);
	//
	uint activation = cuda_math_LOGISTIC;
	//
	if (mega_t == 0) {
		//f = logistique(sF = Fx@x + Fh@h +          + Fb)
		nvidia_F_AX__shared_16(
			T,			//KERD(T)
			//
			x0, X0_vars, X0, depart_x0, x0_depart__t,
			//
			y, Y__vars, Y, depart__y, y__depart__t,
			l, L__vars,    depart__l,
			//
			a0, depart_a0,
			b , depart__b,
			//
			activation);
	} else {
		//f = logistique(sF = Fx@x + Fh@h + Fc@c[-1] + Fb)
		nvidia_F_AX_AX_AX__shared_16(
			T,			//KERD(T)
			//
			x0, X0_vars, X0, depart_x0, x0_depart__t,
			x1, X1_vars, X1, depart_x1, x1_depart__t,
			x2, X2_vars, X2, depart_x2, x2_depart__t,
			//
			y, Y__vars, Y, depart__y, y__depart__t,
			l, L__vars,    depart__l,
			//
			a0, depart_a0,
			a1, depart_a1,
			a2, depart_a2,
			b , depart__b,
			//
			activation);
	}
}

static void lstm_peephole_u(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * l)
{
	float * x0=x; uint X0_vars=X_vars;         uint X0=X; uint depart_x0=DEPART_x;      uint x0_depart__t=mega_t*T;
	float * x1=y; uint X1_vars=lstm_VARS(X,Y); uint X1=Y; uint depart_x1=depart_h(X,Y); uint x1_depart__t=(mega_t-1)*T;
	//
	uint Y__vars=lstm_VARS (X,Y); uint depart__y=depart_u  (X,Y); uint y__depart__t=mega_t*T;
	uint L__vars=lstm_LOCDS(X,Y); uint depart__l=depart_dsU(X,Y);
	//
	float * a0=p; uint depart_a0=depart_poids_u(X,Y);
	float * a1=p; uint depart_a1=depart_poids_u(X,Y)+Ux(X,Y);
	float * b =p; uint depart__b=depart_poids_u(X,Y)+Ux(X,Y)+Uh(X,Y);
	//
	uint activation = cuda_math_TANH;
	//
	if (mega_t == 0) {
		//f = logistique(sF = Fx@x + Fh@h +          + Fb)
		nvidia_F_AX__shared_16(
			T,			//KERD(T)
			//
			x0, X0_vars, X0, depart_x0, x0_depart__t,
			//
			y, Y__vars, Y, depart__y, y__depart__t,
			l, L__vars,    depart__l,
			//
			a0, depart_a0,
			b , depart__b,
			//
			activation);
	} else {
		//f = logistique(sF = Fx@x + Fh@h + Fc@c[-1] + Fb)
		nvidia_F_AX_AX__shared_16(
			T,			//KERD(T)
			//
			x0, X0_vars, X0, depart_x0, x0_depart__t,
			x1, X1_vars, X1, depart_x1, x1_depart__t,
			//
			y, Y__vars, Y, depart__y, y__depart__t,
			l, L__vars,    depart__l,
			//
			a0, depart_a0,
			a1, depart_a1,
			b , depart__b,
			//
			activation);
	}
}

static __global__ void kerd_lstm1d_peephole_naive___partie_cch(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * l)
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
		y[(mega_t*T+_t)*_lstm_VARS + depart_c (X,Y) + _y] = _c;
		y[(mega_t*T+_t)*_lstm_VARS + depart_ch(X,Y) + _y] = lstmpeephole_activ_CH(_c);
	}
}

static void lstm_peephole_o(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * l)
{
	float * x0=x; uint X0_vars=X_vars;         uint X0=X; uint depart_x0=DEPART_x;      uint x0_depart__t=mega_t*T;
	float * x1=y; uint X1_vars=lstm_VARS(X,Y); uint X1=Y; uint depart_x1=depart_h(X,Y); uint x1_depart__t=(mega_t-1)*T;
	float * x2=y; uint X2_vars=lstm_VARS(X,Y); uint X2=Y; uint depart_x2=depart_c(X,Y); uint x2_depart__t=mega_t*T;
	//
	uint Y__vars=lstm_VARS (X,Y); uint depart__y=depart_o  (X,Y); uint y__depart__t=mega_t*T;
	uint L__vars=lstm_LOCDS(X,Y); uint depart__l=depart_dsO(X,Y);
	//
	float * a0=p; uint depart_a0=depart_poids_o(X,Y);
	float * a1=p; uint depart_a1=depart_poids_o(X,Y)+Ox(X,Y);
	float * a2=p; uint depart_a2=depart_poids_o(X,Y)+Ox(X,Y)+Oh(X,Y);
	float * b =p; uint depart__b=depart_poids_o(X,Y)+Ox(X,Y)+Oh(X,Y)+Oc(X,Y);
	//
	uint activation = cuda_math_LOGISTIC;
	//
	if (mega_t == 0) {
		nvidia_F_AX_AX__shared_16(
			T,			//KERD(T)
			//
			x0, X0_vars, X0, depart_x0, x0_depart__t,
			x2, X2_vars, X2, depart_x2, x2_depart__t,
			//
			y, Y__vars, Y, depart__y, y__depart__t,
			l, L__vars,    depart__l,
			//
			a0, depart_a0,
			a2, depart_a2,
			b , depart__b,
			//
			activation);
	} else {
		nvidia_F_AX_AX_AX__shared_16(
			T,			//KERD(T)
			//
			x0, X0_vars, X0, depart_x0, x0_depart__t,
			x1, X1_vars, X1, depart_x1, x1_depart__t,
			x2, X2_vars, X2, depart_x2, x2_depart__t,
			//
			y, Y__vars, Y, depart__y, y__depart__t,
			l, L__vars,    depart__l,
			//
			a0, depart_a0,
			a1, depart_a1,
			a2, depart_a2,
			b , depart__b,
			//
			activation);
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
	float * l)
{
	//	--- Partie h ---
	//h = o * ch

	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		uint _lstm_VARS  = lstm_VARS (X,Y);
		uint _lstm_LOCDS = lstm_LOCDS(X,Y);
		//
		float _o  = y[(mega_t*T+_t)*_lstm_VARS + depart_o (X,Y) + _y];
		float _ch = y[(mega_t*T+_t)*_lstm_VARS + depart_ch(X,Y) + _y];
		//
		y[(mega_t*T+0+_t)*_lstm_VARS + depart_h(X,Y) + _y] = _o * _ch;
	}
}

void nvidia_lstm1d_peephole_shared_16_2(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * l)
{
	//f = logistique(sF = Fx@x + Fh@h + Fc@c[-1] + Fb)
	//i = logistique(sI = Ix@x + Ih@h + Ic@c[-1] + Ib)
	//u =       tanh(sU = Ux@x + Uh@h +          + Ub)
	lstm_peephole_f(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		l); //mega_t==0 => df=0
	lstm_peephole_i(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		l);
	lstm_peephole_u(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		l);
	ATTENDRE_CUDA();


	//c = f*c[-1] + i*u
	//ch = tanh(c)
	kerd_lstm1d_peephole_naive___partie_cch<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		l);
	ATTENDRE_CUDA();


	//	o = logistique(sO = Ox@x + Oh@h + Oc@c     + Ob)
	lstm_peephole_o(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		l);
	ATTENDRE_CUDA();


	//h = o * ch
	kerd_lstm1d_peephole_naive___partie_h<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		l);
	ATTENDRE_CUDA();
}