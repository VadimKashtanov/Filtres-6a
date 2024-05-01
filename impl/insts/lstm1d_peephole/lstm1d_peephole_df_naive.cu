#include "lstm1d_peephole.cuh"

#define BLOQUE_T 16//32
#define BLOQUE_Y 16//32


static __global__ void d__kerd_lstm1d_peephole_naive___partie_h(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
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
		//y[(mega_t*T+0+_t)*_lstm_VARS + depart_h(X,Y) + _y] = _o * _ch;
		//
		float _dy = dy[(mega_t*T+0+_t)*_lstm_VARS + depart_h(X,Y) + _y];// = _o * _ch;
		//
		atomicAdd(&dy[(mega_t*T+0+_t)*_lstm_VARS + depart_o (X,Y) + _y], _ch*_dy);
		//dy[(mega_t*T+0+_t)*_lstm_VARS + depart_o (X,Y) + _y] /*=*/+= _ch*_dy;
		atomicAdd(&dy[(mega_t*T+0+_t)*_lstm_VARS + depart_ch(X,Y) + _y], _o*_dy);
		//dy[(mega_t*T+0+_t)*_lstm_VARS + depart_ch(X,Y) + _y] /*=*/+= _o*_dy;
	}
}

static __global__ void d__kerd_lstm1d_peephole_naive___partie_o(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	//	--- Partie och ---
	//o = logistique(sO = Ox@x + Oh@h + Oc@c + Ob)

	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;

	if (_t < T && _y < Y) {
		uint _lstm_VARS  = lstm_VARS (X,Y);
		uint _lstm_LOCDS = lstm_LOCDS(X,Y);
		//
		float _dy  =   dy[(mega_t*T+0+_t)*_lstm_VARS  + depart_o(X,Y)   + _y];
		float l_so = locd[(mega_t*T+0+_t)*_lstm_LOCDS + depart_dsO(X,Y) + _y];
		float ds   = _dy * l_so;

		//so += p[depart_poids_o(X,Y) + Ox(X,Y) + Oh(X,Y) + Oc(X,Y) + _y];
		atomicAdd(&dp[depart_poids_o(X,Y) + Ox(X,Y) + Oh(X,Y) + Oc(X,Y) + _y], ds);

		FOR(0, i_y, Y) {
			float _c = (mega_t==0 ? 0.0 : y[(mega_t    *T+0+_t)*_lstm_VARS + depart_c(X,Y) + i_y]);
			float _h = (mega_t==0 ? 0.0 : y[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_h(X,Y) + i_y]);
			//
			//so += _h * p[depart_poids_o(X,Y) + Ox(X,Y) + _y*Y + i_y];
			float d_h = p[depart_poids_o(X,Y) + Ox(X,Y) + _y*Y + i_y] * ds;
			atomicAdd(&dp[depart_poids_o(X,Y) + Ox(X,Y) + _y*Y + i_y], ds * _h);
			if (mega_t!=0) atomicAdd(&dy[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_h(X,Y) + i_y], d_h);
			//
			//so += _c * p[depart_poids_o(X,Y) + Ox(X,Y) + Oh(X,Y) + _y*Y + i_y];
			float d_c = p[depart_poids_o(X,Y) + Ox(X,Y) + Oh(X,Y) + _y*Y + i_y] * ds;
			atomicAdd(&dp[depart_poids_o(X,Y) + Ox(X,Y) + Oh(X,Y) + _y*Y + i_y], _c * ds);
			atomicAdd(&dy[(mega_t*T+0+_t)*_lstm_VARS + depart_c(X,Y) + i_y], d_c);
		}

		FOR(0, i_x, X) {	//x
			float _x = x[(mega_t*T+0+_t)*X_vars + DEPART_x + i_x];
			//so += _x * p[depart_poids_o(X,Y) + _y*X + i_x];
			float d_x = p[depart_poids_o(X,Y) + _y*X + i_x] * ds;
			atomicAdd(&dp[depart_poids_o(X,Y) + _y*X + i_x], _x*ds);
			atomicAdd(&dx[(mega_t*T+0+_t)*X_vars + DEPART_x + i_x], d_x);
		};
	}
}

static __global__ void d__kerd_lstm1d_peephole_naive___partie_cch(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	//	--- Partie c ---
	//c = f*c[-1] + i*u
	//ch = tanh(c)

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
		float _ch = lstmpeephole_activ_CH(_c);
		//
		float dch = dy[(mega_t*T+0+_t)*_lstm_VARS + depart_ch(X,Y) + _y];
		
		atomicAdd(&dy[(mega_t*T+0+_t)*_lstm_VARS + depart_c (X,Y) + _y], dch*d_lstmpeephole_activ_CH(_c,_ch));

		float dc = dy[(mega_t*T+0+_t)*_lstm_VARS + depart_c (X,Y) + _y];

		float d_f  = dc*_c1;
		float d_c1 = dc*_f;

		float d_i  = dc*_u;
		float d_u  = dc*_i;

		               atomicAdd(&dy[( mega_t   *T+0+_t)*_lstm_VARS + depart_f(X,Y) + _y], d_f);
		if (mega_t!=0) atomicAdd(&dy[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_c(X,Y) + _y], d_c1);
		               atomicAdd(&dy[( mega_t   *T+0+_t)*_lstm_VARS + depart_i(X,Y) + _y], d_i);
		               atomicAdd(&dy[( mega_t   *T+0+_t)*_lstm_VARS + depart_u(X,Y) + _y], d_u);
	}
}

static __global__ void d__kerd_lstm1d_peephole_naive___partie_fiu(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
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
		float lf = locd[(mega_t*T+0+_t)*_lstm_LOCDS + depart_dsF(X,Y) + _y];
		float li = locd[(mega_t*T+0+_t)*_lstm_LOCDS + depart_dsI(X,Y) + _y];
		float lu = locd[(mega_t*T+0+_t)*_lstm_LOCDS + depart_dsU(X,Y) + _y];
		//
		float df = dy[(mega_t*T+0+_t)*_lstm_VARS + depart_f(X,Y) + _y];
		float di = dy[(mega_t*T+0+_t)*_lstm_VARS + depart_i(X,Y) + _y];
		float du = dy[(mega_t*T+0+_t)*_lstm_VARS + depart_u(X,Y) + _y];
		//
		float dsf = lf * df;
		float dsi = li * di;
		float dsu = lu * du;
		//
		atomicAdd(&dp[depart_poids_f(X,Y) + Fx(X,Y) + Fh(X,Y) + Fc(X,Y) + _y], dsf);
		atomicAdd(&dp[depart_poids_i(X,Y) + Ix(X,Y) + Ih(X,Y) + Ic(X,Y) + _y], dsi);
		atomicAdd(&dp[depart_poids_u(X,Y) + Ux(X,Y) + Uh(X,Y) +         + _y], dsu);
		//
		FOR(0, i_y, Y) {	//h & c
			float _c = (mega_t==0 ? 0.0 : y[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_c(X,Y) + i_y]);
			float _h = (mega_t==0 ? 0.0 : y[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_h(X,Y) + i_y]);

			//	-------------------------------------------------
			
			//sf += _h * p[depart_poids_f(X,Y) + Fx(X,Y) + _y*Y + i_y];
			float dsf_h = dsf * p[depart_poids_f(X,Y) + Fx(X,Y) + _y*Y + i_y];
			atomicAdd(&dp[depart_poids_f(X,Y) + Fx(X,Y) + _y*Y + i_y], dsf * _h);
			if (mega_t!=0) atomicAdd(&dy[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_h(X,Y) + i_y], dsf_h);

			//si += _h * p[depart_poids_i(X,Y) + Ix(X,Y) + _y*Y + i_y];
			float dsi_h = dsi * p[depart_poids_i(X,Y) + Ix(X,Y) + _y*Y + i_y];
			atomicAdd(&dp[depart_poids_i(X,Y) + Ix(X,Y) + _y*Y + i_y], dsi * _h);
			if (mega_t!=0) atomicAdd(&dy[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_h(X,Y) + i_y], dsi_h);
			
			//su += _h * p[depart_poids_u(X,Y) + Ux(X,Y) + _y*Y + i_y];
			float dsu_h = dsu * p[depart_poids_u(X,Y) + Ux(X,Y) + _y*Y + i_y];
			atomicAdd(&dp[depart_poids_u(X,Y) + Ux(X,Y) + _y*Y + i_y], dsu * _h);
			if (mega_t!=0) atomicAdd(&dy[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_h(X,Y) + i_y], dsu_h);
			
			//	-------------------------------------------------

			//sf += _c * p[depart_poids_f(X,Y) + Fx(X,Y) + Fh(X,Y) + _y*Y + i_y];
			float dsf_c = dsf * p[depart_poids_f(X,Y) + Fx(X,Y) + Fh(X,Y) + _y*Y + i_y];
			atomicAdd(&dp[depart_poids_f(X,Y) + Fx(X,Y) + Fh(X,Y) + _y*Y + i_y], dsf * _c);
			if (mega_t!=0) atomicAdd(&dy[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_c(X,Y) + i_y], dsf_c);
			
			//si += _c * p[depart_poids_i(X,Y) + Ix(X,Y) + Ih(X,Y) + _y*Y + i_y];
			float dsi_c = dsi * p[depart_poids_i(X,Y) + Ix(X,Y) + Ih(X,Y) + _y*Y + i_y];
			atomicAdd(&dp[depart_poids_i(X,Y) + Ix(X,Y) + Ih(X,Y) + _y*Y + i_y], dsi * _c);
			if (mega_t!=0) atomicAdd(&dy[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_c(X,Y) + i_y], dsi_c);
			
			//su += _c * p[depart_poids_u(X,Y) + Ux(X,Y) + Uh(X,Y) + _y*Y + i_y];
			//float dsu_c = dsu * p[depart_poids_u(X,Y) + Ux(X,Y) + Uh(X,Y) + _y*Y + i_y];
			//atomicAdd(&dp[depart_poids_u(X,Y) + Ux(X,Y) + Uh(X,Y) + _y*Y + i_y], dsu * _c);
			//if (mega_t!=0) atomicAdd(&dy[((mega_t-1)*T+0+_t)*_lstm_VARS + depart_c(X,Y) + i_y], dsu_c);
		};

		FOR(0, i_x, X) {	//x
			float _x = x[(mega_t*T+0+_t)*X_vars + DEPART_x + i_x];
			float _dx = 0;
			//sf += _x * p[depart_poids_f(X,Y) + _y*X + i_x];
			_dx += dsf * p[depart_poids_f(X,Y) + _y*X + i_x];
			atomicAdd(&dp[depart_poids_f(X,Y) + _y*X + i_x], _x * dsf);

			//si += _x * p[depart_poids_i(X,Y) + _y*X + i_x];
			_dx += dsi * p[depart_poids_i(X,Y) + _y*X + i_x];
			atomicAdd(&dp[depart_poids_i(X,Y) + _y*X + i_x], _x * dsi);

			//su += _x * p[depart_poids_u(X,Y) + _y*X + i_x];
			_dx += dsu * p[depart_poids_u(X,Y) + _y*X + i_x];
			atomicAdd(&dp[depart_poids_u(X,Y) + _y*X + i_x], _x * dsu);

			//dx
			atomicAdd(&dx[(mega_t*T+0+_t)*X_vars + DEPART_x + i_x], _dx);
		};
	}
};

void d_nvidia_lstm1d_peephole_naive(
	uint mega_t,
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	//	--- DERIVE Partie h ---
	d__kerd_lstm1d_peephole_naive___partie_h<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();

	//	--- DERIVE Partie och ---
	d__kerd_lstm1d_peephole_naive___partie_o<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();

	//	--- DERIVE Partie c ---
	d__kerd_lstm1d_peephole_naive___partie_cch<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();

	//	--- DERIVE Partie fiu ---
	d__kerd_lstm1d_peephole_naive___partie_fiu<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		mega_t,
		_t_MODE, GRAINE,
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();
};