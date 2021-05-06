#ifndef _optimize_
#define _optimize_

#include <cmath>


namespace sgd {


// need to implement element-wise sqrt to use ADAM
template<class Pars_type>
Pars_type sqrt(const Pars_type &v){
	return v.sqrt();
}


template<>
double sqrt<double>(const double &v){
	return std::sqrt(v);
}

struct identity_func
{
	identity_func() {}

	inline double operator()(const int t){return 1;}
};



template<class Pars_type,class Func=identity_func>
class adam
{
	Pars_type m,v;
	const double beta1,beta2,eps,alpha;
	double beta1t,beta2t;
	size_t t;
	Func f;
public:
	adam():  beta1(0.9), beta2(0.999), eps(1e-8), alpha(0.001), f(identity_func()) {}
	adam(const double alpha_):  beta1(0.9), beta2(0.999), eps(1e-8), alpha(alpha_), f(identity_func()) {}
	adam(const double alpha_,Func f_): beta1(0.9), beta2(0.999), eps(1e-8), alpha(alpha_), f(f_) {}
	~adam() {}

	void init(Pars_type &pars){
		m = 0*pars;
		v = 0*pars;
		beta2t = beta1t = 1;
		t = 0;

	}

	void step(Pars_type &g,Pars_type &step){
		t++;
		beta1t *= beta1;
		beta2t *= beta2;
		const double alphat = f(t)*alpha*std::sqrt(1-beta2t)/(1-beta1t);
	
		m = beta1 * m + (1 - beta1) * g;
		v = beta2 * v + (1 - beta2) * g * g;
		step = - alphat * m / (sqrt(v) + eps) ;
	}

};

}









#endif