#ifndef _AIS_
#define _AIS_

#include <vector>
#include "random.h"

namespace ais {


template<class S_type,class P_type>
class Dist_template // base class for distribution training
{
public:
	typedef S_type Sample_type; // class to store inputs to model
	typedef P_type Pars_type; // class to store parameters

	// Needed for P
	virtual double log_dist(Sample_type) = 0;
	// Needed for Q, if not generating sample just implement blank versions.
	virtual void generate_samples(std::vector<Sample_type>&, // Vector storing Nk spin-configs
								  std::vector<double>&,      // vector stroing Nk log_dist (Free-energy) for each spin-config
								  std::vector<Pars_type>&) = 0; // vector storing Nk gradients of log_dist (free-energy) w/ parameter for each spin-config
	// virtual Pars_type log_grad(Sample_type) = 0;
	virtual Pars_type get_pars(void) = 0; // extract current parameters of model
	virtual void put_pars(Pars_type) = 0; // replace current parameters of model 
};






template<class Dist>
class ais_KL
{
	Dist * P; // distribution which is the exact distribution
	Dist * Q; // distribution which is trying to model P.
	std::vector<typename Dist::Sample_type> Xk;
	std::vector<typename Dist::Pars_type> gradk;
	std::vector<double> wk,ck,logQk;
	rand_uniform uniform;


public:
	ais_KL(Dist * P_, Dist * Q_) : 
	Q(Q_), P(P_) {
		uniform = rand_uniform();
	}
	~ais_KL() {}
	//full_grad =  int dX P(X) grad(X) = int dX P(X)/Q(X) grad(X) Q(X) 
	void calc_weights(){ // calculate ration of P(x)/Q(x) for all samples
		const int Nk = Xk.size();

		for(int k=0;k<Nk;k++){
			const double logP = P->log_dist(Xk[k]);
			wk[k] = logP - logQk[k];
		}

		const double w_max = *std::max_element(wk.begin(),wk.end());

		double Zk = 0;
		for(int k=0;k<Nk;k++){
			Zk += std::exp(wk[k]-w_max);
		}

		wk[0] = std::exp(wk[0]-w_max)/Zk;
		ck[0] = wk[0];

		for(int k=1;k<Nk;k++){
			wk[k] = std::exp(wk[k]-w_max)/Zk;
			ck[k] = ck[k-1]+wk[k];
		}
	}

	void generate_samples(const int Nk){
		Xk.resize(Nk);
		wk.resize(Nk);
		ck.resize(Nk);
		logQk.resize(Nk);
		gradk.resize(Nk);

		Q->generate_samples(Xk,logQk,gradk);
		this->calc_weights();
	}

	void KL_grad(typename Dist::Pars_type &grad){
		const int Nk = Xk.size();

		const double a = 1.0/Nk;
		grad = gradk[0] * (a-wk[0]);

		for(int k=1;k<Nk;k++){
			grad = grad + gradk[k] * (a-wk[k]);
		}
	}

	template<class SGD>
	void train_Q(const size_t Niter,const int Nk,SGD &stepper){
		// Niter: number of SGD steps
		// Nk: batch size for evaluating gradient
		// stepper: implements the stepper for gradient decent. 
		typename Dist::Pars_type g,pars,step;
		pars = Q->get_pars();
		g = 0 * pars;
		step = 0 * pars;

		stepper.init(pars);

		for(size_t i=0;i<Niter;i++){
			this->generate_samples(Nk);		
			this->KL_grad(g);

			stepper.step(g,step);
			pars = pars + step;
			Q->put_pars(pars);
		}
	} 

	template<class SGD,class Callback>
	bool train_Q(const size_t Niter,const int Nk,SGD &stepper,Callback &callback){
		// Niter: number of SGD steps
		// Nk: batch size for evaluating gradient
		// stepper: implements the stepper for gradient decent. 

		typename Dist::Pars_type g,pars,step;
		pars = Q->get_pars();
		g = 0 * pars;
		step = 0 * pars;

		stepper.init(pars);

		for(size_t i=0;i<Niter;i++){
			this->generate_samples(Nk);		
			this->KL_grad(g);

			stepper.step(g,step);

			bool end = callback(i,pars,step,wk);

			pars = pars + step;

			Q->put_pars(pars);

			if(end){return true;}

		}
		return false;
	} 
};




}




#endif