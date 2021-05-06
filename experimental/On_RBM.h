#ifndef _On_RBM_
#define _On_RBM_

#include <iostream>
#include <Eigen/Dense>
#include "random.h"
#include "ais.h"


typedef Eigen::Array<signed char,Eigen::Dynamic,1> Array1c;
typedef Eigen::Array<double,Eigen::Dynamic,1> Array1d;
typedef Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Array2d;


template<int Nstep=1,int warmup=1000>
class RBM_ising : public ais::Dist_template<Array1c,Array2d> 
{

	Pars_type Wab;
	Array1c h,v;
	const int Nv,Nh;
	rand_uniform ran_u;
	rand_normal ran_n;


public:
	RBM_ising(const int Nv_,const int Nh_): Nv(Nv_), Nh(Nh_) {

		ran_u = rand_uniform();
		ran_n = rand_normal();
		Wab = Pars_type::Zero(Nh_+1,Nv_+1);
		h = Array1c::Zero(Nh_);
		v = Array1c::Zero(Nv_);
		for(int i=0;i<Nh_;i++){
			h[i] = (signed char)(std::floor(2*ran_u())-1);
		}

		for(int i=0;i<Nv_;i++){
			v[i] = (signed char)(std::floor(2*ran_u())-1);
		}

		for(int i=0;i<Nh_+1;i++){
			for(int j=0;j<Nv_+1;j++){
				Wab(i,j) = 0.001*ran_n();
			}
		}
		this->MC_sweep(warmup);

	}


	RBM_ising(const int Nv_): Nv(Nv_), Nh(Nv_) {

		ran_u = rand_uniform();
		ran_n = rand_normal();
		Wab = Pars_type::Zero(Nv_+1,Nv_+1);
		h = Array1c::Zero(Nv_);
		v = Array1c::Zero(Nv_);

		for(int i=0;i<Nv_+1;i++){
			if(i<Nv_){
				h[i] = (signed char)(std::floor(2*ran_u())-1);
				v[i] = (signed char)(std::floor(2*ran_u())-1);				
			}

			for(int j=0;j<Nv_+1;j++){
				Wab(i,j) = 0.001*ran_n();
			}
		}
		this->MC_sweep(warmup);

	}


	~RBM_ising() {}

	void MC_sweep(const int Niter){

		for(int it=0;it<Niter;it++){

			for(int i=0;i<Nh;i++){
				double beta = Wab(i,Nv);

				for(int j=0;j<Nv;j++){
					beta += Wab(i,j)*v[j];
				}

				const double p = 1.0/(1.0+std::exp(-2*beta));

				h[i] = (ran_u() < p ? 1 : -1);
			}

			for(int j=0;j<Nv;j++){
				double beta = Wab(Nh,j);

				for(int i=0;i<Nh;i++){
					beta += Wab(i,j)*h[i];
				}

				const double p = 1.0/(1.0+std::exp(-2*beta));

				v[j] = (ran_u() < p ? 1 : -1);
			}

		}
	}

	void generate_samples(std::vector<Sample_type> &Xs){
		const int Nsample = Xs.size();

		for(int s=0;s<Nsample;s++){
			Xs[s] = v;
			this->MC_sweep(Nstep);
		}
	}

	void generate_samples(std::vector<Sample_type> &Xs,std::vector<double> &logQs){
		const int Nsample = Xs.size();

		for(int s=0;s<Nsample;s++){
			Xs[s] = v;
			logQs[s] = this->log_dist(v);
			this->MC_sweep(Nstep);
		}
	}

	double log_dist(Sample_type X){
		double logQ = Nh*std::log(2);
		
		for(int i=0;i<Nh;i++){
			double beta = Wab(i,Nv);

			for(int j=0;j<Nv;j++){
				beta += Wab(i,j)*X[j];
			}

			logQ += std::cosh(beta);
		}

		for(int j=0;j<Nv;j++){
			logQ += Wab(Nh,j) * X[j];
		}

		return logQ;
	}

	Pars_type log_grad(Sample_type X){
		Pars_type grad = 0*Wab;

		for(int j=0;j<Nv;j++){
			grad(Nh,j) = (double)X[j];
		}

		for(int i=0;i<Nh;i++){
			double beta = Wab(i,Nv);
			for(int j=0;j<Nv;j++){
				beta += Wab(i,j) * X[j];
			}

			const double h = std::sinh(beta);

			for(int j=0;j<Nv;j++){
				grad(i,j) = X[j] * h;
			}

			grad(i,Nv) = h;
		}

		return grad;
	}
	Pars_type get_pars(void){return Wab;}
	void put_pars(Pars_type in){Wab = in;}
};




/*


double C_exp(const double y,const double a){

	static const double taylor_0_bound = std::numeric_limits<double>::epsilon();
	static const double taylor_2_bound = std::sqrt(taylor_0_bound); // second order
	static const double taylor_n_bound = std::sqrt(taylor_2_bound); // fourth order

	if(std::abs(a) > taylor_n_bound){
		if(a > 0){
			const double e = std::exp(-2.0*a);

			return (std::exp(a*(y-1.0)) - e) / (1.0 - e);
		}
		else{
			return (1.0 - std::exp(a*(1.0-y))) / (1 - std::exp(2*a));
		}
	}
	else{
		double result = 0.5*(y+1.0); // O(a^0)

		if(std::abs(a) >= taylor_0_bound){ // go up to O(a^2)
			const double y2 = y*y;
			const double a2 = a*a;
			const double y2m1 = (y2-1.0);

			result += 0.25*a*y2m1*(1.0 + a*y/3.0); // O(a) + O(a^2)

			if(std::abs(a) >= taylor_2_bound){ // to up to O(a^4)
				const double y4 = y2*y2;
				const double a4 = a2*a2;

				result += a2*a*y2m1*y2m1/48.0; // O(a^3)
				result += a4*y*(3.0*y4-10.0*y2+7.0); // O(a^4)
			}
		}

		return result;
	}
}


class random_exp
{
	const double a;
	random_uniform ran;
public:
	random_exp(const double a_): a(a_) {
		ran = random_uniform();
	}
	~random_exp() {}

	double operator()(){
		const double u = ran();
		double x_min = -1; // C_exp = 0;
		double x_max =  1; // C_exp = 1;
		double x_mid =  0;
		double f = C_exp(x_mid,a);
		double df = std::abs(f - u);


		while(df > 1e-14){

			if(f > u){
				x_mid = x_max;
			}
			else if (f < u){
				x_mid = x_min;
			}

			f = C_exp(x_mid,a);
			df = std::abs(f-u);

		}

		return x_mid;

	}
};

*/





#endif