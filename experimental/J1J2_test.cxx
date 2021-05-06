

#define EIGEN_DONT_PARALLELIZE 1


#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <complex>
#include <Eigen/Dense>
#include <omp.h>
#include "ais.h"
#include "J1J2.h"
#include "On_RBM.h"
#include "optimize.h"


class callaback
{
	
	const double tol,beta;
	double fluc_running;
public:
	callaback(const double tol_) : tol(tol_), beta(0.9), fluc_running(0) {}
	~callaback() {}

	template<class Pars_type>
	bool operator()(const size_t t,Pars_type &pars,Pars_type &step,std::vector<double> &wk){


		double mean=0,var=0;

		for(auto w:wk){	mean += w;}
		mean /= wk.size();

		for(auto w:wk){	var += (w-mean)*(w-mean);}
		var /= (wk.size()-1);

		if(t>0){
			fluc_running = (1-beta)*(std::sqrt(var)/mean) + beta*fluc_running;
		}
		else{
			fluc_running = std::sqrt(var)/mean;
		}

		// std::cout << step << std::endl;
		std::cout << std::scientific << std::setprecision(5) << std::setw(15) << fluc_running;
		// for(auto w : wk){
		// 	std::cout << std::setw(15) << w;
		// }
		std::cout << std::endl;
		

		// return (fluc_running<tol);

		return false;

	}
	
};

#define NSTEP 1
#define WARMUP 50

typedef RBM_ising<NSTEP,WARMUP>::Pars_type Pars_type;
typedef RBM_ising<NSTEP,WARMUP>::Sample_type Sample_type;

typedef ais::Dist_template<Sample_type,Pars_type> dist;

struct decay
{
	decay() {}
	inline double operator()(const size_t t){
		// double d = 1.0/std::pow(t,0.5);

		return 1.0;
	}
};


int main(int argc, char const *argv[])
{

	const int L = 4;
	const int N = L*L;

	RBM_ising<NSTEP,WARMUP> Q(N,2*N);
	J1J2<Sample_type,Pars_type> P(L,0.0,2.1);


	typedef sgd::adam<dist::Pars_type> stepper_type;

	stepper_type stepper = stepper_type();
	ais::ais_KL<dist> ais((dist*)&P,(dist*)&Q);


	ais.train_Q(10000000,100,stepper,callaback(1e-3));

	


	return 0;
}



