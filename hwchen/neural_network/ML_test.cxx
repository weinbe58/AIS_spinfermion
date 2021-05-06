

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
// #include "RKKY.h"
#include "optimize.h" // sgd namespace
#include "spin_fermion.h"
#include "neural_network.h"


template<class Pars_type>
class callaback
{

public:
	Pars_type pars_min,pars_running;	
	const double tol,beta;
	double fluc_running,fluc_min;
	int nfunc,Nmin;


	callaback(const double tol_) : tol(tol_), beta(0.9), fluc_running(0),nfunc(0),Nmin(0) {}
	~callaback() {}


	bool operator()(const size_t t,Pars_type &pars,Pars_type &step,std::vector<double> &wk){

		
		double mean=0,var=0;
		
		nfunc += wk.size();

		for(auto w:wk){	mean += w;}
		mean /= wk.size();

		for(auto w:wk){	var += (w-mean)*(w-mean);}
		var /= (wk.size()-1);
		double fluc = (std::sqrt(var)/mean);

		if(t>0){
			fluc_running = (1-beta)*fluc + beta*fluc_running;
			pars_running = (1-beta)*pars + beta*pars_running;

			if(fluc_running < fluc_min){
				fluc_min = fluc_running;
				pars_min = pars_running;
				Nmin = 0;

			}
			else{
				Nmin+=wk.size();
			}
		}
		else{
			fluc_running = fluc;
			fluc_min = fluc;
			Nmin=0;
			pars_running = pars;
			pars_min = pars;
		}
		
		auto d = step.abs().matrix();
		
		if(t%1==0){

			std::cout << std::scientific << std::setprecision(5);
			std::cout << std::setw(10) << nfunc;
			std::cout << std::setw(10) << Nmin;
			std::cout << std::setw(5) << wk.size();
			std::cout << std::setw(15) << fluc_running;
			std::cout << std::setw(15) << fluc_min;
			std::cout << std::setw(15) << d.maxCoeff();
			// for(auto w:wk){std::cout << std::setw(15) << w;}
			
			std::cout << std::endl;
		}

		return ((fluc_running<tol) ||(Nmin >= 10000) || (d.maxCoeff() < 1e-6));

	}
	
};


typedef Neural_Net<Model, net_range_1D, 1000> Model_t;
typedef ais::Dist_template<Model_t::Sample_type,Model_t::Pars_type> dist;

typedef dist::Sample_type Sample_type;
typedef dist::Pars_type Pars_type;

// 
typedef sgd::adam<dist::Pars_type> stepper_type;

int main(int argc, char const *argv[])
{


	std::cout << std::scientific << std::setprecision(5);

	int nneighbour = 5; 
    int input_size = (nneighbour*2 + 1) * 3; 
    int h1_size = 20, h2_size = 10;

	const int N = 50;
	const double beta = 1/0.25;

	Model Net(nneighbour, input_size, h1_size, h2_size);
	net_range_1D lat(N, nneighbour);


	Neural_Net<Model, net_range_1D, 1000> Q(beta, N, Net, lat);


	const double g = 1.0;
	const double mu = 0.0;
	const double M = 1000;
	spin_fermion_chebyshev::spin_fermion_1d<Sample_type,Pars_type> P1(N,g,beta,mu,M);



	int ns = 10; // batch size for traning


	stepper_type stepper = stepper_type();

	ais::ais_KL<dist> ais((dist*)&P1,(dist*)&Q);

	auto cb = callaback<Pars_type>(1e-3);

	ais.train_Q(1000000,ns,stepper,cb);

	
	auto J_opt = Q.get_pars();



	return 0;
}



