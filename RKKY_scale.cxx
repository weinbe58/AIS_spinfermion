

#define EIGEN_DONT_PARALLELIZE 1


#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <complex>
#include <fstream>
#include <Eigen/Dense>
#include <omp.h>


#include "ais.h"
#include "RKKY.h"
#include "optimize.h" // sgd namespace
#include "spin_fermion.h"



/*

class mag
{
public:
	mag() {}
	~mag() {}

	template<class Sample_type>
	double operator()(Sample_type &angles){
		double m = 0;
		int stag = 1;
		for(int i=0;i<angles.cols();i++){
			m += stag*std::cos(angles(0,i));
			stag *= -1;
		}

		return m/angles.cols();
	}
};


struct decay
{
	decay() {}
	inline double operator()(const size_t t){
		double d = 1.0/std::pow(1+std::log1p(t),1.0);
		return d;
	}
};


template<class DistQ,class DistP,class Measurement,class R,int Nss=10>
R sample_dist(const size_t Nsample,DistQ &Q,DistP &P,Measurement m){


	const size_t nthread = omp_get_max_threads();
	const size_t NPerDist = (Nsample+nthread-1)/nthread;

	R avg;

	// P = e^{logP-logQ}/Z. first pass w_list = logP-logQ, second pass w_list = exp(logP-logQ)
	std::vector<double> w_list(Nsample);
	// saving the measurements from each configuration. 
	std::vector<R> R_list(Nsample);

	double * w_ptr = &w_list[0];
	R * R_ptr = &R_list[0];

	double w_max = - std::numeric_limits<double>::infinity();
	double Z = 0;

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		const size_t begin = NPerDist*threadn;
		const size_t end = std::min(Nsample,NPerDist*(threadn+1));

		DistQ thread_Q(Q);
		DistP thread_P(P);

		size_t samples = begin;
		std::vector<typename DistQ::Sample_type> Xx;
		std::vector<double> logQ;

		logQ.reserve(Nss);
		Xx.reserve(Nss);

		// need to find the maximum exponent to prevent overflows
		double w_max_thread = w_max;
		while(samples!=end){
			const int n = std::min((size_t)Nss,end - samples);
			Xx.clear(); logQ.clear();

			Xx.resize(n); logQ.resize(n);

			thread_Q.generate_samples(Xx,logQ);

			for(int i=0;i<n;i++){
				const double w = thread_P.log_dist(Xx[i]) - logQ[i];
				w_max_thread = std::max(w_max_thread,w);
				w_ptr[samples] = w;
				R_ptr[samples++] = m(Xx[i]);
			}
		}
		// find largest exponent to prevent overflows
		#pragma omp critical
		{
			w_max = std::max(w_max,w_max_thread);			
		}

		// initialize avg variable
		#pragma omp master
		{
			avg = 0 * R_ptr[begin];
		}

		#pragma omp barrier

		// calculate the normalization for this thread
		double Z_thread = 0;

		for(int i=begin;i<end;i++){
			const double w = std::exp(w_ptr[i] - w_max); 
			w_ptr[i] = w;
			Z_thread += w;
		}
		// sum normalizations from all threads
		#pragma omp atomic
		Z += Z_thread;

		#pragma omp barrier

		// calculate local average, initialize variable
		R avg_local = R_ptr[begin] * w_ptr[begin] / Z;
		for(int i=begin+1;i<end;i++){
			avg_local += R_ptr[i] * w_ptr[i] / Z;
		}

		// collect averages over all threads
		#pragma omp critical
		{
			avg += avg_local;
		}

	}

	return avg;
}
*/



template<class Pars_type>
class callaback
{

public:
	Pars_type pars_min,pars_running;	
	const double tol,beta;
	double mse_running,mse_min;
	int nfunc,Nmin;
	std::fstream filestream;

	callaback(const double tol_,std::string outfile) : tol(tol_), beta(0.9), mse_running(0),nfunc(0),Nmin(0) {
		filestream.open(outfile.c_str(),std::fstream::out);
		filestream << std::setprecision(15) << std::scientific;
	}
	~callaback() {filestream.close();}


	bool operator()(const size_t t,Pars_type &pars,Pars_type &step,std::vector<double> &wk){

		
		double mse=0;
		int count = 0;


		const size_t Nk = wk.size();

		for(auto w : wk){
			count++;
			double delta =  std::log(Nk*w)*std::log(Nk*w) - mse;
			mse += delta/count;
		}

		nfunc += Nk;

		if(t>0){
			mse_running = (1-beta)*mse + beta*mse_running;
			pars_running = (1-beta)*pars + beta*pars_running;

			if(mse < mse_min){
				mse_min = mse;
				pars_min = pars_running;
				Nmin = 0;

			}
			else{
				Nmin++;
			}
		}
		else{
			mse_running = mse;
			mse_min = mse;
			Nmin=0;
			pars_running = pars;
			pars_min = pars;
		}
		std::cout << t << std::endl;
		filestream << std::setw(30) << mse;
		filestream << std::setw(30) << mse_running;
		filestream << std::setw(30) << mse_min;
		filestream << std::setw(30) << Nk;
		filestream << std::endl;

		return false;

	}
	
};




typedef RKKY_gen<long_range_1D,100,1000,false,true> Model_t;
typedef ais::Dist_template<Model_t::Sample_type,Model_t::Pars_type> dist;

typedef dist::Sample_type Sample_type;
typedef dist::Pars_type Pars_type;

// 
typedef sgd::adam<dist::Pars_type> stepper_type;

int main(int argc, char const *argv[])
{
	rand_normal normal;

	int ns,N,R,Nt;
	double beta,g;
	std::cin >> ns >> Nt >> beta >> g >> N >> R;


	long_range_1D lat(N,R),lat2(2*N,R),lat4(4*N,R);

	Pars_type Jr(R);

	int stag = 1;
	for(int i=1;i<R+1;i++){

		Jr[i-1] = 1e-1*normal();
		stag *= -1;

	}
	Model_t Q(beta,N,Jr,&lat);



	const double mu = 0.0;
	const double M = 10000;
	spin_fermion_chebyshev::spin_fermion_1d<Sample_type,Pars_type> P(N,g,beta,mu,M);





	std::stringstream outfile;
	outfile << std::setprecision(3);

	outfile << "data/training_mse_spinferion_1d_nk_" << ns;
	outfile << "_beta_" << beta << "_g_" << g;
	outfile << "_L_" << N << "_R_" << R << ".txt";

	std::cout << outfile.str() << std::endl;
	stepper_type stepper = stepper_type();

	ais::ais_KL<dist> ais((dist*)&P,(dist*)&Q);

	callaback cb = callaback<Pars_type>(1e-6,outfile.str());

	bool ended = ais.train_Q(Nt,ns,stepper,cb);



	outfile.str(std::string());
	outfile << std::setprecision(3);

	outfile << "data/post_training_samples_spinferion_1d_nk_" << ns;
	outfile << "_beta_" << beta << "_g_" << g;
	outfile << "_L_" << N << "_R_" << R << ".txt";

	std::fstream outstream(outfile.str().c_str(),std::fstream::out);

	Model_t Q2(beta,2*N,Q.get_pars(),&lat2);
	Model_t Q4(beta,4*N,Q.get_pars(),&lat4);

	spin_fermion_chebyshev::spin_fermion_1d<Sample_type,Pars_type> P2(2*N,g,beta,mu,M);
	spin_fermion_chebyshev::spin_fermion_1d<Sample_type,Pars_type> P4(4*N,g,beta,mu,M);

	const int Nsample = 1000;

	outstream << std::setprecision(15) << std::scientific;
	for(int i=0;i<Nsample;i++){
		std::cout << i << std::endl;
		{
			std::vector<Sample_type> Xk(1);
			std::vector<double> logQk(1);
			std::vector<Pars_type> gradk(1);
			Q.generate_samples(Xk,logQk,gradk);
			outstream << std::setw(30) << logQk[0];
			outstream << std::setw(30) << P.log_dist(Xk[0]);
		}

		{
			std::vector<Sample_type> Xk(1);
			std::vector<double> logQk(1);
			std::vector<Pars_type> gradk(1);
			Q2.generate_samples(Xk,logQk,gradk);
			outstream << std::setw(30) << logQk[0];
			outstream << std::setw(30) << P2.log_dist(Xk[0]);
		}

		{
			std::vector<Sample_type> Xk(1);
			std::vector<double> logQk(1);
			std::vector<Pars_type> gradk(1);
			Q4.generate_samples(Xk,logQk,gradk);
			outstream << std::setw(30) << logQk[0];
			outstream << std::setw(30) << P4.log_dist(Xk[0]);
		}

		outstream << std::endl;
	}

	outstream.flush();
	outstream.close();
	




	return 0;
}



