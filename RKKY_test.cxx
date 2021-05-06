

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
#include <set>


#include "ais/ais.h"
#include "ais/input.h"
#include "ais/classical/RKKY.h"
#include "ais/optimize.h" // sgd namespace
#include "ais/quantum/spin_fermion.h"



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
R sample_dist(const size_t nSample,DistQ &Q,DistP &P,Measurement m){


	const size_t nthread = omp_get_max_threads();
	const size_t NPerDist = (nSample+nthread-1)/nthread;

	R avg;

	// P = e^{logP-logQ}/Z. first pass w_list = logP-logQ, second pass w_list = exp(logP-logQ)
	std::vector<double> w_list(nSample);
	// saving the measurements from each configuration. 
	std::vector<R> R_list(nSample);

	double * w_ptr = &w_list[0];
	R * R_ptr = &R_list[0];

	double w_max = - std::numeric_limits<double>::infinity();
	double Z = 0;

	#pragma omp parallel
	{
		const int threadn = omp_get_thread_num();
		const size_t begin = NPerDist*threadn;
		const size_t end = std::min(nSample,NPerDist*(threadn+1));

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


/*
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

		filestream << std::setw(30) << mse;
		filestream << std::setw(30) << mse_running;
		filestream << std::setw(30) << mse_min;
		filestream << std::setw(30) << Nk;
		filestream << std::endl;

		return false;

	}
	
};




// typedef RKKY_gen<long_range_1D,100,1000,false,true> Model_t;
typedef RKKY_imp<fcc_lat,100,1000,false,true> Model_t;
typedef ais::Dist_template<Model_t::Sample_type,Model_t::Pars_type> dist;

typedef dist::Sample_type Sample_type;
typedef dist::Pars_type Pars_type;
// 
typedef sgd::adam<dist::Pars_type> stepper_type;

int main(int argc, char const *argv[])
{
	rand_normal normal;
	std::string infile;


	std::set<std::string> int_inputs = {"nBatch","N","R","nAdam","nSample"};
	std::set<std::string> double_inputs = {"beta","g"};

	std::cin >> infile;

	auto pars = ais::process_inputfile(infile,int_inputs,double_inputs);
	// first -> integer values
	// second -> double values

	const int nBatch  = pars.first["nBatch"];
	const int N       = pars.first["N"];
	const int R       = pars.first["R"];
	const int nAdam   = pars.first["nAdam"];
	const int nSample = pars.first["nSample"];
	const double beta = pars.second["beta"];
	const double g    = pars.second["g"];

	// long_range_1D lat(N,R);
	// Pars_type Jr(R);

	FCC::FCC fcc(N,true);
	const int N_tot = fcc.pos.size();

	fcc_lat lat(N,R);
	Pars_type Jr(lat.N_nn_dist());

	int stag = 1;
	for(int i=0;i<lat.N_nn_dist();i++){

		Jr[i] = 1e-1*normal();
		stag *= -1;

	}
	std::cout << "here" << std::endl;
	Model_t Q(beta,10.0,N_tot,Jr,&lat);

	const double mu = 0.0;
	const double M = 10000;
	// spin_fermion_chebyshev::spin_fermion_1d<Sample_type,Pars_type> P(N,g,beta,mu,M);
	spin_fermion_chebyshev::spin_fermion_EuO_1_band P(N,g,beta,mu,M);





	std::stringstream outfile;
	outfile << std::setprecision(3);

	outfile << "data/training_mse_spinferion_1d_nBatch_" << nBatch;
	outfile << "_beta_" << beta << "_g_" << g;
	outfile << "_L_" << N << "_R_" << R << ".txt";


	stepper_type stepper = stepper_type();

	ais::ais_KL<dist> ais((dist*)&P,(dist*)&Q);

	callaback cb = callaback<Pars_type>(1e-6,outfile.str());

	bool ended = ais.train_Q(nAdam,nBatch,stepper,cb);


	std::vector<Sample_type> Xk(nSample);
	std::vector<double> logQk(nSample),Wk(nSample);
	std::vector<Pars_type> gradk(nSample);

	Q.generate_samples(Xk,logQk,gradk);

	std::cout << "here" << std::endl;
	outfile.str(std::string());
	outfile << std::setprecision(3);

	outfile << "data/post_training_samples_spinferion_1d_nBatch_" << nBatch;
	outfile << "_beta_" << beta << "_g_" << g;
	outfile << "_L_" << N << "_R_" << R << ".txt";

	std::fstream outstream(outfile.str().c_str(),std::fstream::out);

	outstream << std::setprecision(15) << std::scientific;
	for(int i=0;i<nSample;i++){
		outstream << std::setw(30) << logQk[i]/(-beta);
		outstream << std::setw(30) << P.log_dist(Xk[i])/(-beta);
		for(int s=0;s<N;s++){
			outstream << std::setw(30) << Xk[i](s,0);
			outstream << std::setw(30) << Xk[i](s,1);
			outstream << std::setw(30) << Xk[i](s,2);
		}
		outstream << std::endl;
	}

	outstream.flush();
	outstream.close();
	




	return 0;
}



