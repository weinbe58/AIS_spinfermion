#ifndef _SPIN_FERMION_
#define _SPIN_FERMION_


#include <complex>
#include <cmath>
#include <limits>
#include <iostream>
#include "ais/ais.h"
#include "ais/random.h"
#include "ais/classical/heis.h"
#include "ais/fcc_lattice.h"

#ifdef _OPENMP
#include <omp.h>
#else

inline
int omp_get_max_threads(void) {return 1;}

inline
int omp_get_num_threads(void) {return 1;}

inline
int omp_get_thread_num(void) {return 0;}

#endif


namespace util {


inline static double logaddexp(double const x, double const y)
{
	double const tmp = x - y;

	if (x == y)
		return x + M_LN2;

	if (tmp > 0)
		return x + log1p(exp(-tmp));
	else if (tmp <= 0)
		return y + log1p(exp(tmp));

	return tmp;
}


struct free_energy
{
	const double beta,mu;
	free_energy(const double beta_,const double mu_) : 
	beta(beta_), mu(mu_) {}

	inline
	double operator()(double x){
		return logaddexp(0.0,-beta*(x-mu)) - M_LN2;
	}
};

template<class sp_mat,const int nbands=1>
class matvec_spin_fermion
{
	sp_mat * Hk;
	const double* x; // array of unit vectors in cartesian coordinates.
	const double* H_diag; // diagonal part of fermion hamiltonian
	const int N;
	const double half_g,scale,sigma;

public:
	matvec_spin_fermion(sp_mat * Hk_,const double* x_,const double g,const double W_,const double* H_diag_=NULL,const double sigma_=0.0) : 
	Hk(Hk_),H_diag(H_diag_), x(x_), half_g(0.5 * g), scale(2.0/W_), 
	sigma(sigma_), N(Hk_->n_cols()/(2*nbands)) {}
	~matvec_spin_fermion() {}

	template<class Vec>
	void operator()(Vec &in,Vec &out){
		Hk->dot(in,out);

		if(H_diag!=NULL){
			for(int b=0;b<nbands;b++){
				const int begin = 2 * N * b;
				const int end   = begin + N;
				const double* v = x;
				for(int i=begin,j=begin+N;i<end;i++,j++){

					// spin-fermion int: J^z sigma^z + J^- sigma^+ + J^+ sigma^-
					const std::complex<double> Jp(v[0],v[1]); // J^+ = J^x + i*J^y

					auto in_i = in[i]; // spin-up
					auto in_j = in[j]; // spin-down

					out[i] += in_i * H_diag[i] + half_g * ( v[2] * in_i + std::conj(Jp) * in_j);
					out[j] += in_j * H_diag[j] + half_g * (-v[2] * in_j + Jp * in_i);

					v += 3;
				}

			}
		}
		else{
			for(int b=0;b<nbands;b++){
				const int begin = 2 * N * b;
				const int end   = begin + N;
				const double* v = x;

				for(int i=begin,j=begin+N;i<end;i++,j++){

					// spin-fermion int: J^z sigma^z + J^- sigma^+ + J^+ sigma^-
					const std::complex<double> Jp(v[0],v[1]); // J^+ = J^x + i*J^y

					auto in_i = in[i]; // spin-up
					auto in_j = in[j]; // spin-down

					out[i] += half_g * ( v[2] * in_i + std::conj(Jp) * in_j);
					out[j] += half_g * (-v[2] * in_j + Jp * in_i);

					v += 3;
				}

			}			
		}

			for(int b=0;b<nbands;b++){
				const int begin = 2 * N * b;
				const int end   = begin + N;

				for(int i=begin,j=begin+N;i<end;i++,j++){
					out[i] = (out[i] - sigma*in[i])*scale;
					out[j] = (out[j] - sigma*in[j])*scale;
				}



			}
	}


};

}


#ifdef AIS_USE_EIGEN

#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

namespace spin_fermion_dense {

typedef Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Matrix2z;
typedef Eigen::SelfAdjointEigenSolver<Matrix2z> EigenSolver;

template<class Sample_type,class Pars_type>
class spin_fermion : public ais::Dist_template<Sample_type,Pars_type>
{
	const int N_spin;
	const double g,beta,mu;

	EigenSolver solver;

public:
	Matrix2z Hk;

	spin_fermion(const int N_spin_,const double g_,const double beta_,const double mu_) : 
	N_spin(N_spin_), g(g_), beta(beta_), mu(mu_) {Hk = Matrix2z::Zero(2*N_spin_,2*N_spin_);}
	~spin_fermion() {}


	double log_dist(Sample_type x){

		for(int i=0;i<N_spin;i++){
			Hk(i,i) = 0.5 * g * x(i,2);
			Hk(i,i+N_spin) = 0.5 * g * std::complex<double>(x(i,0),-x(i,1));
			Hk(i+N_spin,i) = 0.5 * g * std::complex<double>(x(i,0), x(i,1));
			Hk(i+N_spin,i+N_spin) = -0.5 * g * x(i,2);
		}

		double F = 0;
		solver.compute(Hk);
		typename EigenSolver::RealVectorType E = solver.eigenvalues();

		for(int i=0;i<E.size();i++){
			F += util::free_energy(beta,mu)(E[i]);
		}
		return F;
	}
	
	void generate_samples(std::vector<Sample_type>&,std::vector<double>&,std::vector<Pars_type>&) {}
	Pars_type log_grad(Sample_type) {}
	Pars_type get_pars(void) {}
	void put_pars(Pars_type) {}
};

template<class Sample_type,class Pars_type>
class spin_fermion_1d : public spin_fermion<Sample_type,Pars_type>
{
public:
	spin_fermion_1d(const int N_spin,const double g,const double beta,const double mu=0.0) : 
	spin_fermion<Sample_type,Pars_type>::spin_fermion(N_spin,g,beta,0.0)
	{
		const int L = N_spin;
		for(int i=0;i<L;i++){
			const int ip = (i+1)%L;

			spin_fermion<Sample_type,Pars_type>::Hk(i,ip) = -1;
			spin_fermion<Sample_type,Pars_type>::Hk(ip,i) = -1;
			spin_fermion<Sample_type,Pars_type>::Hk(i+L,ip+L) = -1;
			spin_fermion<Sample_type,Pars_type>::Hk(ip+L,i+L) = -1;
		}

	}
	~spin_fermion_1d() {}
};


} // end namespace spin_fermion_dense

#endif






namespace spin_fermion_chebyshev {

#include "ais/quantum/chebyshev.h"
#include "ais/quantum/csr.h"


typedef Eigen::Matrix<std::complex<double>,Eigen::Dynamic,1> VectorXcd;


template<class sp_mat,int nbands,class Chebyshev_t,class Sample_type,class Pars_type>
class spin_fermion : public ais::Dist_template<Sample_type,Pars_type>
{



public:

	const double g,beta,mu,W,sigma;
	std::vector<double> C_m;
	sp_mat * Hk;
	std::vector<double> H_diag;
	Chebyshev_t * chebyshev_calc;


	spin_fermion(const double g_,const double beta_,const double mu_,const double W_,const double _sigma=0) : 
	g(g_), beta(beta_), mu(mu_), W(W_), sigma(_sigma) {}
	~spin_fermion() {}

	void init(void){
		chebyshev_calc->coeffs(util::free_energy(beta,mu),C_m,W);
	}

	virtual double log_dist(Sample_type x){

		
		const int n_rows = Hk->n_rows();
		const int nthread = omp_get_max_threads();

		const double * H_diag_data = NULL;

		if(H_diag.size()!=0){
			H_diag_data = &H_diag[0];
		}

		auto matvec = util::matvec_spin_fermion<sp_mat,nbands>(Hk,x.data(),g,W,H_diag_data,sigma);

		std::vector<double> thread_sums(nthread);


		double * sums = &thread_sums[0];
		double * C_m_ptr = &C_m[0];

		#pragma omp parallel shared(matvec,chebyshev_calc)
		{
			const int threadn = omp_get_thread_num();
			std::vector<std::complex<double>> v0(n_rows,0.0);
			std::vector<std::complex<double>>  v1(n_rows);
			std::vector<std::complex<double>>  v2(n_rows);
			std::vector<std::complex<double>>  v3(n_rows);
			
			for(size_t r=threadn;r < n_rows;r += nthread){
				v0[r] = 1.0;
				const double f = chebyshev_calc->sum_series(matvec,v0,v1,v2,v3,C_m_ptr);
				v0[r] = 0.0;

				// update mean and sum of squares. 
				sums[threadn] += f;
				

			}

		}

		double result = 0;
		for(auto s : thread_sums){result += s;}

		return result;
	}

	void generate_samples(std::vector<Sample_type>&,std::vector<double>&,std::vector<Pars_type>&) {}
	Pars_type log_grad(Sample_type) {}
	Pars_type get_pars(void) {}
	void put_pars(Pars_type) {}
};


template<class Sample_type,class Pars_type>
class spin_fermion_1d : public spin_fermion<csr_matrix<double,int>,1,Chebyshev,Sample_type,Pars_type> 
{

	typedef spin_fermion<csr_matrix<double,int>,1,Chebyshev,Sample_type,Pars_type> parent_type;
public:
	spin_fermion_1d(const int L,const double g,const double beta,const double mu=0.0,const int M=100) : 
	parent_type::spin_fermion(g,beta,mu,4.1+g) 
	{
		std::vector<double>	me;
		std::vector<int> row,col;

		const int N_me= 2*L;

		me.reserve(2*N_me);
		row.reserve(2*N_me);
		col.reserve(2*N_me);

		for(int i=0;i<L;i++){
			const int ip = (i+1)%L ;
			row.push_back(i); col.push_back(ip);
			me.push_back(-1.0);

			row.push_back(ip); col.push_back(i);
			me.push_back(-1.0);

			row.push_back(ip+L); col.push_back(i+L);
			me.push_back(-1.0);

			row.push_back(i+L);	col.push_back(ip+L);
			me.push_back(-1.0);
		}

		parent_type::Hk = new csr_matrix<double,int>(std::forward_as_tuple(me,row,col),2*L,2*L);
		parent_type::chebyshev_calc = new Chebyshev(M);
		this->init();

	}
	~spin_fermion_1d() {
		delete parent_type::Hk;
		delete parent_type::chebyshev_calc;
	}
};




class spin_fermion_EuO_1_band : public spin_fermion<csr_matrix<double,int>,1,Chebyshev,O3::Sample_type,O3::Pars_type> 
{
	// Samples both spins and impurities on EuO material.
	// to be pair with Eu0_1_band_RKKY to do adaptive importance sampling

	// units in eV
	static const int nbands = 1;
	const double Jnn = -0.1E-3;
	const double Jnnn = 3.5E-4;
	const double Jd = 0.1;

	const double ed[nbands] = {-.4 * 8};
	const double tnn[nbands] = {-1.0};

	const double mu_d;


	FCC::FCC * fcc;
	typedef spin_fermion<csr_matrix<double,int>,nbands,Chebyshev,O3::Sample_type,O3::Pars_type> parent_type;
public:
	spin_fermion_EuO_1_band(const int L,const double mu_d_,
		const double beta,const double mu=0.0,const int M=100) : 
	parent_type::spin_fermion(0.1,beta,mu,17.2), mu_d(mu_d_)
	{

		rand_uniform ran();

		fcc = new FCC::FCC(L,true);

		const int N = fcc->nn.size();

		parent_type::H_diag.resize(2*N);


		std::vector<double>	me;
		std::vector<int> row,col;

		for(int site=0;site<N;site++){ // looping over sites
			for(auto nn_site : fcc->nn[site]){ // looping over nn hopping
				for(int b=0;b<nbands;b++){ // looping over bands
					const int site_index_down = 2*N*b + site;
					const int site_index_up = site_index_down + N;
					const int site_index_down_nn = 2*N*b + nn_site;
					const int site_index_up_nn = site_index_down_nn + N;

					if(site_index_down_nn >= 2*N){
						std::cout << nn_site << std::endl;
					}

					// down spin
					col.push_back(site_index_down_nn);
					col.push_back(site_index_down);

					row.push_back(site_index_down);
					row.push_back(site_index_down_nn);

					me.push_back(tnn[b]);
					me.push_back(tnn[b]);

					// up spin
					col.push_back(site_index_up_nn);
					col.push_back(site_index_up);

					row.push_back(site_index_up);
					row.push_back(site_index_up_nn);

					me.push_back(tnn[b]);
					me.push_back(tnn[b]);
				}
			}
		}


		parent_type::Hk = new csr_matrix<double,int>(std::forward_as_tuple(me,row,col),2*N*nbands,2*N*nbands);
		parent_type::chebyshev_calc = new Chebyshev(M);
		this->init();

	}

	~spin_fermion_EuO_1_band() {
		delete parent_type::Hk;
		delete parent_type::chebyshev_calc;
		delete fcc;
	}


	double log_dist(Sample_type x){
		const int N = fcc->nn.size();

		
		double F = 0;
		for(int i=N,j=0;i<2*N;i++,j++){
			H_diag[j  ] = x(i,0)*ed[0]; // 0 or 1 based to specify occupation of impurity
			H_diag[j+N] = x(i,0)*ed[0];
			F += mu_d * x(i,0);
		}

		F += parent_type::log_dist(x);
		
		for(int i = 0;i<N;i++){
			O3::Vector3d si = x.row(i);
			for(auto j : fcc->nn[i]){
				O3::Vector3d sj = x.row(j);
				F += -parent_type::beta*Jnn*si.dot(sj);
			}

			for(auto j : fcc->nnn[i]){
				O3::Vector3d sj = x.row(j);
				F += -parent_type::beta*Jnnn*si.dot(sj);
			}
		}

		return F;
	}
};



} // end namespace spin_fermion_chebyshev


#endif