#ifndef _Ising_
#define _Ising_ value





#include <Eigen/Dense>
#include "random.h"
#include "ais.h"

typedef Eigen::Array<double,2,Eigen::Dynamic,Eigen::RowMajor> Sample_type;
typedef Eigen::Array<double,Eigen::Dynamic,1> Pars_type;
typedef Eigen::Matrix<double,3,1> Vector3d;

class RKKY_1d : public ais::Dist_template<Sample_type,Pars_type> 
{

private:
	Pars_type Jr;
	Sample_type angles;
	rand_uniform ran;
	const double beta;
	const int N,R;


public:
	RKKY_1d(const double beta_,const int N_,const int R_) : beta(beta_), N(N_), R(R_) {
		ran = rand_uniform();
		angles = Sample_type::Zero(2,N_);
		Jr = Pars_type::Zero(R_+1);
		for(int r=0;r<=R_;r++){
			Jr = 2*ran()-1;
		}

		this->init_angles();
	}

	RKKY_1d(const double beta_,const int N_,Pars_type Jr_) : beta(beta_),N(N_), R(Jr_.size()-1) {
		ran = rand_uniform();
		angles = Sample_type::Zero(2,N_);
		Jr = Jr_;
		this->init_angles();
	}

	void init_angles(){
		for(int i=0;i<N;i++){
			angles(0,i) = std::acos(2*ran()-1);
			angles(1,i) = 2*M_PI*ran();
		}

		for(int i=0;i<1000;i++){
			this->metropolis_sweep();
		}
	}


	void metropolis_sweep(){

	}

	void generate_samples(std::vector<Sample_type> &Xk){
		const int Nk = Xk.size();
		int k=0;
		for(int i=1;i<=100*Nk;i++){
			this->metropolis_sweep();
			if((i%100)==0){
				Xk[k++] = angles;
			}
		}
	}


	double log_dist(Sample_type x){

	}

	Pars_type log_grad(Sample_type x){

	}

	Pars_type get_pars(void){
		return Jr;
	}

	void put_pars(Pars_type Jr_){
		Jr = Jr_;
	}

};





#endif