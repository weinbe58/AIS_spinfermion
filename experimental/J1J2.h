#ifndef _J1J2_
#define _J1J2_


#include <Eigen/Dense>
#include "random.h"
#include "ais.h"


template<class Sample_type,class Pars_type>
class J1J2 : public ais::Dist_template<Sample_type,Pars_type>
{
	const double J2,beta;
	const int L;
public:
	J1J2(const int L_,const double J2_,const double beta_): J2(J2_), beta(beta_), L(L_) {}
	~J1J2() {}


	double log_dist(Sample_type X){
		double E=0;
		for(int y=0;y<L;y++){
			const int yp = (y+1)%L;
			const int ym = (y-1+L)%L;
			for(int x=0;x<L;x++){
				const int xp = (x+1)%L;

				const int i = x+L*y;

				const int ixp = xp + L*y;
				const int iyp = x + L*yp;

				const int ixpyp = xp+L*yp;
				const int ixpym = xp+L*ym;

				E += X[i] * ( X[ixp] + X[iyp] + J2*(X[ixpyp]+X[ixpym]) );

			}
		}
		return -beta*E;
	}

	void generate_samples(std::vector<Sample_type>&){}
	void generate_samples(std::vector<Sample_type>&,std::vector<double>&){};
	Pars_type log_grad(Sample_type){};
	Pars_type get_pars(void){};
	void put_pars(Pars_type){};

};







#endif