#ifndef _CHEBYSHEV_
#define _CHEBYSHEV_

#include <cmath>
#include <limits>



template<class Vec>
double vdot(Vec &x,Vec &y){
	double vd = 0.0;
	for(int i=0;i<x.size();i++){
		vd += (double)(x[i].real() * y[i].real() + x[i].imag() * y[i].imag());
	}
	return vd;
}


double jackson_kernel(const int M,const int m)
{
	return ( (M - m + 1) * std::cos(m*M_PI/(M+1)) + std::sin(m*M_PI/(M+1)) / std::tan(M_PI/(M+1)) )/(M+1); 
}



class Chebyshev
{
	const int M;
	std::vector<double> g;
	static constexpr double alpha = std::exp(-2.0);

public:
	Chebyshev(const int M_) : M(M_){
		// g.push_back(jackson_kernel(M_,0));

		// for(int m=1;m<M_;m++){
		// 	g.push_back(2*jackson_kernel(M_,m));
		// }

		g.push_back(1);
		for(int m=1;m<M;m++){
			g.push_back(2);
		}
	}

	Chebyshev(Chebyshev &other) : M(other.M){
		g = other.g;
	}

	~Chebyshev() {}

	template<class MatVec,class Vec,class Arr>
	double sum_series(MatVec &matvec,Vec &v0,Vec &v1,Vec &v2, Vec &v3,
		Arr &C_m,const double atol = std::numeric_limits<double>::epsilon() ){

		const int n = v1.size();
		v1 = v0;
		matvec(v1,v2); // v2 = H * v1;
		double d0,d1;

		d0 = C_m[0] * vdot(v0,v1);
		d1 = C_m[1] * vdot(v0,v2);

		double avg_err = (1.0-alpha)*std::abs(d1) + alpha*std::abs(d0);

		double result = g[0]*d0 + g[1]*d1;

		for(int m=2;m<M;m++){
			matvec(v2,v3); // v3 = H * v2

			// v3 = 2 * v3 - v1
			for(int i=0;i<n;i++){
				v3[i] = 2.0 * v3[i] - v1[i];
			}

			const double D = C_m[m] * vdot(v0,v3);
			const double abs_D = std::abs(D);
			avg_err = (1.0-alpha)*abs_D + alpha * avg_err;

			result += g[m] * D;

			if(std::max(abs_D,avg_err) < atol){
				break;
			}

			v1 = v2; v2 = v3;
		}
		return result;
	}
	
	template<class Func>
	void coeffs(Func func,std::vector<double> &C_m,const double W=2,const double sigma=0){

		C_m.clear();

		for(int m=0;m<M;m++){

			// perform Chebyshev-guass quaderature
			const int N_cq = M+m;
			double Int = 0.0;
			for(int k=1;k<=N_cq;k++){
				const double theta_k = (2*k - 1) * M_PI / (2 * N_cq);
				const double eps = (W / 2) * std::cos(theta_k) + sigma;
				
				Int += std::cos(m*theta_k) * func(eps);

			} 
			Int /= N_cq;
			C_m.push_back(Int);
		}

	}
};





#endif