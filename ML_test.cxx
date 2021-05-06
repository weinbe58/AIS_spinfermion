

#define EIGEN_DONT_PARALLELIZE 1


#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <complex>
#include <Eigen/Dense>
#include <omp.h>
#include <torch/torch.h>


#include "ais/ais.h"
#include "ais/nn/neural_network.h"
#include "ais/optimize.h" // sgd namespace
#include "ais/quantum/spin_fermion.h"



template<class Pars_type>
class callaback
{

public:
	const double tol,beta;
	double fluc_running,fluc_min,mse_running,mse_min;
	int nfunc,Nmin;


	callaback(const double tol_) : tol(tol_), beta(0.9), fluc_running(0),nfunc(0),Nmin(0) {}
	~callaback() {}


	bool operator()(const size_t t,Pars_type &pars,Pars_type &step,std::vector<double> &wk){

		
		double mean=0,var=0,mse=0;
		
		nfunc += wk.size();

		int count = 0;


		const size_t Nk = wk.size();

		for(auto w : wk){
			count++;
			double delta =  std::log(Nk*w)*std::log(Nk*w) - mse;
			mse += delta/count;
		}

		for(auto w:wk){	mean += w;}
		mean /= wk.size();

		for(auto w:wk){	var += (w-mean)*(w-mean);}
		var /= (wk.size()-1);
		double fluc = (std::sqrt(var)/mean);

		if(t>0){
			fluc_running = (1-beta)*fluc + beta*fluc_running;
			mse_running = (1-beta)*mse + beta*mse_running;

			Nmin+=wk.size();
			fluc_min = std::min(fluc_min,fluc_running);


			if(mse_running < mse_min){
				mse_min = mse_running;
				Nmin = 0;

			}



		}
		else{
			fluc_running = fluc;
			fluc_min = fluc;
			mse_running = mse;
			mse_min = mse;
			Nmin=0;
		}
		
		
		if(t%1==0){
			// std::cout << pars[2] << std::endl << std::endl;
			// std::cout << step[2] << std::endl << std::endl;
			// for(auto p : step){
			// 	std::cout << p << std::endl << std::endl;
			// }
			std::cout << std::scientific << std::setprecision(5);
			std::cout << std::setw(10) << nfunc;
			std::cout << std::setw(10) << Nmin;
			std::cout << std::setw(5) << wk.size();
			std::cout << std::setw(15) << mse_running;
			std::cout << std::setw(15) << mse_min;
			// for(auto w:wk){std::cout << std::setw(15) << w;}
			
			std::cout << std::endl;
		}

		return (fluc_running<tol) ||(Nmin >= 100000);
		// return false;

	}
	
};


//construct neural network model
struct Model3 : torch::nn::Module {

    int nn, input_size, h1_size, h2_size, h3_size; 

    Model3(int nnneighbour_, int input_size_, int h1_size_, int h2_size_, int h3_size_):
        nn(nnneighbour_), input_size(input_size_), h1_size(h1_size_), h2_size(h2_size_), h3_size(h3_size_)
    {
        // construct and register layers
        layer1 = register_module("layer1", torch::nn::Linear(input_size, h1_size));
        layer1->to(torch::kDouble);  //use double 

        layer2 = register_module("layer2", torch::nn::Linear(h1_size, h2_size));
        layer2->to(torch::kDouble); 

        layer3 = register_module("layer3", torch::nn::Linear(h2_size, h3_size));
        layer3->to(torch::kDouble); 

        out = register_module("out", torch::nn::Linear(h3_size, 1));
        out->to(torch::kDouble); 

        Init();
    }

    torch::Tensor forward(torch::Tensor X)
    {
        X = torch::gelu(layer1->forward(X));
        X = torch::gelu(layer2->forward(X));
        X = torch::gelu(layer3->forward(X));
        X = out->forward(X);
        
        return X;
    }

    void Init() {

        torch::NoGradGuard no_grad;
        double w = 0.01;
        for (auto& p : this->parameters()) {
            
            p.uniform_(-w, w); // or whatever initialization you are looking for.
        }
    }

    void update_(const std::vector<torch::Tensor>& new_pars){
        torch::NoGradGuard no_grad;
        layer1->weight.copy_(new_pars[0]);
        layer1->bias.copy_(new_pars[1]);
        layer2->weight.copy_(new_pars[2]);
        layer2->bias.copy_(new_pars[3]);
        layer3->weight.copy_(new_pars[4]);
        layer3->bias.copy_(new_pars[5]);
        out->weight.copy_(new_pars[6]);
        out->bias.copy_(new_pars[7]);

    }

    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, out{nullptr};

};

//construct neural network model
struct Model2 : torch::nn::Module {

    int nn, input_size, h1_size, h2_size; 

    Model2(int nnneighbour_, int input_size_, int h1_size_, int h2_size_):
        nn(nnneighbour_), input_size(input_size_), h1_size(h1_size_), h2_size(h2_size_)
    {
        // construct and register layers
        layer1 = register_module("layer1", torch::nn::Linear(input_size, h1_size));
        layer1->to(torch::kDouble);  //use double 

        layer2 = register_module("layer2", torch::nn::Linear(h1_size, h2_size));
        layer2->to(torch::kDouble); 

        out = register_module("out", torch::nn::Linear(h2_size, 1));
        out->to(torch::kDouble); 

        Init();
    }

    torch::Tensor forward(torch::Tensor X)
    {
        X = torch::gelu(layer1->forward(X));
        X = torch::gelu(layer2->forward(X));
        X = out->forward(X);
        
        return X;
    }

    void Init() {

        torch::NoGradGuard no_grad;
        double w = 0.01;
        for (auto& p : this->parameters()) {
            
            p.uniform_(-w, w); // or whatever initialization you are looking for.
        }
    }

    void update_(const std::vector<torch::Tensor>& new_pars){
        torch::NoGradGuard no_grad;
        layer1->weight.set_data(new_pars[0]);
        layer1->bias.set_data(new_pars[1]);
        layer2->weight.set_data(new_pars[2]);
        layer2->bias.set_data(new_pars[3]);
        out->weight.set_data(new_pars[4]);
        out->bias.set_data(new_pars[5]);

    }

    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, out{nullptr};

};

typedef nn::Neural_Net<Model2, nn::net_range_1D, 100,5> Model_t;
typedef ais::Dist_template<Model_t::Sample_type,Model_t::Pars_type> dist;

typedef dist::Sample_type Sample_type;
typedef dist::Pars_type Pars_type;

// 
typedef sgd::adam<dist::Pars_type> stepper_type;

int main(int argc, char const *argv[])
{


	std::cout << std::scientific << std::setprecision(5);

	int nneighbour = 4; 
    int input_size = (nneighbour*2 + 1) * 3; 
    int h1_size = 20, h2_size = 10, h3_size = 25;

	const int N = 10;
	const double beta = 8;

	// Model3 Net(nneighbour, input_size, h1_size, h2_size, h3_size);
	Model2 Net(nneighbour, input_size, h1_size, h2_size);
	nn::net_range_1D lat(N, nneighbour);





	Model_t Q(beta, N, Net, lat);


	const double g = 1.0;
	const double mu = 0.0;
	const double M = 10000;
	spin_fermion_chebyshev::spin_fermion_1d<Sample_type,Pars_type> P1(N,g,beta,mu,M);



	int ns = 1000; // batch size for traning


	stepper_type stepper = stepper_type();

	ais::ais_KL<dist> ais((dist*)&P1,(dist*)&Q);



	auto cb = callaback<Pars_type>(1e-9);

	ais.train_Q(100000,ns,stepper,cb);



	// const int Nsample = 100;
	// std::vector<Sample_type> Xk(Nsample);
	// std::vector<double> logQk(Nsample),logPk(Nsample),Wk(Nsample);
	// std::vector<Pars_type> gradk(Nsample);
	// Q.generate_samples(Xk,logQk,gradk);
	// double avg = 0;
	// for(int i=0;i<Nsample;i++){
	// 	logPk[i] = P1.log_dist(Xk[i]);
	// 	avg += (logQk[i] + logPk[i])/Nsample;
	// }

	// for(int i=0;i<Nsample;i++){
	// 	std::cout << std::setw(30) << (logQk[i] - avg/2.0) - (logPk[i] + avg/2.0);
	// 	std::cout << std::endl;
	// }


	
	// auto J_opt = Q.get_pars();



	return 0;
}



