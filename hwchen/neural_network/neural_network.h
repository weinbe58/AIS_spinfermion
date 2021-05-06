#ifndef _neural_network_
#define _neural_network_

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <torch/torch.h>
// #include <Eigen/Dense>
using namespace std; 

// #include "heis.h"
// #include "ais.h"
typedef Eigen::Matrix<double,3,1> Vector3d;
typedef Eigen::Array<double,Eigen::Dynamic,3,Eigen::RowMajor> Sample_type;
typedef vector< torch::Tensor > Pars_type; 
// typedef vector< vector<double> > Sample_type; 


Pars_type operator*(const double& a, const Pars_type& b)
{
    Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a * b[i]; 

    return res; 
}

Pars_type operator*(const int& a, const Pars_type& b)
{
    Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a * b[i]; 

    return res; 
}


double normal()
{
    srand(time(NULL)); 
    double r = (double)(rand()/ (double)RAND_MAX);
    return r;
}

class net_range_1D
{

public:
    const int N, len; 

    vector< vector<int> > nn_list; 

    net_range_1D(const int N_, const int len_):
    N(N_), len(len_)
    {
        nn_list.resize(N); 

        for(int i=0;i<N_;i++)
            for(int j=-len_;j<=len_;j++)
            {
                int pos = (i+j+N_)%N_;
                nn_list[i].push_back(pos); 
            }

    }


};


//construct neural network model
struct Model : torch::nn::Module {

    int nn, input_size, h1_size, h2_size; 

    Model(int nnneighbour_, int input_size_, int h1_size_, int h2_size_):
        nn(nnneighbour_), input_size(input_size_), h1_size(h1_size_), h2_size(h2_size_)
    {
        // construct and register layers
        layer1 = register_module("layer1", torch::nn::Linear(input_size, h1_size));
        layer1->to(torch::kDouble);  //use double 

        layer2 = register_module("layer2", torch::nn::Linear(h1_size, h2_size));
        layer2->to(torch::kDouble); 

        out = register_module("out", torch::nn::Linear(h2_size, 1));
        out->to(torch::kDouble); 


    }

    torch::Tensor forward(torch::Tensor X)
    {
        X = torch::relu(layer1->forward(X));
        X = torch::relu(layer2->forward(X));
        X = out->forward(X);
        
        return X;
    }

    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, out{nullptr};

};


// template<int Nstep = 1, int warmup=1000, bool cluster_updates=false, bool local_updates=true>

template<class Net_Model, class Lattice,  int warmup = 1000>

class Neural_Net : public ais::Dist_template<Sample_type, Pars_type>
{

public:

    Pars_type parameter, grad; 
    Sample_type spins; 

    int N, len; 
    Net_Model Net; 
    Lattice lat; 

    // Model Net(input_size, h1_size, h2_size); 

    // Sample_type spins;
    double beta;
    double logQ;


    Neural_Net(double beta_, int N_, Net_Model& net_, Lattice& lat_):
    beta(beta_), N(N_), Net(net_), lat(lat_)
    {
        len = Net.nn; 
        // spins.resize(N_); 
        spins = Sample_type::Zero(N_, 3); 
        this->init_grad(); 
        this->init_system(); 
    }

    

    void init_system()
    {
        for(int i=0;i<N;i++)
        {
            double x = normal(); 
            double y = normal(); 
            double z = normal(); 

            double norm = sqrt(x*x + y*y + z*z); 

            spins(i, 0) = x/norm; 
            spins(i, 1) = y/norm; 
            spins(i, 2) = z/norm; 
            // double norm_17 = std::hypot(x,y,z); 
            // spins[i].push_back(x/norm); 
            // spins[i].push_back(y/norm); 
            // spins[i].push_back(z/norm); 
        }

        this->log_dist(spins);
    }

    void mc_sweep()
    {   
        this->mc_sweep_local();
    }

    void mc_sweep_local()
    {
        for(int nattempt=0;nattempt<N;nattempt++)
        {
            double x = normal(); 
            double y = normal(); 
            double z = normal(); 

            double norm = sqrt(x*x + y*y + z*z);

            // vector<double> tmp{x/norm, y/norm, z/norm}; 
            Vector3d tmp; tmp << x/norm,y/norm,z/norm;

            int pos = rand()%N;


            vector<torch::Tensor> para_old; 
            para_old.assign(parameter.begin(), parameter.end()); 

            Vector3d ori; 
            ori = spins.row(pos); 
            // ori.assign(spins[pos].begin(), spins[pos].end()); 


            spins.row(pos) = tmp;  

            double F_old = logQ; 
            double F_new = this->log_dist(spins); 
            double dlogQ = F_new - F_old; 

            if(dlogQ < 0.0 or normal() < exp(-beta*dlogQ))
                continue; 

            // spins[pos].assign(ori.begin(), ori.end()); 
            spins.row(pos) = ori; 
            parameter.assign(para_old.begin(), para_old.end()); 

     

        }
    }

    void generate_samples(std::vector<Sample_type> &Xk,
        std::vector<double> &logQk,std::vector<Pars_type> &gradk){
        // this->init_angles();
        const int Nk = Xk.size();
        for(int i=0;i<Nk;i++){
            this->mc_sweep();
            Xk[i] = spins;
            // logQk[i] = this->log_dist(spins);;
            // gradk[i] = this->log_grad(spins);
            logQk[i] = logQ;
            gradk[i] = grad;
        }
    }

    void init_grad()
    {
        //let grad get the shape of parameter
        grad.clear();
        this->get_pars(); 

        for(int i=0;i<parameter.size();i++)
        {
            torch::Tensor x = torch::zeros(parameter[i].sizes(), torch::dtype(torch::kDouble));
            grad.push_back(x); 
        }
        
    }

    void log_grad()
    {
        grad[0] += Net.layer1->weight.grad(); 
        grad[1] += Net.layer1->bias.grad(); 

        grad[2] += Net.layer2->weight.grad(); 
        grad[3] += Net.layer2->bias.grad(); 

        grad[4] += Net.out->weight.grad(); 
        grad[5] += Net.out->bias.grad(); 

    }


    double log_dist(Sample_type x)
    {
        double F = 0.0; 
        vector<double> f; 

        for(int i=0;i<grad.size();i++)
            grad[i] *= 0.0; 
        
        for(int i=0;i<N;i++)
        {
            //obtain data from spin configuration
            vector<double> conf; 

            for(int j=0;j<lat.nn_list[i].size();j++)
            {
                conf.push_back(x(lat.nn_list[i][j], 0)); 
                conf.push_back(x(lat.nn_list[i][j], 1)); 
                conf.push_back(x(lat.nn_list[i][j], 2)); 
                // conf.push_back(x[lat.nn_list[i][j]][0]); 
                // conf.push_back(x[lat.nn_list[i][j]][1]); 
                // conf.push_back(x[lat.nn_list[i][j]][2]); 
            }

            // for(int j=-len;j<=len;j++)
            // {
            //     int pos = (i+j+N)%N; 
            //     conf.push_back(x[pos][0]); 
            //     conf.push_back(x[pos][1]); 
            //     conf.push_back(x[pos][2]); 
            // }

            //convert vector to tensor
            torch::Tensor s = torch::from_blob(conf.data(), {static_cast<long long>(conf.size())}, torch::kDouble);
            torch::Tensor out = Net.forward(s);

            //back propagation
            out.backward(); 

            this->log_grad();  

            //convert tensor to vector<double>
            f.insert(f.end(), out.data_ptr<double>(), out.data_ptr<double>() + out.numel());

            F += f[i]; 
        }
        

        logQ = F;

        return F;

    }


    void put_pars(Pars_type pars_)
    {

        Net.layer1->weight.set_data(pars_[0]); 
        Net.layer1->bias.set_data(pars_[1]); 

        Net.layer2->weight.set_data(pars_[2]); 
        Net.layer2->bias.set_data(pars_[3]); 

        Net.out->weight.set_data(pars_[4]); 
        Net.out->bias.set_data(pars_[5]); 

    }


    Pars_type get_pars()
    {
        //it is invalid to operate network parameters directly. 
        Pars_type res = Net.parameters();
        parameter.resize(res.size()); 

        for(int i=0;i<res.size();i++)
            parameter[i] = res[i].clone(); 

        return parameter; 

    }


};





#endif