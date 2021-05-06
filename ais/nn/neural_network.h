#ifndef _neural_network_
#define _neural_network_

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <torch/torch.h>

#include "ais/ais.h"
#include "ais/optimize.h"
#include "ais/random.h"


#include <Eigen/Dense>



using namespace std; 

typedef Eigen::Matrix<double,3,1> _nn_Vector3d;
typedef Eigen::Array<double,Eigen::Dynamic,3,Eigen::RowMajor> _nn_Sample_type;
typedef vector< torch::Tensor > _nn_Pars_type; 



template<class T>
_nn_Pars_type operator*(const T& a, const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = b[i] * a; 

    return res; 
}

template<class T>
_nn_Pars_type operator*(const _nn_Pars_type& b, const T& a)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = b[i] * a; 

    return res; 
}

template<class T>
_nn_Pars_type operator/(const _nn_Pars_type& b, const T& a)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = b[i] / a; 

    return res; 
}

template<class T>
_nn_Pars_type operator/(const T& a, const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a / b[i]; 

    return res; 
}

template<class T>
_nn_Pars_type operator+(const _nn_Pars_type& b,const T& a)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a + b[i]; 

    return res; 
}

template<class T>
_nn_Pars_type operator+(const T& a, const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = b[i] + a; 

    return res; 
}



template<class T>
_nn_Pars_type operator-(const T& a, const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a - b[i]; 

    return res; 
}

template<class T>
_nn_Pars_type operator-(const _nn_Pars_type& b, const T& a)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = b[i] - a; 

    return res; 
}




_nn_Pars_type operator+(const _nn_Pars_type& a,const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a[i] + b[i]; 

    return res; 
}

_nn_Pars_type operator-(const _nn_Pars_type& a,const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a[i] - b[i]; 

    return res; 
}

_nn_Pars_type operator*(const _nn_Pars_type& a,const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a[i] * b[i]; 

    return res; 
}


_nn_Pars_type operator/(const _nn_Pars_type& a,const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size()); 

    for(int i=0;i<res.size();i++)
        res[i] = a[i] / b[i]; 

    return res; 
}

namespace sgd {

template<>
_nn_Pars_type sqrt<_nn_Pars_type>(const _nn_Pars_type& b)
{
    _nn_Pars_type res; 
    res.resize(b.size());

    for(int i=0;i<res.size();i++)
        res[i] = torch::sqrt(b[i]); 

    return res; 
} 

}

   






namespace nn {




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





// template<int Nstep = 1, int warmup=1000, bool cluster_updates=false, bool local_updates=true>

template<class Net_Model, class Lattice,  int warmup = 1000, int ndecorr=100>

class Neural_Net : public ais::Dist_template<_nn_Sample_type, _nn_Pars_type>
{

public:

    _nn_Pars_type parameter, grad; 
    _nn_Sample_type spins; 

    const int N, len,N_pars; 
    Net_Model Net; 
    Lattice lat; 

    // Model Net(input_size, h1_size, h2_size); 

    // _nn_Sample_type spins;
    const double beta;
    double logQ;
    rand_normal normal;
    rand_uniform ran;


    Neural_Net(double beta_, int N_, Net_Model& net_, Lattice& lat_):
    beta(beta_), N(N_), len(Net.nn), Net(net_), lat(lat_), N_pars(net_.parameters().size())
    {
        normal = rand_normal();
        ran = rand_uniform();
        
        // spins.resize(N_); 
        spins = _nn_Sample_type::Zero(N_, 3); 
        this->init_grad(); 
        this->init_system(); 
    }

    void init_grad()
    {
        //let grad get the shape of parameter
        grad.clear();
        parameter = Net.parameters(); 

        for(int i=0;i<N_pars;i++)
        {

            torch::Tensor x = torch::zeros(parameter[i].sizes(), torch::dtype(torch::kDouble));
            grad.push_back(x); 
        }
        
    }

    void init_system()
    {
        for(int i=0;i<N;i++)
        {
            double x = normal(); 
            double y = normal(); 
            double z = normal(); 

            double norm = hypot(x,y,z); 

            spins(i, 0) = x/norm; 
            spins(i, 1) = y/norm; 
            spins(i, 2) = z/norm; 
            // double norm_17 = std::hypot(x,y,z); 
            // spins[i].push_back(x/norm); 
            // spins[i].push_back(y/norm); 
            // spins[i].push_back(z/norm); 
        }

        this->log_dist(spins);

        for(int i=0;i<warmup;i++){
            this->mc_sweep();
        }



    }

    void mc_sweep()
    {   
        for(int i=0;i<ndecorr;i++){
            this->mc_sweep_local();            
        }

    }

    void mc_sweep_local()
    {
        for(int nattempt=0;nattempt<N;nattempt++)
        {
            double x = normal(); 
            double y = normal(); 
            double z = normal(); 

            double norm = hypot(x,y,z);

            // vector<double> tmp{x/norm, y/norm, z/norm}; 
            _nn_Vector3d tmp; tmp << x/norm,y/norm,z/norm;

            int pos = static_cast<int>(floor(ran()*N));


            _nn_Vector3d ori; 
            ori = spins.row(pos); 

            spins.row(pos) = tmp;  


            double F_new = this->eval_Net(spins,false);
            double dlogQ = F_new - logQ;

            if(dlogQ < 0.0 or ran() < exp(-dlogQ)){
                logQ = F_new; 
            }
            else{
                spins.row(pos) = ori; 
            }
     

        }
    }

    void generate_samples(std::vector<_nn_Sample_type> &Xk,
        std::vector<double> &logQk,std::vector<_nn_Pars_type> &gradk){
        // this->init_angles();
        const int Nk = Xk.size();
        for(int i=0;i<Nk;i++){
            this->mc_sweep();
            Xk[i] = spins;
            logQk[i] = this->eval_Net(spins,true);
            gradk[i].clear();
            for(int g=0;g<grad.size();g++){
                gradk[i].push_back(-beta * grad[g].clone());
            }
            
        }
    }



    // void log_grad()
    // {
    //     grad[0] += Net.layer1->weight.grad(); 
    //     grad[1] += Net.layer1->bias.grad(); 

    //     grad[2] += Net.layer2->weight.grad(); 
    //     grad[3] += Net.layer2->bias.grad(); 

    //     grad[4] += Net.out->weight.grad(); 
    //     grad[5] += Net.out->bias.grad(); 

    // }


    double eval_Net(_nn_Sample_type x,const bool backprop=false){
        
        double F = 0.0; 
        vector<double> f,conf; 

        if(backprop)
        {
            for(int i=0;i<grad.size();i++)
                grad[i] *= 0.0;            
        }

        
        for(int i=0;i<N;i++)
        {
            //obtain data from spin configuration

            
            conf.resize(lat.nn_list[i].size());
            conf.clear();

            for(int j=0;j<lat.nn_list[i].size();j++)
            {
                conf.push_back(x(lat.nn_list[i][j], 0)); 
                conf.push_back(x(lat.nn_list[i][j], 1)); 
                conf.push_back(x(lat.nn_list[i][j], 2)); 
            }

            //convert vector to tensor

            torch::Tensor s = torch::from_blob(conf.data(), {static_cast<long long>(conf.size())}, torch::kDouble);
            if(backprop){ Net.zero_grad(); }
            torch::Tensor out = Net.forward(s);

            if(backprop){
                //back propagation
                out.backward(); 
                parameter = Net.parameters();
                for(int g=0;g<N_pars;g++){
                    grad[g] += parameter[g].grad();
                }

                // this->log_grad();                 
            }

            F += *out.data_ptr<double>();

        }

        return -beta * F;        
    }


    double log_dist(_nn_Sample_type x) {return eval_Net(x,false);}


    void put_pars(_nn_Pars_type pars_)
    {
        // torch::NoGradGuard no_grad;
        // Net.layer1->weight.copy_(pars_[0]);
        // Net.layer1->bias.copy_(pars_[1]);
        // Net.layer2->weight.copy_(pars_[2]);
        // Net.layer2->bias.copy_(pars_[3]);
        // Net.out->weight.copy_(pars_[4]);
        // Net.out->bias.copy_(pars_[5]);
        Net.update_(pars_);

    }


    _nn_Pars_type get_pars()
    {
        //it is invalid to operate network parameters directly. 
        parameter = Net.parameters();

        _nn_Pars_type res;

        for(int i=0;i<parameter.size();i++)
            res.push_back(parameter[i].clone());

        return res; 

    }


};


}


#endif