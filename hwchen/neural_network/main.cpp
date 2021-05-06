#include <torch/torch.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>
// #include <Eigen/Dense>
#include "neural_network.h"

using namespace std; 



int main() 
{

    //basic tensor operation

    torch::Tensor t = torch::randn({3,3}); 
    torch::Tensor s = torch::randn({3,3}); 

    cout<<"original tensor t"<<endl<<t<<endl<<endl;

    cout<<"original tensor s"<<endl<<s<<endl<<endl; 

    //operation with scalar
    /*
    t += 2; 
    cout<<"addition"<<endl<<t<<endl<<endl; 

    t -= 2; 
    cout<<"subtraction"<<endl<<t<<endl<<endl;

    t *= 2; 
    cout<<"multiplication"<<endl<<t<<endl<<endl;

    t /= 2;
    cout<<"division"<<endl<<t<<endl<<endl;
    */

    //element wise multiplication
    torch::Tensor r1 = t * s;
    //matrix multiplication
    torch::Tensor r2 = torch::matmul(t, s); 

    cout<<"element wise multiplication"<<endl<<r1<<endl<<endl; 
    cout<<"matrix multiplication"<<endl<<r2<<endl<<endl; 

    //element-wise operation

    //get square root of elements in matrix
    r1 = torch::sqrt(t); //or
    r2 = t.sqrt(); 

    cout<<"using torch::sqrt"<<endl<<r1<<endl<<endl; 
    cout<<"using tensor.sqrt()"<<endl<<r2<<endl<<endl; 



    //test neural network model
    int nneighbour = 5; 
    int input_size = (nneighbour*2 + 1) * 3; 
    int h1_size = 20, h2_size = 10; 


    Model Net(nneighbour, input_size, h1_size, h2_size);  


    torch::Tensor x = torch::randn({input_size}, torch::dtype(torch::kDouble)); //create input tensor
    auto out = Net.forward(x); 

    //convert tensor to vector<double> to get output
    vector<double> output; 
    output.insert(output.end(), out.data_ptr<double>(), out.data_ptr<double>() + out.numel()); 
    double f = output[0]; 

    out.backward();  //back propagation

   

    double beta = 0.1, N = 50; 

    net_range_1D lat(N, nneighbour);

    Neural_Net<Model, net_range_1D, 1000, false, true> Q(beta, N, Net, lat);  

    //test get_pars, use one tensor as an example
    cout<<"result from Model"<<endl<<Q.Net.out->weight<<endl<<endl; 
    auto xx = Q.get_pars(); 
    cout<<"result from get_pars"<<endl<<xx[4]<<endl<<endl; 


    for(int i=0;i<xx.size();i++)
        xx[i] *= 2.0; 

    Q.put_pars(xx);

    cout<<"weight * 2"<<endl<<Q.Net.out->weight<<endl<<endl;





  return 0; 
}