#ifndef _random_
#define _random_ value

#include <random>
#include <queue>
#include <cmath>
#include <algorithm>


class rand_normal
{
	std::mt19937_64 gen;
	std::normal_distribution<double> dist;
	unsigned int s;

	public:
		
		rand_normal(){
			//seeding random number generator
			unsigned int lo,hi;
			__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
			s=((unsigned long long)hi << 32) | lo;

			gen.seed(s);
			dist = std::normal_distribution<double>(0.0,1.0);
		};



		inline double operator()(void){
			return dist(gen);
		}

		inline 
		unsigned int seed(void) const {return s;}

};


class rand_uniform
{
	std::mt19937_64 gen;
	std::uniform_real_distribution<double> dist;
	unsigned int s;

	public:


		rand_uniform(){
			//seeding random number generator
			unsigned int lo,hi;
			__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
			s=((unsigned long long)hi << 32) | lo;

			gen.seed(s);
			dist = std::uniform_real_distribution<double>(0.0,1.0);
		};

		inline double operator()(void){
			return dist(gen);
		}

		inline
		unsigned int seed(void) const {return s;}

};


/*
template<class T>
class select_random
{
	const size_t N;
	rand_uniform ran;

	std::vector<T> P;
	std::vector<size_t> A;

	select_random(T * weights,const size_t N_): N(N_){
		ran = rand_uniform();
		P.reserve(N_);
		A.reserve(N_);
		A.insert(A.end(),N,-1);

		std::queue<size_t> under,over;


		T norm = 0;
		for(size_t i=0;i<N_;i++){
			norm += weights_[i];
		}

		norm /= N_;

		for(size_t i=0;i<N_;i++){
			const T p = weights[i] / norm;
			if(p > 1){
				over.push(i);
			}
			else if (p < 1){
				under.push(i);
			}

			P.push_back(p);
		}


		while(true){
			const size_t i = over.pop();
			const size_t j = under.pop();
			
			const T res = P[i] + P[j] - 1.0;

			P[i] = res;
			A[j] = i;

			if(res > 1){
				over.push(i);
			}
			else if (res < 1){
				under.push(i);
			}

			if(over.empty() || under.empty()){
				break;
			}
		}
	}

	inline
	size_t operator()(){
		const double q = N * ran();
		const size_t i = (size_t)std::floor(q);
		if((q-i) < P[i]){
			return i;
		}
		else{
			return A[i];
		}

	}

	template<class S>
	std::pair<S,S> bootstrap(S * data,const int Nbootstrap=1000){
		S mean=0,err=0; 

		for(size_t i=0;i<N;i++){
			mean += data[i];
		}

		mean /= N;

		for(int i=0;i<Nbootstrap;i++){
			S sample = 0;
			for(int j=0;j<N;j++){
				size_t ind = this->operator()();
				sample += data[ind];
			}
			sample /= N;
			err += (mean - sample)*(mean - sample);
		}

		err = std::sqrt(err/N);
		return std::make_pair(mean,err);
	}

};
*/

#endif