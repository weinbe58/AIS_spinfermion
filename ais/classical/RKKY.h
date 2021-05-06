#ifndef _spin_fermion_
#define _spin_fermion_ value

#include <stack>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

#include "ais/random.h"
#include "ais/classical/heis.h"
#include "ais/fcc_lattice.h"
#include "ais/ais.h"




class long_range_1D
{
	const int N,R;
	std::vector<std::vector<std::pair<int,int>>> nn_lists,bond_lists;
	// first : index of site
	// second : index in Jr
	typedef std::vector<std::pair<int,int>>::iterator pair_iter;

public:
	long_range_1D(const int N_,const int R_) : N(N_), R(R_) {
		nn_lists.resize(N_);
		bond_lists.resize(N_);


		for(int i=0;i<N_;i++){
			for(int r=1;r<=R_;r++){
				const int jp = (i+r)%N;
				const int jm = (i-r+N)%N;

				nn_lists[i].push_back(std::make_pair(jm,r-1));
				nn_lists[i].push_back(std::make_pair(jp,r-1));
				bond_lists[i].push_back(std::make_pair(jp,r-1));
			}
		}
	}
	~long_range_1D(){}


	inline size_t N_nn(const int i) const {
		return nn_lists[i].size();
	}

	inline size_t N_nn_dist(const int i) const {
		return bond_lists[0].size();
	}

	inline
	pair_iter nn_begin(const int i){
		return nn_lists[i].begin();
	}

	inline
	pair_iter nn_end(const int i){
		return nn_lists[i].end();
	}

	inline
	pair_iter bonds_begin(const int i){
		return bond_lists[i].begin();
	}

	inline
	pair_iter bonds_end(const int i){
		return bond_lists[i].end();
	}

	inline
	int neighbor(const int i,const int r){
		return nn_lists[i][r].first;
	}
};




template<class Lattice,int Nstep = 1,int warmup=1000,bool cluster_updates=false,bool local_updates=true>
class RKKY_gen : public ais::Dist_template<O3::Sample_type,O3::Pars_type> 
{
	/*
		TODO: set up method to update logQ after cluster update 
		      without calculating it from scratch in O(N*N_nn) operations.
	*/
public:
	Pars_type Jr,grad; // shape (Nd,): coupling constant
	Lattice * lat; 

	Sample_type spins;
	rand_uniform ran;
	rand_normal normal;
	const double beta;
	const int N;
	double logQ;

	RKKY_gen(const double beta_,const int N_,Lattice * lat_) : 
	beta(beta_), N(N_), lat(lat_) {
		const int N_Jr = lat_->N_nn_dist();
		ran = rand_uniform();
		normal = rand_normal();
		spins = Sample_type::Zero(N_,3);
		Jr = Pars_type::Zero(N_Jr);

		for(int r=0;r<N_Jr;r++){
			Jr[r] = normal();
		}

		this->init_system();
	}

	RKKY_gen(const double beta_,const int N_,Pars_type Jr_,Lattice * lat_) : 
	beta(beta_), N(N_), lat(lat_) {
		ran = rand_uniform();
		normal = rand_normal();
		spins = Sample_type::Zero(N_,3);
		Jr = Jr_;
		this->init_system();
	}

	RKKY_gen(RKKY_gen<Lattice,Nstep,warmup> &other) : 
	beta(other.beta), N(other.N), lat(other.lat) {
		ran = rand_uniform();
		normal = rand_normal();
		spins = other.spins;
		Jr = other.Jr;
		logQ = this->log_dist(spins);
		grad = this->log_grad(spins);
		for(int i=0;i<warmup;i++){this->mc_sweep();}
	}

	void init_system(){
		for(int i=0;i<N;i++){
			const double x = normal();
			const double y = normal();
			const double z = normal();
			const double norm = std::hypot(x,y,z);

			spins(i,0) = x/norm;
			spins(i,1) = y/norm;
			spins(i,2) = z/norm;

		}
		
		logQ = this->log_dist(spins);
		grad = this->log_grad(spins);

		for(int i=0;i<warmup;i++){
			// std::cout << std::setw(10) << i << std::setw(10) << spins.transpose() << std::endl;
			this->mc_sweep();

		}
	}

	void mc_sweep(){
		
		// if constexpr(cluster_updates){
		// 	this->mc_sweep_cluster();				
		// }

		if constexpr(local_updates){
			this->mc_sweep_local();
		}
	}

	void mc_sweep_local(){
		for(int nattempt=0;nattempt<N;nattempt++){
			const double x = normal();
			const double y = normal();
			const double z = normal();
			const double norm = std::hypot(x,y,z);

			O3::Vector3d r;	r << x/norm,y/norm,z/norm;

			const int i = (int) std::floor(N*ran());
			O3::Vector3d si = spins.row(i);

			O3::Vector3d ds = r - si;


			double dlogQ = 0;
			Pars_type dgrad  = 0*grad;


			auto nn = lat->nn_begin(i);
			auto nn_end = lat->nn_end(i);
			for(;nn!=nn_end;nn++){
				const int j = nn->first;
				const double J = Jr[nn->second];
				O3::Vector3d sj = spins.row(j);
				const double sdots = sj.dot(ds);

				dgrad[nn->second] += sdots;
				dlogQ += J * sdots;
			}

			if(dlogQ < 0 || ran() < std::exp(-beta*dlogQ)){
				logQ += dlogQ;
				grad += dgrad;
				spins.row(i) = r;

			}

		}
	}


	// void mc_sweep_cluster(){

	// 	const int N_nn = lat->N_nn();

	// 	std::stack<int> cluster_stack;
	// 	std::stack<int> site_stack;
	// 	std::stack<double> proj_stack;
	// 	std::vector<bool> visited(N,false);
	// 	std::vector<double> p,C(N_nn,0.0);

	// 	p.reserve(N_nn);

	// 	for(int nsweep=0;nsweep<Nstep;nsweep++){
	// 		// get random axis of reflection.
	// 		const double x = normal();
	// 		const double y = normal();
	// 		const double z = normal();
	// 		const double norm = std::hypot(x,y,z);

	// 		O3::Vector3d r;	r << x/norm,y/norm,z/norm;
	// 		cluster_stack = std::stack<int>();
	// 		site_stack = std::stack<int>();
	// 		proj_stack = std::stack<double>();

	// 		p.clear();

	// 		std::fill(visited.begin(),visited.end(),false);
	// 		const int seed = (int) std::floor(N*ran());

	// 		cluster_stack.push(seed);
	// 		visited[seed] = true;

	// 		while(!cluster_stack.empty()){
	// 			const int i = cluster_stack.top(); cluster_stack.pop();
	// 			O3::Vector3d si = spins.row(i);
	// 			const double proj_i = r.dot(si);
	// 			spins.row(i) = si - 2 * proj_i * r;

	// 			proj_stack.push(proj_i);
	// 			site_stack.push(i);

	// 			// std::cout << "stack site:" << std::setw(5) << i << std::endl;

	// 			// setting up probability tables
	// 			p.clear(); // p_n
	// 			auto nn = lat->nn_begin(i);
	// 			auto nn_end = lat->nn_end(i);

	// 			for(;nn != nn_end; nn++){
	// 				const int j = nn->first;

	// 				if(!visited[j]){ // if site not previously visited, calculate probability
	// 					const double J = Jr[nn->second];
	// 					O3::Vector3d sj = spins.row(j);
	// 					const double proj_j = r.dot(sj);
	// 					const double exp_p = std::min(0.0,2.0*J*beta*proj_i*proj_j);

	// 					p.push_back(1-std::exp(exp_p));			
	// 				}
	// 				else{ // otherwise set probability to 0 to skip in next step
	// 					p.push_back(0.0);
	// 				}

	// 				// std::cout << "nn site:" << std::setw(5) << j << " probability:" << std::setw(10) << p.back() << std::endl;
	// 			} // end for(;nn != nn_end; nn++)

	// 			int k = 0;

	// 			while(true){// sample the cumulant distribution for adding neighbors. 

	// 				double pp = 1.0;
	// 				for(int l=0;l<k;l++){ C[l] = 0.0; }
	// 				for(int l=k;l<N_nn;l++){
	// 					pp *= (1-p[l]);
	// 					C[l] = 1.0-pp; 
	// 				}



	// 				auto lb = std::lower_bound(C.begin()+k,C.end(),ran());
	// 				k = (int)(lb - C.begin());

	// 				if(lb != C.end()){ // if k==Nn no more sites can be added.
	// 					const int j = lat->neighbor(i,k);
	// 					if(!visited[j]){
	// 						// std::cout << "adding site: " << std::setw(5) << j << std::endl;
	// 						cluster_stack.push(j);
	// 						visited[j] = true;							
	// 					}

	// 					k++;
	// 				}
	// 				else{
	// 					break;
	// 				} // end if(k < N_nn)

	// 			} // end while(k < N_nn)

	// 		} // end while(!stack.empty())


	// 		while(!site_stack.empty()){ // updating logQ and grad after cluster update
	// 			const int i = site_stack.top(); site_stack.pop();
	// 			const double proj_i = proj_stack.top(); proj_stack.pop();

	// 			auto nn = lat->nn_begin(i);
	// 			auto nn_end = lat->nn_end(i);

	// 			for(;nn!=nn_end;nn++){
	// 				const int j = nn->first;
	// 				if(!visited[j]){
	// 					O3::Vector3d sj = spins.row(j);
	// 					const double proj_j = r.dot(sj);
	// 					const double K = 2 * beta * proj_i * proj_j;
	// 					const int k = nn->second;

	// 					logQ += Jr[k] * K;
	// 					grad[k] += K;
	// 				} // end if(!visited[j])

	// 			} // end for(;nn!=nn_end;nn++)

	// 		} // end while(!site_stack.empty())
	// 	}
	// }

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

	double log_dist(Sample_type x){
		double F=0;

		for(int i=0;i<N;i++){
			double F_local = 0;
			O3::Vector3d si = x.row(i);

			auto nn = lat->bonds_begin(i);
			auto nn_end = lat->bonds_end(i);

			for(;nn != nn_end; nn++){
				O3::Vector3d sj = x.row(nn->first);
				F_local += Jr[nn->second] * si.dot(sj);
			}

			F += F_local;
		}

		return -beta * F;
	}

	Pars_type log_grad(Sample_type x){
		Pars_type grad = Pars_type::Zero(Jr.size());

		for(int i=0;i<N;i++){
			O3::Vector3d si = x.row(i);

			auto nn = lat->bonds_begin(i);
			auto nn_end = lat->bonds_end(i);

			for(;nn != nn_end;nn++){
				O3::Vector3d sj = x.row(nn->first);
				grad[nn->second] += si.dot(sj);
			}
		}

		return -beta * grad;
	}

	Pars_type get_pars(void){
		return Jr;
	}

	void put_pars(Pars_type Jr_){
		Jr = Jr_;
	}

};



template<class Lattice,int Nstep = 1,int warmup=1000,bool cluster_updates=false,bool local_updates=true>
class RKKY_imp : public ais::Dist_template<O3::Sample_type,O3::Pars_type> 
{
	/*
		TODO: set up method to update logQ after cluster update 
		      without calculating it from scratch in O(N*N_nn) operations.
	*/
public:
	Pars_type Jr,grad; // shape (Nd,): coupling constant
	Lattice * lat; 

	Sample_type spins;
	rand_uniform ran;
	rand_normal normal;
	const double beta;
	const int N;
	int Nimp = 0;
	double logQ,mu_imp;

	RKKY_imp(const double beta_,const double mu_imp_,const int N_,Lattice * lat_) : 
	mu_imp(mu_imp_), beta(beta_), N(N_), lat(lat_) {
		const int N_Jr = lat_->N_nn_dist();
		ran = rand_uniform();
		normal = rand_normal();
		spins = Sample_type::Zero(2*N_,3);
		Nimp = 0;
		Jr = Pars_type::Zero(N_Jr);

		for(int r=0;r<N_Jr;r++){
			Jr[r] = normal();
		}

		this->init_system();
	}

	RKKY_imp(const double beta_,const double mu_imp_,const int N_,Pars_type Jr_,Lattice * lat_) : 
	mu_imp(mu_imp_), beta(beta_), N(N_), lat(lat_) {
		ran = rand_uniform();
		normal = rand_normal();
		spins = Sample_type::Zero(2*N_,3);
		Jr = Jr_;
		this->init_system();
	}

	RKKY_imp(RKKY_imp<Lattice,Nstep,warmup> &other) : 
	beta(other.beta), N(other.N), lat(other.lat) {
		ran = rand_uniform();
		normal = rand_normal();
		spins = other.spins;
		Jr = other.Jr;
		logQ = this->log_dist(spins);
		grad = this->log_grad(spins);
		for(int i=0;i<warmup;i++){this->mc_sweep();}
	}

	void init_system(){
		for(int i=0;i<N;i++){
			const double x = normal();
			const double y = normal();
			const double z = normal();
			const double norm = std::hypot(x,y,z);

			spins(i,0) = x/norm;
			spins(i,1) = y/norm;
			spins(i,2) = z/norm;

		}
		
		logQ = this->log_dist(spins);
		grad = this->log_grad(spins);

		for(int i=0;i<warmup;i++){

			this->mc_sweep();

		}
	}

	void mc_sweep(){

		// if constexpr(cluster_updates){
		// 	this->mc_sweep_cluster();				
		// }

		if constexpr(local_updates){
			this->mc_sweep_local();
		}

		for(int nattempt=0;nattempt<N;nattempt++){

			const int i = (int) std::floor(N*ran());

			if(spins(N+i,0) == 0.0){
				if(mu_imp <= 0 || ran() < std::exp(-mu_imp)){
					spins(N+i,0) = 1.0;
					Nimp++;
				}
			}
			else{
				if(mu_imp >= 0 || ran() < std::exp(mu_imp)){
					spins(N+i,0) = 0.0;
					Nimp--;
				}
			}
		}
	}

	void mc_sweep_local(){
		for(int nattempt=0;nattempt<N;nattempt++){
			const double x = normal();
			const double y = normal();
			const double z = normal();
			const double norm = std::hypot(x,y,z);

			O3::Vector3d r;	r << x/norm,y/norm,z/norm;

			const int i = (int) std::floor(N*ran());
			O3::Vector3d si = spins.row(i);

			O3::Vector3d ds = r - si;


			double dlogQ = 0;
			Pars_type dgrad  = 0*grad;


			auto nn = lat->nn_begin(i);
			auto nn_end = lat->nn_end(i);
			for(;nn!=nn_end;nn++){
				const int j = nn->first;

				const double J = Jr[nn->second];
				O3::Vector3d sj = spins.row(j);
				const double sdots = sj.dot(ds);

				dgrad[nn->second] += sdots;
				dlogQ += J * sdots;
			}

			if(dlogQ < 0 || ran() < std::exp(-beta*dlogQ)){
				logQ += dlogQ;
				grad += dgrad;
				spins.row(i) = r;

			}

		}
	}


	// void mc_sweep_cluster(){

	// 	const int N_nn = lat->N_nn();

	// 	std::stack<int> cluster_stack;
	// 	std::stack<int> site_stack;
	// 	std::stack<double> proj_stack;
	// 	std::vector<bool> visited(N,false);
	// 	std::vector<double> p,C(N_nn,0.0);

	// 	p.reserve(N_nn);

	// 	for(int nsweep=0;nsweep<Nstep;nsweep++){
	// 		// get random axis of reflection.
	// 		const double x = normal();
	// 		const double y = normal();
	// 		const double z = normal();
	// 		const double norm = std::hypot(x,y,z);

	// 		O3::Vector3d r;	r << x/norm,y/norm,z/norm;
	// 		cluster_stack = std::stack<int>();
	// 		site_stack = std::stack<int>();
	// 		proj_stack = std::stack<double>();

	// 		p.clear();

	// 		std::fill(visited.begin(),visited.end(),false);
	// 		const int seed = (int) std::floor(N*ran());

	// 		cluster_stack.push(seed);
	// 		visited[seed] = true;

	// 		while(!cluster_stack.empty()){
	// 			const int i = cluster_stack.top(); cluster_stack.pop();
	// 			O3::Vector3d si = spins.row(i);
	// 			const double proj_i = r.dot(si);
	// 			spins.row(i) = si - 2 * proj_i * r;

	// 			proj_stack.push(proj_i);
	// 			site_stack.push(i);

	// 			// std::cout << "stack site:" << std::setw(5) << i << std::endl;

	// 			// setting up probability tables
	// 			p.clear(); // p_n
	// 			auto nn = lat->nn_begin(i);
	// 			auto nn_end = lat->nn_end(i);

	// 			for(;nn != nn_end; nn++){
	// 				const int j = nn->first;

	// 				if(!visited[j]){ // if site not previously visited, calculate probability
	// 					const double J = Jr[nn->second];
	// 					O3::Vector3d sj = spins.row(j);
	// 					const double proj_j = r.dot(sj);
	// 					const double exp_p = std::min(0.0,2.0*J*beta*proj_i*proj_j);

	// 					p.push_back(1-std::exp(exp_p));			
	// 				}
	// 				else{ // otherwise set probability to 0 to skip in next step
	// 					p.push_back(0.0);
	// 				}

	// 				// std::cout << "nn site:" << std::setw(5) << j << " probability:" << std::setw(10) << p.back() << std::endl;
	// 			} // end for(;nn != nn_end; nn++)

	// 			int k = 0;

	// 			while(true){// sample the cumulant distribution for adding neighbors. 

	// 				double pp = 1.0;
	// 				for(int l=0;l<k;l++){ C[l] = 0.0; }
	// 				for(int l=k;l<N_nn;l++){
	// 					pp *= (1-p[l]);
	// 					C[l] = 1.0-pp; 
	// 				}



	// 				auto lb = std::lower_bound(C.begin()+k,C.end(),ran());
	// 				k = (int)(lb - C.begin());

	// 				if(lb != C.end()){ // if k==Nn no more sites can be added.
	// 					const int j = lat->neighbor(i,k);
	// 					if(!visited[j]){
	// 						// std::cout << "adding site: " << std::setw(5) << j << std::endl;
	// 						cluster_stack.push(j);
	// 						visited[j] = true;							
	// 					}

	// 					k++;
	// 				}
	// 				else{
	// 					break;
	// 				} // end if(k < N_nn)

	// 			} // end while(k < N_nn)

	// 		} // end while(!stack.empty())


	// 		while(!site_stack.empty()){ // updating logQ and grad after cluster update
	// 			const int i = site_stack.top(); site_stack.pop();
	// 			const double proj_i = proj_stack.top(); proj_stack.pop();

	// 			auto nn = lat->nn_begin(i);
	// 			auto nn_end = lat->nn_end(i);

	// 			for(;nn!=nn_end;nn++){
	// 				const int j = nn->first;
	// 				if(!visited[j]){
	// 					O3::Vector3d sj = spins.row(j);
	// 					const double proj_j = r.dot(sj);
	// 					const double K = 2 * beta * proj_i * proj_j;
	// 					const int k = nn->second;

	// 					logQ += Jr[k] * K;
	// 					grad[k] += K;
	// 				} // end if(!visited[j])

	// 			} // end for(;nn!=nn_end;nn++)

	// 		} // end while(!site_stack.empty())
	// 	}
	// }

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

	double log_dist(Sample_type x){
		double F=0;

		for(int i=0;i<N;i++){
			double F_local = 0;
			O3::Vector3d si = x.row(i);

			auto nn = lat->bonds_begin(i);
			auto nn_end = lat->bonds_end(i);

			for(;nn != nn_end; nn++){
				O3::Vector3d sj = x.row(nn->first);
				F_local += Jr[nn->second] * si.dot(sj);
			}

			F += F_local;
		}

		return -beta * F - mu_imp * Nimp;
	}

	Pars_type log_grad(Sample_type x){
		Pars_type grad = Pars_type::Zero(Jr.size());

		for(int i=0;i<N;i++){
			O3::Vector3d si = x.row(i);

			auto nn = lat->bonds_begin(i);
			auto nn_end = lat->bonds_end(i);

			for(;nn != nn_end;nn++){
				O3::Vector3d sj = x.row(nn->first);
				grad[nn->second] += si.dot(sj);
			}
		}

		return -beta * grad;
	}

	Pars_type get_pars(void){
		return Jr;
	}

	void put_pars(Pars_type Jr_){
		Jr = Jr_;
	}

};



#endif