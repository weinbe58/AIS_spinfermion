#ifndef _fcc_lattice_
#define _fcc_lattice_

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <ctime>
#include <unordered_map>
#include <algorithm>
using namespace std; 


double random_number()
{
	srand(time(NULL)); 
	double r = (double)(rand()/ (double)RAND_MAX);
    return r;
}

double random_theta()
{
	double r = acos(1 - 2 * random_number()); 
	return r;
}

double random_phi()
{
	double r = random_number() * M_PI * 2; 
	return r; 
}

class FCC
{

public:

	const int L; // number of atoms on the edge of fcc lattice or length+1
	const bool pbc; //true for periodic boundary condition, use false for nonperiodic
	vector<int> pos; // store position that has atom
	vector< vector<double> > spin; 

	vector< vector<int> > nn, nnn; //nearest neighbour, next nearest neighbour
	unordered_map<int, int> mp; //get the index of position in vector<int> pos
	
	vector< vector<int> > nn_dir, nnn_dir; //direction for nearest neighbour and next nearest neighbour

	int check(int x, int y, int z, int LEN, vector<int>& dir, vector<int>&pos)
	{	
		/*
			check whether it is possible to find nn or nnn for given direction. 
			return -1 for invalid direction
			return position of nn or nnn if valid
		*/

		if(pbc)
		{
			x = (x + dir[0] + LEN)%LEN; 
			y = (y + dir[1] + LEN)%LEN; 
			z = (z + dir[2] + LEN)%LEN;
		}
		else
		{
			x = x + dir[0];
			y = y + dir[1]; 
			z = z + dir[2]; 

			if(x < 0 or x > LEN -2 or y < 0 or y > LEN - 2 or z < 0 or z > LEN - 2)
				return -1; 
		}

		int num = x * LEN  * LEN + y * LEN + z; 
		int p = lower_bound(pos.begin(), pos.end(), num) - pos.begin(); 

		if(pos[p] != num)
			return -1; 
		else 
			return num; 

	}

	FCC(const int L_, const bool pbc_) : L(L_), pbc(pbc_)
	{

		int len = 2 * L - 1; 
		int LEN = L * 2; 

		pos.resize(0); 
		mp.clear(); 
		int cnt = 0; 

		// ofstream file; 
		// file.open("npbc.txt");

		nn_dir.push_back(vector<int>{1, 1, 0}); 
		nn_dir.push_back(vector<int>{1, -1, 0}); 
		nn_dir.push_back(vector<int>{1, 0, 1}); 
		nn_dir.push_back(vector<int>{1, 0, -1}); 
		nn_dir.push_back(vector<int>{0, 1, 1}); 
		nn_dir.push_back(vector<int>{0, 1, -1}); 

		nnn_dir.push_back(vector<int>{2, 0, 0}); 
		nnn_dir.push_back(vector<int>{0, 2, 0}); 
		nnn_dir.push_back(vector<int>{0, 0, 2}); 

		for(int i=0;i<6;i++)
		{
			vector<int> tmp(nn_dir[i].begin(), nn_dir[i].end()); 
			for(int j=0;j<3;j++)
				tmp[j] *= -1;
			nn_dir.push_back(tmp);  

		}

		for(int i=0;i<3;i++)
		{
			vector<int> tmp(nnn_dir[i].begin(), nnn_dir[i].end()); 
			for(int j=0;j<3;j++)
				tmp[j] *= -1;
			nnn_dir.push_back(tmp);  
		}

		/*
		cout<<"nearest neighbour"<<endl; 
		for(int i=0;i<nn_dir.size();i++)
		{
			for(int j=0;j<3;j++)
				cout<<nn_dir[i][j]<<' '; 
			cout<<endl; 
		}

		cout<<"next nearest neighbour"<<endl; 
		for(int i=0;i<nnn_dir.size();i++)
		{
			for(int j=0;j<3;j++)
				cout<<nnn_dir[i][j]<<' '; 
			cout<<endl; 
		}
		*/


		for(int k=0;k<len;k++) // z axis
		{
			// get position of atoms in each z layer
			for(int i=0;i<len;i++)
				for(int j = (k+i)%2 ; j < len ; j += 2)
				{
					int tmp = k*LEN*LEN + i*LEN + j;
					// cout<<"layer "<<k<<" ,"<<i<<' '<<j<<' '<<tmp<<endl; 
					 
					pos.push_back(tmp); 
					cnt++; 
					mp[tmp] = cnt; 
				}
		}
 		
		nn.resize(cnt); 
		nnn.resize(cnt); 
		spin.resize(cnt); 

		for(int i=0;i<cnt;i++)
		{
			spin[i].push_back(random_theta()); // or use sx, sy, sz
			spin[i].push_back(random_phi()); 
		}

		for(int i=0;i<len;i++)
			for(int j=0;j<len;j++)
				for(int k=(i+j)%2;k<len;k+=2)
				{

					int num = i * LEN*LEN + j * LEN + k; 
					int loc = mp[num] - 1;

					//find nearest neighbour
					for(int x = 0;x<nn_dir.size();x++)
					{
						int tmp = check(i, j, k, LEN, nn_dir[x], pos); 
						if(tmp >= 0)
							nn[loc].push_back(tmp); 

					}

					//find next nearest neighbour
					for(int x = 0; x < nnn_dir.size(); x ++)
					{
						int tmp = check(i, j, k, LEN, nnn_dir[x], pos); 

						if(tmp >=0 )
							nnn[loc].push_back(tmp); 
					}



					/*
					nn[loc].push_back(i * LEN*LEN + ((j + 1)%LEN) * LEN + ( k + 1)%LEN );
					nn[loc].push_back(i * LEN*LEN + ((j - 1+LEN)%LEN) * LEN + ( k - 1 + LEN)%LEN );
					nn[loc].push_back(i * LEN*LEN + ((j + 1)%LEN) * LEN + ( k - 1 + LEN)%LEN );
					nn[loc].push_back(i * LEN*LEN + ((j - 1 + LEN)%LEN) * LEN + ( k + 1)%LEN );

					nn[loc].push_back(((i + 1)%LEN) * LEN*LEN + ((j + 1)%LEN) * LEN + k );
					nn[loc].push_back( ((i - 1 + LEN)%LEN) * LEN*LEN + ((j - 1 + LEN)%LEN) * LEN + k );
					nn[loc].push_back(((i + 1)%LEN) * LEN*LEN + ((j - 1 + LEN)%LEN) * LEN + k );
					nn[loc].push_back(((i - 1 + LEN)%LEN) * LEN*LEN + ((j + 1)%LEN) * LEN + k );


					nn[loc].push_back(((i + 1)%LEN) * LEN*LEN + j * LEN + ( k + 1)%LEN );
					nn[loc].push_back(((i - 1 + LEN)%LEN) * LEN*LEN + j * LEN + ( k - 1 + LEN)%LEN );
					nn[loc].push_back(((i + 1)%LEN) * LEN*LEN + j * LEN + ( k - 1 + LEN)%LEN );
					nn[loc].push_back(((i - 1 + LEN)%LEN) * LEN*LEN + j * LEN + ( k + 1)%LEN );
					

					for(auto it = nn[loc].begin(); it != nn[loc].end(); )
					{
						int p = lower_bound(pos.begin(), pos.end(), *it) - pos.begin(); 
						if(*it != pos[p])
							nn[loc].erase(it); 
						else 
							it++; 
					}

					nnn[loc].push_back( ((i-2+LEN)%LEN)*LEN*LEN + j*LEN + k); 
					nnn[loc].push_back( ((i+2)%LEN)*LEN*LEN + j*LEN + k); 

					nnn[loc].push_back(i*LEN*LEN + ((j-2+LEN)%LEN)*LEN + k ); 
					nnn[loc].push_back(i*LEN*LEN + ((j+2)%LEN)*LEN + k ); 

					nnn[loc].push_back(i*LEN*LEN + j*LEN + (k-2+LEN)%LEN ); 
					nnn[loc].push_back(i*LEN*LEN + j*LEN + (k+2)%LEN ); 

					for(auto it = nnn[loc].begin(); it != nnn[loc].end(); )
					{
						int p = lower_bound(pos.begin(), pos.end(), *it) - pos.begin(); 
						if(*it != pos[p])
							nnn[loc].erase(it); 
						else 
							it++; 
					}
					*/


					//output
					/*
					sort(nn[loc].begin(), nn[loc].end()); 
					sort(nnn[loc].begin(), nnn[loc].end()); 
					
					file<<loc<<' '<<nn[loc].size()<<' '<<nnn[loc].size()<<endl;
					for(int x = 0; x < nn[loc].size(); x++) file<<nn[loc][x]<<' '; 
					file<<endl; 
					for(int x = 0; x < nnn[loc].size(); x++) file<<nnn[loc][x]<<' '; 
					file<<endl; 
					 */

				}


	
	} 

	

};





#endif
