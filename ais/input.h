#ifndef _input_h_
#define _input_h_


#include <string>
#include <fstream>
#include <map>
#include <set>
#include <exception>

namespace ais {

auto process_inputfile(std::string infile,
	std::set<std::string> int_inputs={},
	std::set<std::string> double_inputs={})
{
	std::map<std::string,int> int_values;
	std::map<std::string,double> double_values;




	for(auto int_input : int_inputs){

		std::ifstream inFile(infile.c_str());    
		std::string line;
		
		while(std::getline(inFile, line)){
			std::stringstream linestream(line);
			std::string segment;
			std::vector<std::string> seglist;

			while(std::getline(linestream, segment, '=')){seglist.push_back(segment);}

			if(seglist.size()!=2){throw std::runtime_error("invalid line format input file.");}

			if(seglist[0] == int_input){
				int_values[int_input] = std::stoi(seglist[1]);
			}
		}


	}

	for(auto double_input : double_inputs){
		std::ifstream inFile(infile.c_str());    
		std::string line;

		while(std::getline(inFile, line)){
			std::stringstream test(line);
			std::string segment;
			std::vector<std::string> seglist;

			while(std::getline(test, segment, '='))
			{
			   seglist.push_back(segment);
			}

			if(seglist.size()!=2){
				throw std::runtime_error("invalid line format input file.");
			}

			if(seglist[0]==double_input){
				double_values[double_input] = std::stod(seglist[1]);
				break;
			}
		}
	}



	return std::make_pair(int_values,double_values);
}

}
#endif