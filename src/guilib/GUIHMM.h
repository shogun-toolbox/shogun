#ifndef __GUIHMM__H
#define __GUIHMM__H

#include "hmm/HMM.h"

class CGUI ;

class CGUIHMM
{
public:
	CGUIHMM(CGUI *);
	~CGUIHMM();

	bool new_hmm(char* param);
	bool load_hmm(char* param);
	bool save_hmm(char* param);
	
	bool baum_welch_train(char* param);
	bool linear_train(char* param);
	bool one_class_test(char* param);
	bool test_hmm(char* param);
	bool append_model(char* param);
	bool add_states(char* param);
	bool set_hmm_as(char* param);
	bool assign_obs(char* param) ;
	bool convergence_criteria(char* param) ;

	CHMM* pos;
	CHMM* neg;
	CHMM* test;
protected:

	bool converge(double x, double y);
	void switch_model(CHMM** m1, CHMM** m2);

	CHMM* working;
	CHMM* working_estimate;


	REAL PSEUDO;
	int M;
	int ORDER;
	REAL EPSILON;

	int iteration_count;
	int ITERATIONS;
	int conv_it;

 protected:
	CGUI* gui ;
};
#endif
