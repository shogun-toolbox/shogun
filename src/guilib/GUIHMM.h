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
	bool load(char* param);
	bool save(char* param);
	
	bool baum_welch_train(char* param);
	bool baum_welch_train_defined(char* param);
	bool viterbi_train_defined(char* param);
	bool viterbi_train(char* param);
	bool linear_train(char* param);
	bool linear_train_from_file(char* param);
	bool one_class_test(char* param);
	bool test_hmm(char* param);
	bool append_model(char* param);
	bool add_states(char* param);
	bool set_hmm_as(char* param);
	bool set_pseudo(char* param) ;
	bool assign_obs(char* param) ;
	bool convergence_criteria(char* param) ;
	bool output_hmm_path(char* param);
	bool output_hmm(char* param);
	bool output_hmm_defined(char* param);
	bool best_path(char* param);
	bool normalize(char* param);
	bool save_path(char* param);
	bool load_defs(char* param);
	bool set_max_dim(char* param);
	bool likelihood(char* param);
	bool chop(char* param);
	inline CHMM* get_pos() { return pos; }
	inline CHMM* get_neg() { return neg; }
	inline CHMM* get_test() { return test; }

protected:

	bool converge(double x, double y);
	void switch_model(CHMM** m1, CHMM** m2);

	CHMM* working;
	
	CHMM* pos;
	CHMM* neg;
	CHMM* test;

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
