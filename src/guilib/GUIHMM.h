#ifndef __GUIHMM__H
#define __GUIHMM__H

#include "distributions/hmm/HMM.h"
#include "features/Labels.h"

class CGUI ;

class CGUIHMM 
{
public:
	CGUIHMM(CGUI *);
	~CGUIHMM();

	bool new_hmm(CHAR* param);
	bool load(CHAR* param);
	bool save(CHAR* param);
	
	bool set_num_hmm_tables(CHAR* param) ;
	bool baum_welch_train(CHAR* param);
	bool baum_welch_trans_train(CHAR* param);
	bool baum_welch_train_defined(CHAR* param);
	bool viterbi_train_defined(CHAR* param);
	bool viterbi_train(CHAR* param);
	bool linear_train(CHAR* param);
	bool linear_train_from_file(CHAR* param);
	bool one_class_test(CHAR* param);
	bool hmm_test(CHAR* param);
	bool hmm_classify(CHAR* param);
	bool append_model(CHAR* param);
	bool add_states(CHAR* param);
	bool set_hmm_as(CHAR* param);
	bool set_pseudo(CHAR* param) ;
	bool convergence_criteria(CHAR* param) ;
	bool output_hmm_path(CHAR* param);
	bool output_hmm(CHAR* param);
	bool output_hmm_defined(CHAR* param);
	bool best_path(CHAR* param);
	bool normalize(CHAR* param);
	bool save_path(CHAR* param);
	bool save_likelihood(CHAR* param);
	bool load_defs(CHAR* param);
	bool set_max_dim(CHAR* param);
	bool likelihood(CHAR* param);
	bool chop(CHAR* param);
	bool relative_entropy(CHAR* param);
	bool entropy(CHAR* param);
	bool permutation_entropy(CHAR* param);
	inline CHMM* get_pos() { return pos; }
	inline CHMM* get_neg() { return neg; }
	inline CHMM* get_test() { return test; }
	inline void set_current(CHMM* h) { working=h; }
	inline CHMM* get_current() { return working; }
	inline REAL get_pseudo() { return PSEUDO; }
	inline INT get_number_of_tables() { return number_of_hmm_tables; }

	CLabels* classify(CLabels* output=NULL);
	REAL classify_example(INT idx);
	CLabels* one_class_classify(CLabels* output=NULL);
	CLabels* linear_one_class_classify(CLabels* output=NULL);
	REAL one_class_classify_example(INT idx);
protected:

	INT number_of_hmm_tables ;
	bool converge(double x, double y);
	void switch_model(CHMM** m1, CHMM** m2);

	CHMM* working;
	
	CHMM* pos;
	CHMM* neg;
	CHMM* test;

	REAL PSEUDO;
	INT M;
	REAL EPSILON;

	INT iteration_count;
	INT ITERATIONS;
	INT conv_it;

 protected:
	CGUI* gui ;
};
#endif
