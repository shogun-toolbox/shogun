/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIHMM__H
#define __GUIHMM__H

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "base/SGObject.h"
#include "distributions/hmm/HMM.h"
#include "features/Labels.h"

class CGUI ;

class CGUIHMM : public CSGObject
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
	inline DREAL get_pseudo() { return PSEUDO; }

	CLabels* classify(CLabels* output=NULL);
	DREAL classify_example(INT idx);
	CLabels* one_class_classify(CLabels* output=NULL);
	CLabels* linear_one_class_classify(CLabels* output=NULL);
	DREAL one_class_classify_example(INT idx);
protected:

	bool converge(double x, double y);
	void switch_model(CHMM** m1, CHMM** m2);

	CHMM* working;
	
	CHMM* pos;
	CHMM* neg;
	CHMM* test;

	DREAL PSEUDO;
	INT M;

 protected:
	CGUI* gui ;
};
#endif //HAVE_SWIG
#endif
