/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Fabio De Bona
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUI_R_H
#define __GUI_R_H

#include "lib/config.h"

#if defined(HAVE_R) && !defined(HAVE_SWIG)
#include "base/SGObject.h"

#include "features/Labels.h"
#include "features/Features.h"

extern "C" {

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
   
class CGUI_R : public CSGObject
{
public:
	CGUI_R();

	// this simply sends a cmd to shogun
	// 		sg('send_command', 'cmdline');
	bool send_command(CHAR* cmd);

// sg('what to do', params ...);
// 		
// 		[p,q,a,b]=sg('get_hmm');
// 		[b,alpha]=sg('get_svm');
// 		[parms]=sg('get_kernel_init');
// 		[feature_matrix]=('get_features', 'train|test');
// 		[labels]=sg('get_labels', 'train|test');
// 		[parms]=sg('get_preproc_init');
// 		[p,q,a,b]=sg('get_hmm_defs', 'cmdline');
//
// 		
// 		sg('set_hmm', p,q,a,b);
// 		sg('set_svm', b,alpha);
// 		sg('set_kernel_init', parms);
// 		sg('set_features', 'train|test', feature_matrix);
// 		sg('set_labels', 'train|test', labels);
// 		sg('set_preproc_init', parms);
// 		sg('set_hmm_defs', p,q,a,b);

	SEXP get_hmm();
	bool set_hmm(SEXP arg_list);
	bool set_svm(SEXP arg_list);
	SEXP best_path(int dim);
	SEXP hmm_classify_example(int idx);
	bool append_hmm(const SEXP arg_list);
	SEXP one_class_hmm_classify_example(int idx);
	SEXP one_class_hmm_classify();

	SEXP hmm_likelihood(); 
	SEXP hmm_classify(); 
	SEXP get_features(CFeatures* features);
	CFeatures* set_features(SEXP feat, SEXP alphabet);
   
	CLabels* set_labels(SEXP labelsR);
	SEXP get_labels(CLabels* label);
	SEXP svm_classify();
	SEXP svm_classify_example(INT idx);
   
	SEXP get_kernel_matrix();
	SEXP get_svm();
	SEXP get_svm_objective();
	bool set_custom_kernel(SEXP args) ;
   
	SEXP get_subkernel_weights();
	SEXP get_version();
};

}
#endif //HAVE_SWIG
#endif
