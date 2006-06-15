/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef HAVE_R
#include "features/Labels.h"
#include "features/Features.h"


extern "C" {

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
   
class CGUI_R
{
public:
	CGUI_R();

	// this simply sends a cmd to shogun
	// 		sg('send_command', 'cmdline');
	bool send_command(CHAR* cmd);

// sg('what to do', params ...);
//
//
// 		
// 		[p,q,a,b]=sg('get_hmm');
// 		[b,alpha]=sg('get_svm');
// 		[parms]=sg('get_kernel_init');
// 		[feature_matrix]=('get_features', 'train|test');
// 		[labels]=sg('get_labels', 'train|test');
// 		[parms]=sg('get_preproc_init');
// 		[p,q,a,b]=sg('get_hmm_defs', 'cmdline');
// 		OBSOLETE sg('get_obs', 'cmdline');
//
// 		
// 		sg('set_hmm', p,q,a,b);
// 		sg('set_svm', b,alpha);
// 		sg('set_kernel_init', parms);
// 		sg('set_features', 'train|test', feature_matrix);
// 		sg('set_labels', 'train|test', labels);
// 		sg('set_preproc_init', parms);
// 		sg('set_hmm_defs', p,q,a,b);
// 		OBSOLETE sg('set_obs', 'cmdline');

	SEXP get_hmm();
	bool set_hmm(SEXP arg_list);
	bool set_svm(SEXP arg_list);
   /*
	bool best_path(mxArray* retvals[], int dim);
	bool best_path_no_b(const mxArray* vals[], mxArray* retvals[]) ;
	bool model_prob_no_b_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_no_b_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_2struct(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_trans_simple(const mxArray* vals[], mxArray* retvals[]) ;
	SEXP hmm_classify_example(mxArray* retvals[], int idx);
	bool append_hmm(const mxArray* vals[]);
	bool one_class_hmm_classify_example(mxArray* retvals[], int idx);
	bool one_class_hmm_classify(mxArray* retvals[], bool linear);

	bool svm_classify_example(mxArray* retvals[], int idx);
	bool classify(mxArray* retvals[]);

	bool set_plugin_estimate(const mxArray* vals[]);
	bool get_plugin_estimate(mxArray* retvals[]);
	bool plugin_estimate_classify_example(mxArray* retvals[], int idx);
	bool plugin_estimate_classify(mxArray* retvals[]);

	//bool get_kernel_init();
	bool set_kernel_parameters(const mxArray* mx_arg);
   */

	SEXP hmm_classify(); 
	SEXP get_features(CFeatures* features);
	CFeatures* set_features(SEXP features, SEXP feature_length);
   
	CLabels* set_labels(SEXP labelsR);
	SEXP get_labels(CLabels* label);
	SEXP svm_classify();
   
	SEXP get_kernel_matrix();
	SEXP get_svm();
	SEXP get_svm_objective();
	//bool set_custom_kernel(SEXP args) ;
   
/*
	bool get_kernel_optimization(mxArray* retvals[]);

	// MKL Kernel stuff
	bool compute_by_subkernels(mxArray* retvals[]);
	bool get_last_subkernel_weights(mxArray* retvals[]);
	bool set_subkernel_weights(const mxArray *mx_arg);
	bool set_last_subkernel_weights(const mxArray *mx_arg);
	bool get_WD_position_weights(mxArray* retvals[]);
	bool set_WD_position_weights(const mxArray *mx_arg);

	static CHAR* get_mxString(const mxArray* s);
   */
	SEXP get_subkernel_weights();
	SEXP get_version();
};

}
#endif
