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

#ifdef HAVE_MATLAB
#ifndef __MATLAB_H_
#define __MATLAB_H_

#include "features/Labels.h"
#include "features/Features.h"

#include <mex.h>

class CGUIMatlab
{
public:
	CGUIMatlab();

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

	bool relative_entropy(mxArray* retvals[]);
	bool entropy(mxArray* retvals[]);
	bool get_hmm(mxArray* retvals[]);
	bool set_hmm(const mxArray* vals[]);
	bool hmm_likelihood(mxArray* retvals[]);
	bool best_path(mxArray* retvals[], int dim);
	bool best_path_no_b(const mxArray* vals[], mxArray* retvals[]) ;
	bool model_prob_no_b_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_no_b_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_2struct(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_trans_simple(const mxArray* vals[], mxArray* retvals[]) ;
	bool append_hmm(const mxArray* vals[]);
	bool hmm_classify_example(mxArray* retvals[], int idx);
	bool hmm_classify(mxArray* retvals[]);
	bool one_class_hmm_classify_example(mxArray* retvals[], int idx);
	bool one_class_hmm_classify(mxArray* retvals[], bool linear);

	bool get_svm(mxArray* retvals[]);
	bool set_svm(const mxArray* vals[]);
	bool svm_classify_example(mxArray* retvals[], int idx);
	bool svm_classify(mxArray* retvals[]);
	bool classify(mxArray* retvals[]);

	bool set_plugin_estimate(const mxArray* vals[]);
	bool get_plugin_estimate(mxArray* retvals[]);
	bool plugin_estimate_classify_example(mxArray* retvals[], int idx);
	bool plugin_estimate_classify(mxArray* retvals[]);

	//bool get_kernel_init();
	bool get_features(mxArray* retvals[], CFeatures* features);
	bool set_kernel_parameters(const mxArray* mx_arg);
	CFeatures* set_features(const mxArray* vals[], int nrhs);

	bool get_kernel_matrix(mxArray* retvals[]);
	bool get_kernel_optimization(mxArray* retvals[]);

	bool set_custom_kernel(const mxArray* vals[], bool source_is_diag, bool dest_is_diag) ;

	// MKL Kernel stuff
	bool compute_by_subkernels(mxArray* retvals[]);
	bool get_subkernel_weights(mxArray* retvals[]);
	bool get_last_subkernel_weights(mxArray* retvals[]);
	bool set_subkernel_weights(const mxArray *mx_arg);
	bool set_last_subkernel_weights(const mxArray *mx_arg);
	bool get_WD_position_weights(mxArray* retvals[]);
	bool get_WD_scoring(mxArray* retvals[]);
	bool set_WD_position_weights(const mxArray *mx_arg);

	bool get_version(mxArray* retvals[]);
	bool get_svm_objective(mxArray* retvals[]);
	bool get_labels(mxArray* retvals[], CLabels* label);
	CLabels* set_labels(const mxArray* vals[]);
	//bool get_preproc_init();
	//bool get_hmm_defs();

	//bool set_kernel_init();
	//bool set_preproc_init();
	//bool set_hmm_defs();
	
	void real_mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]);
	static CHAR* get_mxString(const mxArray* s);
};
#endif
#endif
