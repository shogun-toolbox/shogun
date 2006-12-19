/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef __OCTAVE_H_
#define __OCTAVE_H_

#if defined(HAVE_OCTAVE) && !defined(HAVE_SWIG)
#include "features/Labels.h"
#include "features/Features.h"

#include <octave/config.h>

#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>
#include <octave/ov-cell.h>

class CGUIOctave
{
public:
	CGUIOctave();

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

	bool get_hmm(octave_value_list& plhs);
	bool set_hmm(const octave_value_list& vals);
	bool best_path_no_b(const octave_value_list& vals, octave_value_list& retvals) ;
	bool model_prob_no_b_trans(const octave_value_list& vals, octave_value_list& retvals) ;
	bool best_path_no_b_trans(const octave_value_list& vals, octave_value_list& retvals) ;
	bool best_path_trans(const octave_value_list& vals, octave_value_list& retvals) ;
	bool append_hmm(const octave_value_list& vals);
	bool hmm_classify_example(octave_value_list& retvals, int idx);
	bool hmm_classify(octave_value_list& retvals);
	bool one_class_hmm_classify_example(octave_value_list& retvals, int idx);
	bool one_class_hmm_classify(octave_value_list& retvals, bool linear);

	bool get_svm(octave_value_list& retvals);
	bool set_svm(const octave_value_list& vals);
	bool svm_classify_example(octave_value_list& retvals, int idx);
	bool svm_classify(octave_value_list& retvals);

	bool set_plugin_estimate(const octave_value_list& vals);
	bool get_plugin_estimate(octave_value_list& retvals);
	bool plugin_estimate_classify_example(octave_value_list& retvals, int idx);
	bool plugin_estimate_classify(octave_value_list& retvals);

	//bool get_kernel_init();
	bool get_features(octave_value_list& retvals, CFeatures* features);
	CFeatures* set_features(const octave_value_list& vals);

	bool get_kernel_matrix(octave_value_list& retvals);
	bool get_kernel_optimization(octave_value_list& retvals);

	bool get_labels(octave_value_list& retvals, CLabels* label);
	CLabels* set_labels(const octave_value_list& vals);

	bool best_path(octave_value_list& retvals, int dim);
	bool hmm_likelihood(octave_value_list& retvals);
	//bool get_preproc_init();
	//bool get_hmm_defs();

	//bool set_kernel_init();
	//bool set_preproc_init();
	//bool set_hmm_defs();
	
	static CHAR* get_octaveString(std::string);
};
#endif //HAVE_SWIG
#endif
