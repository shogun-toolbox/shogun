#include "lib/config.h"

#ifdef HAVE_OCTAVE
#ifndef __OCTAVE_H_
#define __OCTAVE_H_

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

	// this simply sends a cmd to genefinder
	// 		gf('send_command', 'cmdline');
	bool send_command(CHAR* cmd);

// gf('what to do', params ...);
//
//
// 		
// 		[p,q,a,b]=gf('get_hmm');
// 		[b,alpha]=gf('get_svm');
// 		[parms]=gf('get_kernel_init');
// 		[feature_matrix]=('get_features', 'train|test');
// 		[labels]=gf('get_labels', 'train|test');
// 		[parms]=gf('get_preproc_init');
// 		[p,q,a,b]=gf('get_hmm_defs', 'cmdline');
// 		OBSOLETE gf('get_obs', 'cmdline');
//
// 		
// 		gf('set_hmm', p,q,a,b);
// 		gf('set_svm', b,alpha);
// 		gf('set_kernel_init', parms);
// 		gf('set_features', 'train|test', feature_matrix);
// 		gf('set_labels', 'train|test', labels);
// 		gf('set_preproc_init', parms);
// 		gf('set_hmm_defs', p,q,a,b);
// 		OBSOLETE gf('set_obs', 'cmdline');

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
	//bool get_preproc_init();
	//bool get_hmm_defs();

	//bool set_kernel_init();
	//bool set_preproc_init();
	//bool set_hmm_defs();
	
	static CHAR* get_octaveString(std::string);
};
#endif
#endif
