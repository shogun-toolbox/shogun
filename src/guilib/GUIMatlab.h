#ifdef HAVE_MATLAB
#ifndef __MATLAB_H_
#define __MATLAB_H_

#include "features/Labels.h"
#include "features/Features.h"

#include "mex.h"

class CGUIMatlab
{
public:
	CGUIMatlab();

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

	bool get_hmm(mxArray* retvals[]);
	bool set_hmm(const mxArray* vals[]);
	bool best_path(mxArray* retvals[], int dim);
	bool best_path_no_b(const mxArray* vals[], mxArray* retvals[]) ;
	bool model_prob_no_b_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_no_b_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool best_path_trans(const mxArray* vals[], mxArray* retvals[]) ;
	bool append_hmm(const mxArray* vals[]);
	bool hmm_classify_example(mxArray* retvals[], int idx);
	bool hmm_classify(mxArray* retvals[]);
	bool one_class_hmm_classify_example(mxArray* retvals[], int idx);
	bool one_class_hmm_classify(mxArray* retvals[], bool linear);

	bool get_svm(mxArray* retvals[]);
	bool set_svm(const mxArray* vals[]);
	bool svm_classify_example(mxArray* retvals[], int idx);
	bool svm_classify(mxArray* retvals[]);

	bool set_plugin_estimate(const mxArray* vals[]);
	bool get_plugin_estimate(mxArray* retvals[]);
	bool plugin_estimate_classify_example(mxArray* retvals[], int idx);
	bool plugin_estimate_classify(mxArray* retvals[]);

	//bool get_kernel_init();
	bool get_features(mxArray* retvals[], CFeatures* features);
	bool set_kernel_parameters(const mxArray* mx_arg);
	CFeatures* set_features(const mxArray* vals[], int nrhs);

	bool get_kernel_matrix(mxArray* retvals[], CFeatures* features);
	bool get_kernel_optimization(mxArray* retvals[]);
	bool compute_WD_by_levels(mxArray* retvals[]);

	bool get_version(mxArray* retvals[]);
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
