#ifndef __MATHLAB_H_
#define __MATHLAB_H_

#include "mex.h"

class CMatlab
{
public:
	CMatlab();

	// this simply sends a cmd to genefinder
	// 		gf('send_command', 'cmdline');
	bool send_command(char* cmd);

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

	//bool get_svm();
	//bool get_kernel_init();
	//bool get_features();
	//bool get_labels();
	//bool get_preproc_init();
	//bool get_hmm_defs();

	//bool set_hmm();
	//bool set_svm();
	//bool set_kernel_init();
	//bool set_features();
	//bool set_labels();
	//bool set_preproc_init();
	//bool set_hmm_defs();
	
	void real_mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]);
protected:
	char* get_mxString(const mxArray* s);
};
#endif
