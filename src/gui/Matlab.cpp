#ifdef MATLAB
#include <stdio.h>
#include <string.h>

#include "lib/io.h"
#include "mex.h"
#include "hmm/HMM.h"
#include "hmm/HMM.h"

#include "gui/Matlab.h"
#include "gui/TextGUI.h"
#include "gui/GUI.h"

static CMatlab gf_matlab;
extern CTextGUI* gui;

static const char* N_SEND_COMMAND=		"send_command";
static const char* N_GET_HMM=			"get_hmm";
static const char* N_GET_SVM=			"get_svm";
static const char* N_GET_KERNEL_INIT=	"get_kernel_init";
static const char* N_GET_FEATURES=		"get_features";
static const char* N_GET_LABELS=		"get_labels";
static const char* N_GET_PREPROC_INIT=	"get_preproc_init";
static const char* N_GET_HMM_DEFS=		"get_hmm_defs";
static const char* N_SET_HMM=			"set_hmm";
static const char* N_SET_SVM=			"set_svm";
static const char* N_SET_KERNEL_INIT=	"set_kernel_init";
static const char* N_SET_FEATURES=		"set_features";
static const char* N_SET_LABELS=		"set_labels";
static const char* N_SET_PREPROC_INIT=	"set_preproc_init";
static const char* N_SET_HMM_DEFS=		"set_hmm_defs";

CMatlab::CMatlab()
{
}

bool CMatlab::send_command(char* cmd)
{
	return (gui->parse_line(cmd));
}

bool CMatlab::get_hmm(mxArray* retvals[])
{
	CHMM* h=gui->guihmm.get_current();

	if (h)
	{
		mxArray* mx_p=mxCreateDoubleMatrix(1, h->get_N(), mxREAL);
		mxArray* mx_q=mxCreateDoubleMatrix(1, h->get_N(), mxREAL);
		mxArray* mx_a=mxCreateDoubleMatrix(h->get_N(), h->get_N(), mxREAL);
		mxArray* mx_b=mxCreateDoubleMatrix(h->get_N(), h->get_M(), mxREAL);

		if (mx_p && mx_q && mx_a && mx_b)
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a);
			double* b=mxGetPr(mx_b);

			int i,j;
			for (i=0; i< h->get_N(); i++)
			{
				p[i]=h->get_p(i);
				q[i]=h->get_q(i);
			}

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_N(); j++)
					a[i+j*h->get_N()]=h->get_a(i,j);

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_M(); j++)
					b[i+j*h->get_N()]=h->get_b(i,j);

			retvals[0]=mx_p;
			retvals[1]=mx_q;
			retvals[2]=mx_a;
			retvals[3]=mx_b;

			return true;
		}
	}

	return false;
}

bool CMatlab::set_hmm(const mxArray* vals[])
{
	CHMM* h=gui->guihmm.get_current();

	if (h)
	{
		const mxArray* mx_p=vals[1];
		const mxArray* mx_q=vals[2];
		const mxArray* mx_a=vals[3];
		const mxArray* mx_b=vals[4];

		if (
				mx_p && mx_q && mx_a && mx_b &&
				mxGetN(mx_p) == h->get_N() && mxGetM(mx_p) == 1 &&
				mxGetN(mx_q) == h->get_N() && mxGetM(mx_q) == 1 &&
				mxGetN(mx_a) == h->get_N() && mxGetM(mx_a) == h->get_N() &&
				mxGetN(mx_b) == h->get_M() && mxGetM(mx_b) == h->get_N()
		   )
		{
			double* p=mxGetPr(mx_p);
			double* q=mxGetPr(mx_q);
			double* a=mxGetPr(mx_a);
			double* b=mxGetPr(mx_b);

			int i,j;
			for (i=0; i< h->get_N(); i++)
			{
				h->set_p(i, p[i]);
				h->set_q(i, q[i]);
			}

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_N(); j++)
					h->set_a(i,j, a[i+j*h->get_N()]);

			for (i=0; i< h->get_N(); i++)
				for (j=0; j< h->get_M(); j++)
					h->set_b(i,j, b[i+j*h->get_N()]);

			return true;
		}
	}

	return false;
}

bool CMatlab::get_svm(mxArray* retvals[])
{
	CSVM* svm=gui->guisvm.get_svm();

	if (svm)
	{
		mxArray* mx_alphas=mxCreateDoubleMatrix(svm->get_num_support_vectors(), 2, mxREAL);
		mxArray* mx_b=mxCreateDoubleMatrix(1, 1, mxREAL);

		if (mx_alphas && mx_b)
		{
			double* b=mxGetPr(mx_b);
			double* alphas=mxGetPr(mx_alphas);

			p[0]=svm->get_bias();

			for (int i=0; i< svm->get_num_support_vectors(); i++)
			{
				alphas[i]=svm->get_alpha(i);
				alphas[i+svm->get_num_support_vectors()]=svm->get_support_vector(i);
			}

			retvals[0]=mx_b;
			retvals[1]=mx_alphas;

			return true;
		}
	}

	return false;
}

bool CMatlab::set_hmm(const mxArray* vals[])
{
	SVM* svm=gui->guisvm.get_svm();

	if (svm)
	{
		const mxArray* mx_b=vals[1];
		const mxArray* mx_alphas=vals[2];

		if (
				mx_b && mx_alphas &&
				mxGetN(mx_b) == 1 && mxGetM(mx_b) == 1 &&
				mxGetN(mx_alphas) == 2
			)
		{
			double* b=mxGetPr(mx_b);
			double* alphas=mxGetPr(mx_alphas);

			svm->create_new_model(mxGetM(mx_alphas));
			svm->set_bias(*b);

			for (int i=0; i< svm->get_num_support_vectors(); i++)
			{
				svm->set_alpha(i, alphas[i]);
				svm->set_support_vector(i, alphas[i+svm->get_num_support_vectors()]);
			}

			return true;
		}
	}

	return false;
}

char* CMatlab::get_mxString(const mxArray* s)
{
	if ( (mxIsChar(s)) && (mxGetM(s)==1) )
	{
		int buflen = (mxGetN(s) * sizeof(mxChar)) + 1;
		char* string=new char[buflen];
		mxGetString(s, string, buflen);
		return string;
	}
	else
		return NULL;
}

void CMatlab::real_mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
	if (!nrhs)
	{
		//add some more text
		mexErrMsgTxt("No input arguments supplied.");
	} 

	char* action=get_mxString(prhs[0]);

	if (action)
	{

		if (!strncmp(action, N_SEND_COMMAND, strlen(N_SEND_COMMAND)))
		{
			if (nrhs==2)
			{
				char* cmd=get_mxString(prhs[1]);
				gf_matlab.send_command(cmd);
				delete[] cmd;
			}
			else
				mexErrMsgTxt("usage is gf('send_command', 'cmdline')");
		}
		else if (!strncmp(action, N_GET_HMM, strlen(N_GET_HMM)))
		{
			if (nlhs==4)
			{
				gf_matlab.get_hmm(plhs);
			}
			else
				mexErrMsgTxt("usage is [p,q,a,b]=gf('get_hmm')");
		}
		else if (!strncmp(action, N_GET_SVM, strlen(N_GET_SVM)))
		{
			if (nlhs==2)
			{
				gf_matlab.get_svm(plhs);
			}
			else
				mexErrMsgTxt("usage is [b,alphas]=gf('get_svm')");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
			if (nrhs==3)
			{
				gf_matlab.set_svm(prhs);
			}
			else
				mexErrMsgTxt("usage is gf('set_svm', [ b, alphas])");
		}
		else if (!strncmp(action, N_GET_KERNEL_INIT, strlen(N_GET_KERNEL_INIT)))
		{
		}
		else if (!strncmp(action, N_GET_FEATURES, strlen(N_GET_FEATURES)))
		{
		}
		else if (!strncmp(action, N_GET_LABELS, strlen(N_GET_LABELS)))
		{
		}
		else if (!strncmp(action, N_GET_PREPROC_INIT, strlen(N_GET_PREPROC_INIT)))
		{
		}
		else if (!strncmp(action, N_GET_HMM_DEFS, strlen(N_GET_HMM_DEFS)))
		{
		}
		else if (!strncmp(action, N_SET_HMM, strlen(N_SET_HMM)))
		{
			if (nrhs==1+4)
			{
				gf_matlab.set_hmm(prhs);
			}
			else
				mexErrMsgTxt("usage is =gf([p,q,a,b])");
		}
		else if (!strncmp(action, N_SET_SVM, strlen(N_SET_SVM)))
		{
		}
		else if (!strncmp(action, N_SET_KERNEL_INIT, strlen(N_SET_KERNEL_INIT)))
		{
		}
		else if (!strncmp(action, N_SET_FEATURES, strlen(N_SET_FEATURES)))
		{
		}
		else if (!strncmp(action, N_SET_LABELS, strlen(N_SET_LABELS)))
		{
		}
		else if (!strncmp(action, N_SET_PREPROC_INIT, strlen(N_SET_PREPROC_INIT)))
		{
		}
		else if (!strncmp(action, N_SET_HMM_DEFS, strlen(N_SET_HMM_DEFS)))
		{
		}
		else
			mexErrMsgTxt("action not defined");

		delete[] action;
	}
	else
		mexErrMsgTxt("string expected as first argument");
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	if (!gui)
		gui=new CTextGUI(0, NULL);
	
	assert(gui);

	gf_matlab.real_mexFunction(nlhs,plhs, nrhs,prhs);
}

#endif
//
//static int  mex_count = 0;
//		/* Make copy of MEX-file name, then create variable for MATLAB
//		   workspace from MEX-file name. */
//		strcpy(array_name, mexFunctionName());
//		strcat(array_name,"_called");
//
//		/* Get variable that keeps count of how many times MEX-file has
//		   been called from MATLAB global workspace. */
//		array_ptr = mexGetVariable("global", array_name);
//
//		/* Check status of MATLAB and MEX-file MEX-file counter */    
//
//		if (array_ptr == NULL ){
//			if( mex_count != 0){
//				mex_count = 0;
//				mexPrintf("Variable %s\n", array_name);
//				mexErrMsgTxt("Global variable was cleared from the MATLAB global workspace.\nResetting count.\n");
//			}
//
//			/* Since variable does not yet exist in MATLAB workspace,
//			   create it and place it in the global workspace. */
//			array_ptr=mxCreateDoubleMatrix(1,1,mxREAL);
//		}
//
//		/* Increment both MATLAB and MEX counters by 1 */
//		mxGetPr(array_ptr)[0]+=1;
//		mex_count=(int) mxGetPr(array_ptr)[0];
//		mexPrintf("%s has been called %i time(s)\n", mexFunctionName(), mex_count);
//
//		/* Put variable in MATLAB global workspace */
//		status=mexPutVariable("global", array_name, array_ptr);
//
//		if (status==1){
//			mexPrintf("Variable %s\n", array_name);
//			mexErrMsgTxt("Could not put variable in global workspace.\n");
//		}
//
//		/* Destroy array */
//		mxDestroyArray(array_ptr);
//		mexErrMsgTxt("string expected as first argument");
	//char array_name[40];
	//mxArray *array_ptr;
	//int status;

