#ifdef MATLAB
#include <stdio.h>
#include <string.h>

#include "mex.h"

static int  mex_count = 0;

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    
    char array_name[40];
    mxArray *array_ptr;
    int status;
    
	if (!gui)
		gui=new CTextGUI(0, NULL);

    /* Check for proper number of input and output arguments */    
    if (nrhs !=0) {
        mexErrMsgTxt("No input arguments required.");
    } 
    if(nlhs > 1){
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Make copy of MEX-file name, then create variable for MATLAB
       workspace from MEX-file name. */
    strcpy(array_name, mexFunctionName());
    strcat(array_name,"_called");
    
    /* Get variable that keeps count of how many times MEX-file has
       been called from MATLAB global workspace. */
    array_ptr = mexGetVariable("global", array_name);
    
    /* Check status of MATLAB and MEX-file MEX-file counter */    
    
    if (array_ptr == NULL ){
	if( mex_count != 0){
	    mex_count = 0;
	    mexPrintf("Variable %s\n", array_name);
	    mexErrMsgTxt("Global variable was cleared from the MATLAB \
global workspace.\nResetting count.\n");
	}
    	
	/* Since variable does not yet exist in MATLAB workspace,
           create it and place it in the global workspace. */
	array_ptr=mxCreateDoubleMatrix(1,1,mxREAL);
    }
    
    /* Increment both MATLAB and MEX counters by 1 */
    mxGetPr(array_ptr)[0]+=1;
    mex_count=mxGetPr(array_ptr)[0];
    mexPrintf("%s has been called %i time(s)\n", mexFunctionName(), mex_count);
    
    /* Put variable in MATLAB global workspace */
    status=mexPutVariable("global", array_name, array_ptr);
    
    if (status==1){
	mexPrintf("Variable %s\n", array_name);
	mexErrMsgTxt("Could not put variable in global workspace.\n");
    }
    
    /* Destroy array */
    mxDestroyArray(array_ptr);
}
#endif
