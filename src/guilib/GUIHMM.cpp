#include "lib/io.h"
#include "lib/Mathmatics.h"

#include "kernel/Kernel.h"
#include "kernel/LinearKernel.h"

#include "features/Features.h"
#include "features/TOPFeatures.h"

#include "preproc/PreProc.h"
#include "preproc/NormOne.h"


#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#ifndef _WIN32
//#include <termios.h>
#endif


clock_t current_time;

// initial parameters
int    ITERATIONS = 100;
double EPSILON    = 1e-6;
double PSEUDO     = 1e-3 ;
int ORDER=1;
int M=4;
double C=1.0; //SVM C

CHMM* lambda=NULL;
CHMM* lambda_train=NULL;	//model and training model

E_OBS_ALPHABET alphabet=DNA;
CObservation* obs_postrain=NULL;//observations
CObservation* obs_negtrain=NULL;//observations
CObservation* obs_postest=NULL; //observations
CObservation* obs_negtest=NULL; //observations
CObservation* obs_test=NULL;	//observations

CHMM* test=NULL;//test model 
CHMM* pos=NULL;	//positive model 
CHMM* neg=NULL;	//negative model

// support vector machine
CSVMLight svm_light;
#ifdef SVMCPLEX
CSVMCplex svm_cplex;	
CSVM* svm=&svm_cplex;	
#else
CSVM* svm=&svm_light;	
#endif

// kernel
CLinearKernel linear_kernel(false);
CKernel* kernel=&linear_kernel;
CNormOne norm_one;
CPreProc* preproc=&norm_one;
// Features
CFeatures* trainfeatures=NULL;
CFeatures* testfeatures=NULL;

static int iteration_count=ITERATIONS ;
static int conv_it=5 ;

//convergence criteria  -tobeadjusted-
bool converge(double x, double y)
{
    double diff=y-x;
    double absdiff=fabs(diff);

    CIO::message("\n #%03d\tbest result so far: %G (eps: %f", iteration_count, y, diff);
    if (diff<0.0)
	//CIO::message(" ***") ;
	CIO::message(" **************** WARNING **************") ;
    CIO::message(")") ;

    if (iteration_count-- == 0 || (absdiff<EPSILON && conv_it<=0))
    {
	iteration_count=ITERATIONS;
	CIO::message("...finished\n");
	conv_it=5 ;
	return true;
    }
    else
    {
	if (absdiff<EPSILON)
	    conv_it-- ;
	else
	    conv_it=5;

	return false;
    }
}

CGUIHMM::CGUIHMM()
{

}
//switch model and train model
static void switch_model(CHMM** m1, CHMM** m2)
{
    CHMM* dummy= *m1;

    *m1= *m2;
    *m2= dummy;
}


