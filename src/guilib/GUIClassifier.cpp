#include "guilib/GUIClassifier.h"
#include "gui/GUI.h"
#include "lib/io.h"
#include "features/RealFileFeatures.h"
#include "features/Labels.h"

#include "classifier/KNN.h"
#include "classifier/PluginEstimate.h"

#include "classifier/Perceptron.h"
#include "classifier/KernelPerceptron.h"

#include "classifier/svm/SVM_light.h"
#include "classifier/svm/LibSVM.h"
#include "classifier/svm/GPBTSVM.h"
#include "classifier/svm/MPD.h"

#include "regression/svr/SVR_light.h"
#include "regression/svr/LibSVR.h"


#include <assert.h>

CGUIClassifier::CGUIClassifier(CGUI* g) : gui(g)
{
	classifier=NULL;

    // SVM stuff
	svm_qpsize=41;
	svm_C1=1;
	svm_C2=1;
	svm_C_mkl=0;
	svm_weight_epsilon=1e-5;
	svm_epsilon=1e-5;
	svm_tube_epsilon=1e-2;

	svm_use_mkl = false ;
	svm_use_linadd = false ;
	svm_use_precompute = false ;
	svm_use_precompute_subkernel = false ;
	svm_use_precompute_subkernel_light = false ;
}

CGUIClassifier::~CGUIClassifier()
{
	delete classifier;
}

bool CGUIClassifier::new_classifier(CHAR* param)
{
	param=CIO::skip_spaces(param);

	if (strcmp(param,"SVMLIGHT")==0)
	{
		delete classifier;
		classifier= new CSVMLight();
		CIO::message(M_INFO, "created SVMLight object\n") ;
	}
	else if (strcmp(param,"LIBSVM")==0)
	{
		delete classifier;
		classifier= new CLibSVM();
		CIO::message(M_INFO, "created SVMlibsvm object\n") ;
	}
	else if (strcmp(param,"GPBTSVM")==0)
	{
		delete classifier;
		classifier= new CGPBTSVM();
		CIO::message(M_INFO, "created GPBT-SVM object\n") ;
	}
	else if (strcmp(param,"MPDSVM")==0)
	{
		delete classifier;
		classifier= new CMPDSVM();
		CIO::message(M_INFO, "created MPD-SVM object\n") ;
	}
	else if (strcmp(param,"LIBSVR")==0)
	{
		delete classifier;
		classifier= new CLibSVR();
		CIO::message(M_INFO, "created SVRlibsvm object\n") ;
	}
	else if (strcmp(param,"SVRLIGHT")==0)
	{
		delete classifier;
		classifier= new CSVRLight();
		CIO::message(M_INFO, "created SVRLight object\n") ;
	}
	else if (strcmp(param,"PERCEPTRON")==0)
	{
		delete classifier;
		classifier= new CPerceptron();
		CIO::message(M_INFO, "created Perceptron object\n") ;
	}
	else
		return false;

	return (classifier!=NULL);
}

bool CGUIClassifier::train(CHAR* param, bool auc_maximization)
{
	param=CIO::skip_spaces(param);

	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CKernel* kernel=gui->guikernel.get_kernel();

	if (!trainfeatures)
	{
		CIO::message(M_ERROR, "no trainfeatures available\n") ;
		return false ;
	}

	if (!classifier)
	{
		CIO::message(M_ERROR, "no classifier available\n") ;
		return false ;
	}

	if (!kernel)
	{
		CIO::message(M_ERROR, "no kernel available\n");
		return false ;
	}

	if (!trainlabels)
	{
		CIO::message(M_ERROR, "no trainlabels available\n");
		return false ;
	}

	if ( !gui->guikernel.is_initialized() || !kernel->get_lhs() )
	{
		CIO::message(M_ERROR, "kernel not initialized\n") ;
		return 0;
	}

	if (trainlabels->get_num_labels() != kernel->get_lhs()->get_num_vectors())
	{
		CIO::message(M_ERROR, "number of train labels (%d) and training vectors (%d) differs!\n", 
				trainlabels->get_num_labels(), kernel->get_lhs()->get_num_vectors()) ;
		return 0;
	}

	CIO::message(M_INFO, "starting svm training on %ld vectors using C1=%lf C2=%lf\n", trainlabels->get_num_labels(), svm_C1, svm_C2) ;

	CSVM* svm= (CSVM*) classifier;

	svm->set_weight_epsilon(svm_weight_epsilon);
	svm->set_epsilon(svm_epsilon);
	svm->set_tube_epsilon(svm_tube_epsilon);
	svm->set_C_mkl(svm_C_mkl);
	svm->set_C(svm_C1, svm_C2);
	svm->set_qpsize(svm_qpsize);
	svm->set_mkl_enabled(svm_use_mkl);
	svm->set_linadd_enabled(svm_use_linadd);
	((CKernelMachine*) svm)->set_labels(trainlabels);
	((CKernelMachine*) svm)->set_kernel(kernel);
	((CSVM*) svm)->set_precomputed_subkernels_enabled(svm_use_precompute_subkernel_light) ;
	kernel->set_precompute_matrix(svm_use_precompute, svm_use_precompute_subkernel);
	
	if (auc_maximization)
		((CSVMLight*)svm)->setup_auc_maximization() ;

	bool result = svm->train();

	kernel->set_precompute_matrix(false,false);
	return result ;	
}

bool CGUIClassifier::test(CHAR* param)
{
	CHAR outputname[1024];
	CHAR rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	INT numargs=-1;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%s %s", outputname, rocfname);

	if (numargs>=1)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			CIO::message(M_ERROR, "could not open %s\n",outputname);
			return false;
		}

		if (numargs==2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				CIO::message(M_ERROR, "could not open %s\n",rocfname);
				return false;
			}
		}
	}

	CLabels* testlabels=gui->guilabels.get_test_labels();
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	CIO::message(M_DEBUG, "I:training: %ld examples each %ld features\n", ((CRealFeatures*) trainfeatures)->get_num_vectors(), ((CRealFeatures*) trainfeatures)->get_num_features());
	CIO::message(M_DEBUG, "I:testing: %ld examples each %ld features\n", ((CRealFeatures*) testfeatures)->get_num_vectors(), ((CRealFeatures*) testfeatures)->get_num_features());

	if (!classifier)
	{
		CIO::message(M_ERROR, "no svm available") ;
		return false ;
	}
	if (!trainfeatures)
	{
		CIO::message(M_ERROR, "no training features available") ;
		return false ;
	}

	if (!testfeatures)
	{
		CIO::message(M_ERROR, "no test features available") ;
		return false ;
	}

	if (!testlabels)
	{
		CIO::message(M_ERROR, "no test labels available") ;
		return false ;
	}

	if (!gui->guikernel.is_initialized())
	{
		CIO::message(M_ERROR, "kernel not initialized\n") ;
		return 0;
	}

	CIO::message(M_INFO, "starting svm testing\n") ;
	((CKernelMachine*) classifier)->set_labels(testlabels);
	((CKernelMachine*) classifier)->set_kernel(gui->guikernel.get_kernel()) ;
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);

	if ( (gui->guikernel.get_kernel()->is_optimizable()) && (gui->guikernel.get_kernel()->get_is_initialized()))
		CIO::message(M_DEBUG, "using kernel optimization\n");

	INT len=0;
	CLabels* predictions= classifier->classify();
	REAL* output= predictions->get_labels(len);
	INT total=	testfeatures->get_num_vectors();
	INT* label= testlabels->get_int_labels(len);

	assert(label);
	CIO::message(M_DEBUG, "len:%d total:%d\n", len, total);
	assert(len==total);

	gui->guimath.evaluate_results(output, label, total, outputfile, rocfile);

	if (rocfile)
		fclose(rocfile);
	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	delete[] output;
	delete[] label;
	return true;
}

bool CGUIClassifier::load(CHAR* param)
{
	return false;
}

bool CGUIClassifier::save(CHAR* param)
{
	return false;
}

bool CGUIClassifier::set_svm_epsilon(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &svm_epsilon) ;

	if (svm_epsilon<0)
		svm_epsilon=1e-4;

	CIO::message(M_INFO, "Set to svm_epsilon=%f\n", svm_epsilon);
	return true ;  
}

bool CGUIClassifier::set_svr_tube_epsilon(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &svm_tube_epsilon) ;

	if (svm_tube_epsilon<0)
		svm_tube_epsilon=1e-2;

	CIO::message(M_INFO, "Set to svr_tube_epsilon=%f\n", svm_tube_epsilon);
	return true ;  
}

bool CGUIClassifier::set_svm_mkl_parameters(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le %le", &svm_weight_epsilon, &svm_C_mkl) ;

	if (svm_weight_epsilon<0)
		svm_weight_epsilon=1e-4;
	if (svm_C_mkl<0)
		svm_C_mkl=1e-4 ;

	CIO::message(M_INFO, "Set to weight_epsilon=%f\n", svm_weight_epsilon);
	CIO::message(M_INFO, "Set to C_mkl=%f\n", svm_C_mkl);
	return true ;  
}

bool CGUIClassifier::set_svm_C(CHAR* param)
{
	param=CIO::skip_spaces(param);

	svm_C1=-1;
	svm_C2=-1;

	sscanf(param, "%le %le", &svm_C1, &svm_C2) ;

	if (svm_C1<0)
		svm_C1=1.0;
	if (svm_C2<0)
		svm_C2=svm_C1;

	CIO::message(M_INFO, "Set to C1=%f C2=%f\n", svm_C1, svm_C2) ;
	return true ;  
}

bool CGUIClassifier::set_svm_qpsize(CHAR* param)
{
	param=CIO::skip_spaces(param);

	svm_qpsize=-1;

	sscanf(param, "%d", &svm_qpsize) ;

	if (svm_qpsize<2)
		svm_qpsize=41;

	CIO::message(M_INFO, "Set qpsize to svm_qpsize=%d\n", svm_qpsize);
	return true ;  
}

bool CGUIClassifier::set_svm_mkl_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int mkl=1;
	sscanf(param, "%d", &mkl) ;

	svm_use_mkl = (mkl==1);

	if (svm_use_mkl)
		CIO::message(M_INFO, "Enabling MKL optimization\n") ;
	else
		CIO::message(M_INFO, "Disabling MKL optimization\n") ;

	return true ;  
}

bool CGUIClassifier::set_svm_precompute_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int precompute=1;
	sscanf(param, "%d", &precompute) ;

	svm_use_precompute = (precompute==1);
	svm_use_precompute_subkernel = (precompute==2);
	svm_use_precompute_subkernel_light = (precompute==3);

	if (svm_use_precompute)
		CIO::message(M_INFO, "Enabling Kernel Matrix Precomputation\n") ;
	else
		CIO::message(M_INFO, "Disabling Kernel Matrix Precomputation\n") ;

	if (svm_use_precompute_subkernel)
		CIO::message(M_INFO, "Enabling Subkernel Matrix Precomputation\n") ;
	else
		CIO::message(M_INFO, "Disabling Subkernel Matrix Precomputation\n") ;

	if (svm_use_precompute_subkernel_light)
		CIO::message(M_INFO, "Enabling Subkernel Matrix Precomputation by SVM Light\n") ;
	else
		CIO::message(M_INFO, "Disabling Subkernel Matrix Precomputation by SVM Light\n") ;

	return true ;  
}

bool CGUIClassifier::set_svm_linadd_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int linadd=1;
	sscanf(param, "%d", &linadd) ;

	svm_use_linadd = (linadd==1);
	
	if (svm_use_linadd)
		CIO::message(M_INFO, "Enabling LINADD optimization\n") ;
	else
		CIO::message(M_INFO, "Disabling LINADD optimization\n") ;

	return true ;  
}

CLabels* CGUIClassifier::classify(CLabels* output)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);

	if (!classifier)
	{
		CIO::message(M_ERROR, "no svm available\n") ;
		return NULL;
	}
	if (!trainfeatures)
	{
		CIO::message(M_ERROR, "no training features available\n") ;
		return NULL;
	}

	if (!testfeatures)
	{
		CIO::message(M_ERROR, "no test features available\n") ;
		return NULL;
	}

	if (!gui->guikernel.is_initialized())
	{
		CIO::message(M_ERROR, "kernel not initialized\n") ;
		return NULL;
	}
	  
	((CKernelMachine*) classifier)->set_kernel(gui->guikernel.get_kernel()) ;

	if ((gui->guikernel.get_kernel()->is_optimizable()) && (gui->guikernel.get_kernel()->get_is_initialized()))
		CIO::message(M_DEBUG, "using kernel optimization\n");

	CIO::message(M_INFO, "starting svm testing\n") ;
	return classifier->classify(output);
}

bool CGUIClassifier::classify_example(INT idx, REAL &result)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);

	if (!classifier)
	{
		CIO::message(M_ERROR, "no svm available\n") ;
		return false;
	}
	if (!trainfeatures)
	{
		CIO::message(M_ERROR, "no training features available\n") ;
		return false;
	}

	if (!testfeatures)
	{
		CIO::message(M_ERROR, "no test features available\n") ;
		return false;
	}

	if (!gui->guikernel.is_initialized())
	{
		CIO::message(M_ERROR, "kernel not initialized\n") ;
		return false;
	}

	((CKernelMachine*) classifier)->set_kernel(gui->guikernel.get_kernel()) ;

	result=classifier->classify_example(idx);
	return true ;
}
