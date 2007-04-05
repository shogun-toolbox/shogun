/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
#include "lib/config.h"

#ifndef HAVE_SWIG
#include "guilib/GUISVM.h"
#include "gui/GUI.h"
#include "lib/io.h"
#include "features/RealFileFeatures.h"
#include "features/Labels.h"

#ifdef USE_SVMLIGHT
#include "classifier/svm/SVM_light.h"
#include "regression/svr/SVR_light.h"
#endif //USE_SVMLIGHT

#include "classifier/svm/LibSVM.h"
#include "classifier/svm/GPBTSVM.h"
#include "classifier/svm/LibSVM_oneclass.h"

#include "regression/svr/LibSVR.h"

#include "classifier/svm/MPD.h"
#include "classifier/svm/GNPPSVM.h"
#include "classifier/svm/GMNPSVM.h"

CGUISVM::CGUISVM(CGUI * gui_)
  : CSGObject(), gui(gui_)
{
	svm=NULL;
	qpsize=41;
	C1=1;
	C2=1;
	C_mkl=0;
	weight_epsilon=1e-5;
	epsilon=1e-5;
	max_train_time=0;
	tube_epsilon=1e-2;
	nu=1e-2;
	use_shrinking = true ;

    // MKL stuff
	use_mkl = false ;
	use_batch_computation = true ;
	use_linadd = true ;
	use_precompute = false ;
	use_precompute_subkernel = false ;
	use_precompute_subkernel_light = false ;
}

CGUISVM::~CGUISVM()
{
	delete svm;
}

bool CGUISVM::new_svm(CHAR* param)
{
	param=CIO::skip_spaces(param);

	if (strcmp(param,"LIBSVM")==0)
	{
		delete svm;
		svm= new CLibSVM();
		SG_INFO( "created SVMlibsvm object\n") ;
	}
	else if (strcmp(param,"LIBSVM_ONECLASS")==0)
	{
		delete svm;
		svm = new CLibSVMOneclass();
		SG_INFO( "created SVMlibsvm object for oneclass\n");
	}
#ifdef USE_SVMLIGHT
	else if (strcmp(param,"LIGHT")==0)
	{
		delete svm;
		svm= new CSVMLight();
		SG_INFO( "created SVMLight object\n") ;
	}
	else if (strcmp(param,"SVRLIGHT")==0)
	{
		delete svm;
		svm= new CSVRLight();
		SG_INFO( "created SVRLight object\n") ;
	}
#endif //USE_SVMLIGHT
	else if (strcmp(param,"GPBT")==0)
	{
		delete svm;
		svm= new CGPBTSVM();
		SG_INFO( "created GPBT-SVM object\n") ;
	}
	else if (strcmp(param,"MPD")==0)
	{
		delete svm;
		svm= new CMPDSVM();
		SG_INFO( "created MPD-SVM object\n") ;
	}
	else if (strcmp(param,"GNPP")==0)
	{
		delete svm;
		svm= new CGNPPSVM();
		SG_INFO( "created GNPP-SVM object\n") ;
	}
	else if (strcmp(param,"GMNP")==0)
	{
		delete svm;
		svm= new CGMNPSVM();
		SG_INFO( "created GMNP-SVM object\n") ;
	}
	else if (strcmp(param,"LIBSVR")==0)
	{
		delete svm;
		svm= new CLibSVR();
		SG_INFO( "created SVRlibsvm object\n") ;
	}
	else
		return false;

	return (svm!=NULL);
}

bool CGUISVM::train(CHAR* param, bool auc_maximization)
{
	param=CIO::skip_spaces(param);

	bool oneclass = (svm->get_classifier_type()==CT_LIBSVMONECLASS);
	
	CLabels* trainlabels=NULL;
	if(!oneclass)
		trainlabels=gui->guilabels.get_train_labels();
	CKernel* kernel=gui->guikernel.get_kernel();

	if (!svm)
	{
		SG_ERROR( "no svm available\n") ;
		return false ;
	}

	if (!kernel)
	{
		SG_ERROR( "no kernel available\n");
		return false ;
	}

	if (!trainlabels && !oneclass)
	{
		SG_ERROR( "no trainlabels available\n");
		return false ;
	}

	if ( !gui->guikernel.is_initialized() || !kernel->get_lhs() )
	{
		SG_ERROR( "kernel not initialized\n") ;
		return 0;
	}

	if (!oneclass && trainlabels->get_num_labels() != kernel->get_lhs()->get_num_vectors())
	{
		SG_ERROR( "number of train labels (%d) and training vectors (%d) differs!\n", 
				trainlabels->get_num_labels(), kernel->get_lhs()->get_num_vectors()) ;
		return 0;
	}

	SG_INFO( "starting svm training on %ld vectors using C1=%lf C2=%lf\n", kernel->get_lhs()->get_num_vectors(), C1, C2) ;

	svm->set_weight_epsilon(weight_epsilon);
	svm->set_epsilon(epsilon);
	svm->set_max_train_time(max_train_time);
	svm->set_tube_epsilon(tube_epsilon);
	svm->set_nu(nu);
	svm->set_C_mkl(C_mkl);
	svm->set_C(C1, C2);
	svm->set_qpsize(qpsize);
	svm->set_mkl_enabled(use_mkl);
	svm->set_shrinking_enabled(use_shrinking);
	svm->set_linadd_enabled(use_linadd);
	svm->set_batch_computation_enabled(use_batch_computation);
	if(!oneclass)
		((CKernelMachine*) svm)->set_labels(trainlabels);
	((CKernelMachine*) svm)->set_kernel(kernel);
	((CSVM*) svm)->set_precomputed_subkernels_enabled(use_precompute_subkernel_light) ;
	kernel->set_precompute_matrix(use_precompute, use_precompute_subkernel);
	
#ifdef USE_SVMLIGHT
	if (auc_maximization)
		((CSVMLight*)svm)->setup_auc_maximization() ;
#endif //USE_SVMLIGHT

	bool result = svm->train();

	//DREAL x=svm->compute_objective();
	//SG_INFO("REFERENCE objective:%f\n", x);

	kernel->set_precompute_matrix(false,false);
	return result ;	
}

bool CGUISVM::test(CHAR* param)
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
			SG_ERROR( "could not open %s\n",outputname);
			return false;
		}

		if (numargs==2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				SG_ERROR( "could not open %s\n",rocfname);
				return false;
			}
		}
	}

	CLabels* testlabels=gui->guilabels.get_test_labels();
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	SG_DEBUG( "I:training: %ld examples each %ld features\n", ((CRealFeatures*) trainfeatures)->get_num_vectors(), ((CRealFeatures*) trainfeatures)->get_num_features());
	SG_DEBUG( "I:testing: %ld examples each %ld features\n", ((CRealFeatures*) testfeatures)->get_num_vectors(), ((CRealFeatures*) testfeatures)->get_num_features());

	if (!svm)
	{
		SG_ERROR( "no svm available") ;
		return false ;
	}
	if (!trainfeatures)
	{
		SG_ERROR( "no training features available") ;
		return false ;
	}

	if (!testfeatures)
	{
		SG_ERROR( "no test features available") ;
		return false ;
	}

	if (!testlabels)
	{
		SG_ERROR( "no test labels available") ;
		return false ;
	}

	if (!gui->guikernel.is_initialized())
	{
		SG_ERROR( "kernel not initialized\n") ;
		return 0;
	}

	SG_INFO( "starting svm testing\n") ;
	((CKernelMachine*) svm)->set_labels(testlabels);
	((CKernelMachine*) svm)->set_kernel(gui->guikernel.get_kernel()) ;
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);
	svm->set_batch_computation_enabled(use_batch_computation);

	CLabels* outlab=svm->classify();
	ASSERT(outlab);
	
	INT len=0;
	DREAL* output=outlab->get_labels(len);
	delete outlab;

	INT total=	testfeatures->get_num_vectors();
	INT* label= testlabels->get_int_labels(len);

	ASSERT(label);
	SG_DEBUG( "len:%d total:%d\n", len, total);
	ASSERT(len==total);

	gui->guimath.evaluate_results(output, label, total, outputfile, rocfile);

	if (rocfile)
		fclose(rocfile);
	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	delete[] output;
	delete[] label;
	return true;
}

bool CGUISVM::load(CHAR* param)
{
    bool result=false;
    param=CIO::skip_spaces(param);
    CHAR filename[1024];
    CHAR type[1024];

    if ((sscanf(param, "%s %s", filename, type))==2)
    {

	if (new_svm(type))
	{
	    FILE* model_file=fopen(filename, "r");

	    if (model_file)
	    {
		if (svm && svm->load(model_file))
		{
		    printf("file successfully read\n");
		    result=true;
		}
		else
		    SG_ERROR( "svm creation/loading failed\n");

		fclose(model_file);
	    }
	    else
		SG_ERROR( "opening file %s failed\n", filename);

	    return result;
	}
	else
	    SG_ERROR( "type of svm unknown\n");
    }
    else
	SG_ERROR( "see help for parameters\n");
	return false;
}

bool CGUISVM::save(CHAR* param)
{
    bool result=false;
    param=CIO::skip_spaces(param);

    if (svm)
    {
	FILE* file=fopen(param, "w");

	if ((!file) ||	(!svm->save(file)))
	    printf("writing to file %s failed!\n", param);
	else
	{
	    printf("successfully written svm into \"%s\" !\n", param);
	    result=true;
	}

	if (file)
	    fclose(file);
    }
    else
	SG_ERROR( "create svm first\n");

    return result;
}

bool CGUISVM::set_svm_epsilon(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &epsilon) ;

	if (epsilon<0)
		epsilon=1e-4;

	SG_INFO( "Set to svm_epsilon=%f\n", epsilon);
	return true ;  
}

bool CGUISVM::set_svm_max_train_time(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%lf", &max_train_time) ;

        if (max_train_time > 0) 
		SG_INFO( "Set to svm_max_train_time=%f\n", max_train_time);
	else
		SG_INFO( "Disabling svm_max_train_time\n");
	return true ;  
}


bool CGUISVM::set_svr_tube_epsilon(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &tube_epsilon) ;

	if (tube_epsilon<0)
		tube_epsilon=1e-2;

	SG_INFO( "Set to svr_tube_epsilon=%f\n", tube_epsilon);
	return true ;  
}

bool CGUISVM::set_svm_one_class_nu(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &nu) ;

	if (nu<0 || nu>1)
		nu=0.5;

	SG_INFO( "Set to nu=%f\n", nu);
	return true ;  
}

bool CGUISVM::set_mkl_parameters(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le %le", &weight_epsilon, &C_mkl) ;

	if (weight_epsilon<0)
		weight_epsilon=1e-4;
	if (C_mkl<0)
		C_mkl=1e-4 ;

	SG_INFO( "Set to weight_epsilon=%f\n", weight_epsilon);
	SG_INFO( "Set to C_mkl=%f\n", C_mkl);
	return true ;  
}

bool CGUISVM::set_C(CHAR* param)
{
	param=CIO::skip_spaces(param);

	C1=-1;
	C2=-1;

	sscanf(param, "%le %le", &C1, &C2) ;

	if (C1<0)
		C1=1.0;
	if (C2<0)
		C2=C1;

	SG_INFO( "Set to C1=%f C2=%f\n", C1, C2) ;
	return true ;  
}

bool CGUISVM::set_qpsize(CHAR* param)
{
	param=CIO::skip_spaces(param);

	qpsize=-1;

	sscanf(param, "%d", &qpsize) ;

	if (qpsize<2)
		qpsize=41;

	SG_INFO( "Set qpsize to qpsize=%d\n", qpsize);
	return true ;  
}

bool CGUISVM::set_mkl_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int mkl=1;
	sscanf(param, "%d", &mkl) ;

	use_mkl = (mkl==1);

	if (use_mkl)
		SG_INFO( "Enabling MKL optimization\n") ;
	else
		SG_INFO( "Disabling MKL optimization\n") ;

	return true ;  
}

bool CGUISVM::set_shrinking_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int shrinking=1;
	sscanf(param, "%d", &shrinking) ;

	use_shrinking = (shrinking==1);

	if (use_shrinking)
		SG_INFO( "Enabling shrinking optimization\n") ;
	else
		SG_INFO( "Disabling shrinking optimization\n") ;

	return true ;  
}

bool CGUISVM::set_batch_computation_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int batch_computation=1;
	sscanf(param, "%d", &batch_computation) ;

	use_batch_computation = (batch_computation==1);

	if (use_batch_computation)
		SG_INFO( "Enabling batch computation\n") ;
	else
		SG_INFO( "Disabling batch computation\n") ;

	return true ;  
}

bool CGUISVM::set_precompute_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int precompute=1;
	sscanf(param, "%d", &precompute) ;

	use_precompute = (precompute==1);
	use_precompute_subkernel = (precompute==2);
	use_precompute_subkernel_light = (precompute==3);

	if (use_precompute)
		SG_INFO( "Enabling Kernel Matrix Precomputation\n") ;
	else
		SG_INFO( "Disabling Kernel Matrix Precomputation\n") ;

	if (use_precompute_subkernel)
		SG_INFO( "Enabling Subkernel Matrix Precomputation\n") ;
	else
		SG_INFO( "Disabling Subkernel Matrix Precomputation\n") ;

	if (use_precompute_subkernel_light)
		SG_INFO( "Enabling Subkernel Matrix Precomputation by SVM Light\n") ;
	else
		SG_INFO( "Disabling Subkernel Matrix Precomputation by SVM Light\n") ;

	return true ;  
}

bool CGUISVM::set_linadd_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int linadd=1;
	sscanf(param, "%d", &linadd) ;

	use_linadd = (linadd==1);
	
	if (use_linadd)
		SG_INFO( "Enabling LINADD optimization\n") ;
	else
		SG_INFO( "Disabling LINADD optimization\n") ;

	return true ;  
}

CLabels* CGUISVM::classify(CLabels* output)
{
	CLabels* testlabels=gui->guilabels.get_test_labels();
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);

	if (!svm)
	{
		SG_ERROR( "no svm available\n") ;
		return NULL;
	}
	if (!trainfeatures)
	{
		SG_ERROR( "no training features available\n") ;
		return NULL;
	}

	if (!testfeatures)
	{
		SG_ERROR( "no test features available\n") ;
		return NULL;
	}

	if (!gui->guikernel.is_initialized())
	{
		SG_ERROR( "kernel not initialized\n") ;
		return NULL;
	}
	  
	((CKernelMachine*) svm)->set_labels(testlabels);
	((CKernelMachine*) svm)->set_kernel(gui->guikernel.get_kernel()) ;
	svm->set_batch_computation_enabled(use_batch_computation);
	SG_INFO( "starting svm testing\n") ;
	return svm->classify(output);
}

bool CGUISVM::classify_example(INT idx, DREAL &result)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);

	if (!svm)
	{
		SG_ERROR( "no svm available\n") ;
		return false;
	}
	if (!trainfeatures)
	{
		SG_ERROR( "no training features available\n") ;
		return false;
	}

	if (!testfeatures)
	{
		SG_ERROR( "no test features available\n") ;
		return false;
	}

	if (!gui->guikernel.is_initialized())
	{
		SG_ERROR( "kernel not initialized\n") ;
		return false;
	}

	((CKernelMachine*) svm)->set_kernel(gui->guikernel.get_kernel()) ;

	result=svm->classify_example(idx);
	return true ;
}
#endif
