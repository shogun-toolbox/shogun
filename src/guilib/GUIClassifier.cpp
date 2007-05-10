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
#include "guilib/GUIClassifier.h"
#include "gui/GUI.h"
#include "lib/io.h"
#include "features/SparseFeatures.h"
#include "features/RealFileFeatures.h"
#include "features/Labels.h"

#include "classifier/KNN.h"
#include "classifier/PluginEstimate.h"

#include "classifier/LDA.h"
#include "classifier/LPM.h"
#include "classifier/Perceptron.h"
#include "classifier/KernelPerceptron.h"

#include "classifier/LinearClassifier.h"
#include "classifier/SparseLinearClassifier.h"

#ifdef USE_SVMLIGHT
#include "classifier/svm/SVM_light.h"
#include "regression/svr/SVR_light.h"
#endif //USE_SVMLIGHT

#include "classifier/svm/LibSVM.h"
#include "classifier/svm/GPBTSVM.h"
#include "classifier/svm/LibSVM_oneclass.h"
#include "classifier/svm/LibSVM_multiclass.h"

#include "regression/svr/LibSVR.h"

#include "classifier/svm/MPD.h"
#include "classifier/svm/GNPPSVM.h"
#include "classifier/svm/GMNPSVM.h"

#include "classifier/svm/SVMLin.h"
#include "classifier/svm/SubGradientSVM.h"
#include "classifier/svm/SVMPerf.h"


CGUIClassifier::CGUIClassifier(CGUI* g) : CSGObject(), gui(g)
{
	classifier=NULL;

    // Perceptron parameters
	perceptron_learnrate=0.1;
	perceptron_maxiter=1000;

    // SVM parameters
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

	if (strcmp(param,"LIBSVM")==0)
	{
		delete classifier;
		classifier= new CLibSVM();
		SG_INFO( "created SVMlibsvm object\n") ;
	}
#ifdef USE_SVMLIGHT
	else if (strcmp(param,"SVMLIGHT")==0)
	{
		delete classifier;
		classifier= new CSVMLight();
		SG_INFO( "created SVMLight object\n") ;
	}
	else if (strcmp(param,"SVRLIGHT")==0)
	{
		delete classifier;
		classifier= new CSVRLight();
		SG_INFO( "created SVRLight object\n") ;
	}
#endif //USE_SVMLIGHT
	else if (strcmp(param,"GPBTSVM")==0)
	{
		delete classifier;
		classifier= new CGPBTSVM();
		SG_INFO( "created GPBT-SVM object\n") ;
	}
	else if (strcmp(param,"MPDSVM")==0)
	{
		delete classifier;
		classifier= new CMPDSVM();
		SG_INFO( "created MPD-SVM object\n") ;
	}
	else if (strcmp(param,"LIBSVR")==0)
	{
		delete classifier;
		classifier= new CLibSVR();
		SG_INFO( "created SVRlibsvm object\n") ;
	}
	else if (strcmp(param,"KERNELPERCEPTRON")==0)
	{
		delete classifier;
		classifier= new CKernelPerceptron();
		SG_INFO( "created Kernel Perceptron object\n") ;
	}
	else if (strcmp(param,"PERCEPTRON")==0)
	{
		delete classifier;
		classifier= new CPerceptron();
		SG_INFO( "created Perceptron object\n") ;
	}
#ifdef HAVE_LAPACK
	else if (strcmp(param,"LDA")==0)
	{
		delete classifier;
		classifier= new CLDA();
		SG_INFO( "created LDA object\n") ;
	}
#endif
	else if (strcmp(param,"LPM")==0)
	{
		delete classifier;
		classifier= new CLPM();
		SG_INFO( "created LPM object\n") ;
	}
	else if (strcmp(param,"KNN")==0)
	{
		delete classifier;
		classifier= new CKNN();
		SG_INFO( "created KNN object\n") ;
	}
	else if (strcmp(param,"SUBGRADIENT_SVM")==0)
	{
		delete classifier;
		classifier= new CSubGradientSVM();
		SG_INFO( "created Subgradient SVM object\n") ;
	}
	else
	{
		SG_ERROR( "unknown classifier \"%s\"\n", param);
		return false;
	}

	return (classifier!=NULL);
}

bool CGUIClassifier::train(CHAR* param)
{
	param=CIO::skip_spaces(param);
	ASSERT(classifier);

	switch (classifier->get_classifier_type())
	{
		case CT_LIGHT:
		case CT_LIBSVM:
		case CT_MPD:
		case CT_GPBT:
		case CT_CPLEXSVM:
		case CT_GMNPSVM:
		case CT_GNPPSVM:
		case CT_KRR:
			return train_svm(param, false);
			break;
		case CT_PERCEPTRON:
			((CPerceptron*) classifier)->set_learn_rate(perceptron_learnrate);
			((CPerceptron*) classifier)->set_max_iter(perceptron_maxiter);
		case CT_KERNELPERCEPTRON:
		case CT_LDA:
		case CT_LPM:
			return train_linear(param);
			break;
		case CT_KNN:
			return train_knn(param);
			break;
		case CT_SVMLIN:
		case CT_SVMPERF:
		case CT_SUBGRADIENTSVM:
			return train_sparse_linear(param);
		default:
			SG_ERROR( "unknown classifier type\n");
			break;
	};
	return false;
}

bool CGUIClassifier::train_svm(CHAR* param, bool auc_maximization)
{
	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CKernel* kernel=gui->guikernel.get_kernel();

	if (!trainfeatures)
	{
		SG_ERROR( "no trainfeatures available\n") ;
		return false ;
	}

	if (!classifier)
	{
		SG_ERROR( "no classifier available\n") ;
		return false ;
	}

	if (!kernel)
	{
		SG_ERROR( "no kernel available\n");
		return false ;
	}

	if (!trainlabels)
	{
		SG_ERROR( "no trainlabels available\n");
		return false ;
	}

	if ( !gui->guikernel.is_initialized() || !kernel->get_lhs() )
	{
		SG_ERROR( "kernel not initialized\n") ;
		return 0;
	}

	if (trainlabels->get_num_labels() != kernel->get_lhs()->get_num_vectors())
	{
		SG_ERROR( "number of train labels (%d) and training vectors (%d) differs!\n", 
				trainlabels->get_num_labels(), kernel->get_lhs()->get_num_vectors()) ;
		return 0;
	}

	SG_INFO( "starting svm training on %ld vectors using C1=%lf C2=%lf\n", trainlabels->get_num_labels(), svm_C1, svm_C2) ;

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
	
#ifdef USE_SVMLIGHT
	if (auc_maximization)
		((CSVMLight*)svm)->setup_auc_maximization() ;
#endif //USE_SVMLIGHT

	bool result = svm->train();

	kernel->set_precompute_matrix(false,false);
	return result ;	
}

bool CGUIClassifier::train_knn(CHAR* param)
{
	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CDistance* distance=gui->guidistance.get_distance();

	bool result=false;

	if (trainlabels)
	{
		if (distance)
		{
			param=CIO::skip_spaces(param);
			INT k=3;
			sscanf(param, "%d", &k);

			((CKNN*) classifier)->set_labels(trainlabels);
			((CKNN*) classifier)->set_distance(distance);
			((CKNN*) classifier)->set_k(k);
			result=((CKNN*) classifier)->train();
		}
		else
			SG_ERROR( "no distance available\n") ;
	}
	else
		SG_ERROR( "no labels available\n") ;

	return result;
}

bool CGUIClassifier::train_linear(CHAR* param)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CLabels* trainlabels=gui->guilabels.get_train_labels();

	bool result=false;

	if (!trainfeatures)
	{
		SG_ERROR( "no trainfeatures available\n") ;
		return false ;
	}

	if (!trainlabels)
	{
		SG_ERROR( "no labels available\n") ;
		return false;
	}

	((CLinearClassifier*) classifier)->set_labels(trainlabels);
	((CLinearClassifier*) classifier)->set_features((CRealFeatures*) trainfeatures);
	result=((CLinearClassifier*) classifier)->train();

	return result;
}

bool CGUIClassifier::train_sparse_linear(CHAR* param)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CLabels* trainlabels=gui->guilabels.get_train_labels();

	bool result=false;

	if (!trainfeatures)
	{
		SG_ERROR( "no trainfeatures available\n") ;
		return false ;
	}

	if (!trainlabels)
	{
		SG_ERROR( "no labels available\n") ;
		return false;
	}

	((CSparseLinearClassifier*) classifier)->set_labels(trainlabels);
	((CSparseLinearClassifier*) classifier)->set_features((CSparseFeatures<DREAL>*) trainfeatures);
	result=((CLinearClassifier*) classifier)->train();

	return result;
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

	if (!classifier)
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
	((CKernelMachine*) classifier)->set_labels(testlabels);
	((CKernelMachine*) classifier)->set_kernel(gui->guikernel.get_kernel()) ;
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);

	if ( (gui->guikernel.get_kernel()->has_property(KP_LINADD)) && (gui->guikernel.get_kernel()->get_is_initialized()))
		SG_DEBUG( "using kernel optimization\n");

	INT len=0;
	CLabels* predictions= classifier->classify();
	DREAL* output= predictions->get_labels(len);
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

bool CGUIClassifier::load(CHAR* param)
{
	return false;
}

bool CGUIClassifier::save(CHAR* param)
{
	return false;
}

bool CGUIClassifier::set_perceptron_parameters(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le %d", &perceptron_learnrate, &perceptron_maxiter) ;

	if (perceptron_learnrate<=0)
		perceptron_learnrate=0.01;
	if (perceptron_maxiter<=0)
		perceptron_maxiter=1000;

	SG_INFO( "Setting to perceptron parameters (learnrate %f and maxiter: %d\n", perceptron_learnrate, perceptron_maxiter);
	return true ;  
}

bool CGUIClassifier::set_svm_epsilon(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &svm_epsilon) ;

	if (svm_epsilon<0)
		svm_epsilon=1e-4;

	SG_INFO( "Set to svm_epsilon=%f\n", svm_epsilon);
	return true ;  
}

bool CGUIClassifier::set_svr_tube_epsilon(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &svm_tube_epsilon) ;

	if (svm_tube_epsilon<0)
		svm_tube_epsilon=1e-2;

	SG_INFO( "Set to svr_tube_epsilon=%f\n", svm_tube_epsilon);
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

	SG_INFO( "Set to weight_epsilon=%f\n", svm_weight_epsilon);
	SG_INFO( "Set to C_mkl=%f\n", svm_C_mkl);
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

	SG_INFO( "Set to C1=%f C2=%f\n", svm_C1, svm_C2) ;
	return true ;  
}

bool CGUIClassifier::set_svm_qpsize(CHAR* param)
{
	param=CIO::skip_spaces(param);

	svm_qpsize=-1;

	sscanf(param, "%d", &svm_qpsize) ;

	if (svm_qpsize<2)
		svm_qpsize=41;

	SG_INFO( "Set qpsize to svm_qpsize=%d\n", svm_qpsize);
	return true ;  
}

bool CGUIClassifier::set_svm_mkl_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int mkl=1;
	sscanf(param, "%d", &mkl) ;

	svm_use_mkl = (mkl==1);

	if (svm_use_mkl)
		SG_INFO( "Enabling MKL optimization\n") ;
	else
		SG_INFO( "Disabling MKL optimization\n") ;

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
		SG_INFO( "Enabling Kernel Matrix Precomputation\n") ;
	else
		SG_INFO( "Disabling Kernel Matrix Precomputation\n") ;

	if (svm_use_precompute_subkernel)
		SG_INFO( "Enabling Subkernel Matrix Precomputation\n") ;
	else
		SG_INFO( "Disabling Subkernel Matrix Precomputation\n") ;

	if (svm_use_precompute_subkernel_light)
		SG_INFO( "Enabling Subkernel Matrix Precomputation by SVM Light\n") ;
	else
		SG_INFO( "Disabling Subkernel Matrix Precomputation by SVM Light\n") ;

	return true ;  
}

bool CGUIClassifier::set_svm_linadd_enabled(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int linadd=1;
	sscanf(param, "%d", &linadd) ;

	svm_use_linadd = (linadd==1);
	
	if (svm_use_linadd)
		SG_INFO( "Enabling LINADD optimization\n") ;
	else
		SG_INFO( "Disabling LINADD optimization\n") ;

	return true ;  
}

CLabels* CGUIClassifier::classify(CLabels* output)
{
	ASSERT(classifier);

	switch (classifier->get_classifier_type())
	{
		case CT_LIGHT:
		case CT_LIBSVM:
		case CT_MPD:
		case CT_GPBT:
		case CT_CPLEXSVM:
		case CT_KERNELPERCEPTRON:
			return classify_kernelmachine(output);
		case CT_KNN:
			return classify_distancemachine(output);
		case CT_PERCEPTRON:
		case CT_LDA:
		case CT_LPM:
			return classify_linear(output);
		case CT_SVMLIN:
		case CT_SVMPERF:
		case CT_SUBGRADIENTSVM:
			return classify_sparse_linear(output);
		default:
			SG_ERROR( "unknown classifier type\n");
			break;
	};

	return false;
}

CLabels* CGUIClassifier::classify_kernelmachine(CLabels* output)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);

	if (!classifier)
	{
		SG_ERROR( "no kernelmachine available\n") ;
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
	  
	((CKernelMachine*) classifier)->set_kernel(gui->guikernel.get_kernel()) ;
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);
	SG_INFO( "starting kernel machine testing\n") ;
	return classifier->classify(output);
}

CLabels* CGUIClassifier::classify_distancemachine(CLabels* output)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	gui->guidistance.get_distance()->set_precompute_matrix(false);

	if (!classifier)
	{
		SG_ERROR( "no kernelmachine available\n") ;
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

	if (!gui->guidistance.is_initialized())
	{
		SG_ERROR( "distance not initialized\n") ;
		return NULL;
	}
	  
	((CDistanceMachine*) classifier)->set_distance(gui->guidistance.get_distance()) ;
	gui->guidistance.get_distance()->set_precompute_matrix(false);
	SG_INFO( "starting distance machine testing\n") ;
	return classifier->classify(output);
}


CLabels* CGUIClassifier::classify_linear(CLabels* output)
{
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

	if (!classifier)
	{
		SG_ERROR( "no classifier available\n") ;
		return NULL;
	}
	if (!testfeatures)
	{
		SG_ERROR( "no test features available\n") ;
		return NULL;
	}


	((CLinearClassifier*) classifier)->set_features((CRealFeatures*) testfeatures);
	SG_INFO( "starting linear classifier testing\n") ;
	return classifier->classify(output);
}

CLabels* CGUIClassifier::classify_sparse_linear(CLabels* output)
{
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

	if (!classifier)
	{
		SG_ERROR( "no svm available\n") ;
		return NULL;
	}
	if (!testfeatures)
	{
		SG_ERROR( "no test features available\n") ;
		return NULL;
	}

	((CSparseLinearClassifier*) classifier)->set_features((CSparseFeatures<DREAL>*) testfeatures);
	SG_INFO( "starting linear classifier testing\n") ;
	return classifier->classify(output);
}

bool CGUIClassifier::classify_example(INT idx, DREAL &result)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	gui->guikernel.get_kernel()->set_precompute_matrix(false,false);

	if (!classifier)
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

	((CKernelMachine*) classifier)->set_kernel(gui->guikernel.get_kernel()) ;

	result=classifier->classify_example(idx);
	return true ;
}
#endif
