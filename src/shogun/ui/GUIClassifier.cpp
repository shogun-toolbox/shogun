/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#include <shogun/ui/GUIClassifier.h>
#include <shogun/ui/SGInterface.h>

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>

#include <shogun/features/SparseFeatures.h>
#include <shogun/features/RealFileFeatures.h>
#include <shogun/labels/Labels.h>

#include <shogun/kernel/AUCKernel.h>

#include <shogun/multiclass/KNN.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/clustering/Hierarchical.h>
#include <shogun/classifier/PluginEstimate.h>

#include <shogun/classifier/LDA.h>
#include <shogun/classifier/LPM.h>
#include <shogun/classifier/LPBoost.h>
#include <shogun/classifier/Perceptron.h>

#include <shogun/machine/LinearMachine.h>

#ifdef USE_SVMLIGHT
#include <shogun/classifier/svm/SVMLight.h>
#include <shogun/classifier/svm/SVMLightOneClass.h>
#include <shogun/regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT

#include <shogun/classifier/mkl/MKLClassification.h>
#include <shogun/regression/svr/MKLRegression.h>
#include <shogun/classifier/mkl/MKLOneClass.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/multiclass/LaRank.h>
#include <shogun/classifier/svm/GPBTSVM.h>
#include <shogun/classifier/svm/LibSVMOneClass.h>
#include <shogun/multiclass/MulticlassLibSVM.h>

#include <shogun/regression/svr/LibSVR.h>
#include <shogun/regression/KernelRidgeRegression.h>

#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/classifier/svm/MPDSVM.h>
#include <shogun/classifier/svm/GNPPSVM.h>
#include <shogun/multiclass/GMNPSVM.h>
#include <shogun/multiclass/ScatterSVM.h>

#include <shogun/classifier/svm/SVMLin.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/classifier/svm/SVMSGD.h>
#include <shogun/classifier/svm/WDSVMOcas.h>

#include <shogun/io/SerializableAsciiFile.h>

using namespace shogun;

CGUIClassifier::CGUIClassifier(CSGInterface* ui_)
: CSGObject(), ui(ui_)
{
	constraint_generator=NULL;
	classifier=NULL;
	max_train_time=0;

    // Perceptron parameters
	perceptron_learnrate=0.1;
	perceptron_maxiter=1000;

    // SVM parameters
	svm_qpsize=41;
	svm_bufsize=3000;
	svm_max_qpsize=1000;
	mkl_norm=1;
	ent_lambda=0;
	mkl_block_norm=4;
	svm_C1=1;
	svm_C2=1;
	C_mkl=0;
	mkl_use_interleaved=true;
	svm_weight_epsilon=1e-5;
	svm_epsilon=1e-5;
	svm_tube_epsilon=1e-2;
	svm_nu=0.5;
	svm_use_shrinking = true ;

	svm_use_bias = true;
	svm_use_batch_computation = true ;
	svm_use_linadd = true ;
	svm_do_auc_maximization = false ;

	// KRR parameters
	krr_tau=1;

	solver_type=ST_AUTO;
}

CGUIClassifier::~CGUIClassifier()
{
	SG_UNREF(classifier);
	SG_UNREF(constraint_generator);
}

bool CGUIClassifier::new_classifier(char* name, int32_t d, int32_t from_d)
{
	if (strcmp(name,"LIBSVM_ONECLASS")==0)
	{
		SG_UNREF(classifier);
		classifier = new CLibSVMOneClass();
		SG_INFO("created SVMlibsvm object for oneclass\n")
	}
	else if (strcmp(name,"LIBSVM_MULTICLASS")==0)
	{
		SG_UNREF(classifier);
		classifier = new CMulticlassLibSVM();
		SG_INFO("created SVMlibsvm object for multiclass\n")
	}
	else if (strcmp(name,"LIBSVM_NUMULTICLASS")==0)
	{
		SG_UNREF(classifier);
		classifier= new CMulticlassLibSVM(LIBSVM_NU_SVC);
		SG_INFO("created SVMlibsvm object for multiclass\n")
	}
#ifdef USE_SVMLIGHT
	else if (strcmp(name,"SCATTERSVM_NO_BIAS_SVMLIGHT")==0)
	{
		SG_UNREF(classifier);
		classifier= new CScatterSVM(NO_BIAS_SVMLIGHT);
		SG_INFO("created ScatterSVM NO BIAS SVMLIGHT object\n")
	}
#endif //USE_SVMLIGHT
	else if (strcmp(name,"SCATTERSVM_NO_BIAS_LIBSVM")==0)
	{
		SG_UNREF(classifier);
		classifier= new CScatterSVM(NO_BIAS_LIBSVM);
		SG_INFO("created ScatterSVM NO BIAS LIBSVM object\n")
	}
	else if (strcmp(name,"SCATTERSVM_TESTRULE1")==0)
	{
		SG_UNREF(classifier);
		classifier= new CScatterSVM(TEST_RULE1);
		SG_INFO("created ScatterSVM TESTRULE1 object\n")
	}
	else if (strcmp(name,"SCATTERSVM_TESTRULE2")==0)
	{
		SG_UNREF(classifier);
		classifier= new CScatterSVM(TEST_RULE2);
		SG_INFO("created ScatterSVM TESTRULE2 object\n")
	}
	else if (strcmp(name,"LIBSVM_NU")==0)
	{
		SG_UNREF(classifier);
		classifier= new CLibSVM(LIBSVM_NU_SVC);
		SG_INFO("created SVMlibsvm object\n")
	}
	else if (strcmp(name,"LIBSVM")==0)
	{
		SG_UNREF(classifier);
		classifier= new CLibSVM();
		SG_INFO("created SVMlibsvm object\n")
	}
	else if (strcmp(name,"LARANK")==0)
	{
		SG_UNREF(classifier);
		classifier= new CLaRank();
		SG_INFO("created LaRank object\n")
	}
#ifdef USE_SVMLIGHT
	else if ((strcmp(name,"LIGHT")==0) || (strcmp(name,"SVMLIGHT")==0))
	{
		SG_UNREF(classifier);
		classifier= new CSVMLight();
		SG_INFO("created SVMLight object\n")
	}
	else if (strcmp(name,"SVMLIGHT_ONECLASS")==0)
	{
		SG_UNREF(classifier);
		classifier= new CSVMLightOneClass();
		SG_INFO("created SVMLightOneClass object\n")
	}
	else if (strcmp(name,"SVRLIGHT")==0)
	{
		SG_UNREF(classifier);
		classifier= new CSVRLight();
		SG_INFO("created SVRLight object\n")
	}
#endif //USE_SVMLIGHT
	else if (strcmp(name,"GPBTSVM")==0)
	{
		SG_UNREF(classifier);
		classifier= new CGPBTSVM();
		SG_INFO("created GPBT-SVM object\n")
	}
	else if (strcmp(name,"MPDSVM")==0)
	{
		SG_UNREF(classifier);
		classifier= new CMPDSVM();
		SG_INFO("created MPD-SVM object\n")
	}
	else if (strcmp(name,"GNPPSVM")==0)
	{
		SG_UNREF(classifier);
		classifier= new CGNPPSVM();
		SG_INFO("created GNPP-SVM object\n")
	}
	else if (strcmp(name,"GMNPSVM")==0)
	{
		SG_UNREF(classifier);
		classifier= new CGMNPSVM();
		SG_INFO("created GMNP-SVM object\n")
	}
	else if (strcmp(name,"LIBSVR")==0)
	{
		SG_UNREF(classifier);
		classifier= new CLibSVR();
		SG_INFO("created SVRlibsvm object\n")
	}
#ifdef HAVE_LAPACK
	else if (strcmp(name, "KERNELRIDGEREGRESSION")==0)
	{
		SG_UNREF(classifier);
		classifier=new CKernelRidgeRegression(krr_tau, ui->ui_kernel->get_kernel(),
			ui->ui_labels->get_train_labels());
		SG_INFO("created KernelRidgeRegression object %p\n", classifier)
	}
#endif //HAVE_LAPACK
	else if (strcmp(name,"PERCEPTRON")==0)
	{
		SG_UNREF(classifier);
		classifier= new CPerceptron();
		SG_INFO("created Perceptron object\n")
	}
#ifdef HAVE_LAPACK
	else if (strncmp(name,"LIBLINEAR",9)==0)
	{
		LIBLINEAR_SOLVER_TYPE st=L2R_LR;

		if (strcmp(name,"LIBLINEAR_L2R_LR")==0)
		{
			st=L2R_LR;
			SG_INFO("created LibLinear l2 regularized logistic regression object\n")
		}
		else if (strcmp(name,"LIBLINEAR_L2R_L2LOSS_SVC_DUAL")==0)
		{
			st=L2R_L2LOSS_SVC_DUAL;
			SG_INFO("created LibLinear l2 regularized l2 loss SVM dual object\n")
		}
		else if (strcmp(name,"LIBLINEAR_L2R_L2LOSS_SVC")==0)
		{
			st=L2R_L2LOSS_SVC;
			SG_INFO("created LibLinear l2 regularized l2 loss SVM primal object\n")
		}
		else if (strcmp(name,"LIBLINEAR_L1R_L2LOSS_SVC")==0)
		{
			st=L1R_L2LOSS_SVC;
			SG_INFO("created LibLinear l1 regularized l2 loss SVM primal object\n")
		}
		else if (strcmp(name,"LIBLINEAR_L2R_L1LOSS_SVC_DUAL")==0)
		{
			st=L2R_L1LOSS_SVC_DUAL;
			SG_INFO("created LibLinear l2 regularized l1 loss dual SVM object\n")
		}
		else
			SG_ERROR("unknown liblinear type\n")

		SG_UNREF(classifier);
		classifier= new CLibLinear(st);
		((CLibLinear*) classifier)->set_C(svm_C1, svm_C2);
		((CLibLinear*) classifier)->set_epsilon(svm_epsilon);
		((CLibLinear*) classifier)->set_bias_enabled(svm_use_bias);
	}
#endif //HAVE_LAPACK
#ifdef HAVE_EIGEN3
	else if (strcmp(name,"LDA")==0)
	{
		SG_UNREF(classifier);
		classifier= new CLDA();
		SG_INFO("created LDA object\n")
	}
#endif //HAVE_EIGEN3
#ifdef USE_CPLEX
	else if (strcmp(name,"LPM")==0)
	{
		SG_UNREF(classifier);
		classifier= new CLPM();
		((CLPM*) classifier)->set_C(svm_C1, svm_C2);
		((CLPM*) classifier)->set_epsilon(svm_epsilon);
		((CLPM*) classifier)->set_bias_enabled(svm_use_bias);
		((CLPM*) classifier)->set_max_train_time(max_train_time);
		SG_INFO("created LPM object\n")
	}
	else if (strcmp(name,"LPBOOST")==0)
	{
		SG_UNREF(classifier);
		classifier= new CLPBoost();
		((CLPBoost*) classifier)->set_C(svm_C1, svm_C2);
		((CLPBoost*) classifier)->set_epsilon(svm_epsilon);
		((CLPBoost*) classifier)->set_bias_enabled(svm_use_bias);
		((CLPBoost*) classifier)->set_max_train_time(max_train_time);
		SG_INFO("created LPBoost object\n")
	}
#endif //USE_CPLEX
	else if (strncmp(name,"KNN", strlen("KNN"))==0)
	{
		SG_UNREF(classifier);
		classifier= new CKNN();
		SG_INFO("created KNN object\n")
	}
	else if (strncmp(name,"KMEANS", strlen("KMEANS"))==0)
	{
		SG_UNREF(classifier);
		classifier= new CKMeans();
		SG_INFO("created KMeans object\n")
	}
	else if (strncmp(name,"HIERARCHICAL", strlen("HIERARCHICAL"))==0)
	{
		SG_UNREF(classifier);
		classifier= new CHierarchical();
		SG_INFO("created Hierarchical clustering object\n")
	}
	else if (strcmp(name,"SVMLIN")==0)
	{
		SG_UNREF(classifier);
		classifier= new CSVMLin();
		((CSVMLin*) classifier)->set_C(svm_C1, svm_C2);
		((CSVMLin*) classifier)->set_epsilon(svm_epsilon);
		((CSVMLin*) classifier)->set_bias_enabled(svm_use_bias);
		SG_INFO("created SVMLin object\n")
	}
	else if (strncmp(name,"WDSVMOCAS", strlen("WDSVMOCAS"))==0)
	{
		SG_UNREF(classifier);
		classifier= new CWDSVMOcas(SVM_OCAS);

		((CWDSVMOcas*) classifier)->set_bias_enabled(svm_use_bias);
		((CWDSVMOcas*) classifier)->set_degree(d, from_d);
		((CWDSVMOcas*) classifier)->set_C(svm_C1, svm_C2);
		((CWDSVMOcas*) classifier)->set_epsilon(svm_epsilon);
		((CWDSVMOcas*) classifier)->set_bufsize(svm_bufsize);
		SG_INFO("created Weighted Degree Kernel SVM Ocas(OCAS) object of order %d (from order:%d)\n", d, from_d)
	}
	else if (strcmp(name,"SVMOCAS")==0)
	{
		SG_UNREF(classifier);
		classifier= new CSVMOcas(SVM_OCAS);

		((CSVMOcas*) classifier)->set_C(svm_C1, svm_C2);
		((CSVMOcas*) classifier)->set_epsilon(svm_epsilon);
		((CSVMOcas*) classifier)->set_bufsize(svm_bufsize);
		((CSVMOcas*) classifier)->set_bias_enabled(svm_use_bias);
		SG_INFO("created SVM Ocas(OCAS) object\n")
	}
	else if (strcmp(name,"SVMSGD")==0)
	{
		SG_UNREF(classifier);
		classifier= new CSVMSGD(svm_C1);
		((CSVMSGD*) classifier)->set_bias_enabled(svm_use_bias);
		SG_INFO("created SVM SGD object\n")
	}
	else if (strcmp(name,"SVMBMRM")==0 || (strcmp(name,"SVMPERF")==0))
	{
		SG_UNREF(classifier);
		classifier= new CSVMOcas(SVM_BMRM);

		((CSVMOcas*) classifier)->set_C(svm_C1, svm_C2);
		((CSVMOcas*) classifier)->set_epsilon(svm_epsilon);
		((CSVMOcas*) classifier)->set_bufsize(svm_bufsize);
		((CSVMOcas*) classifier)->set_bias_enabled(svm_use_bias);
		SG_INFO("created SVM Ocas(BMRM/PERF) object\n")
	}
	else if (strcmp(name,"MKL_CLASSIFICATION")==0)
	{
		SG_UNREF(classifier);
		classifier= new CMKLClassification();
	}
	else if (strcmp(name,"MKL_ONECLASS")==0)
	{
		SG_UNREF(classifier);
		classifier= new CMKLOneClass();
	}
	else if (strcmp(name,"MKL_MULTICLASS")==0)
	{
		SG_UNREF(classifier);
		classifier= new CMKLMulticlass();
	}
	else if (strcmp(name,"MKL_REGRESSION")==0)
	{
		SG_UNREF(classifier);
		classifier= new CMKLRegression();
	}
	else
	{
		SG_ERROR("Unknown classifier %s.\n", name)
		return false;
	}
	SG_REF(classifier);

	return (classifier!=NULL);
}

bool CGUIClassifier::train_mkl_multiclass()
{
	CMKLMulticlass* mkl= (CMKLMulticlass*) classifier;
	if (!mkl)
		SG_ERROR("No MKL available.\n")

	CLabels* trainlabels=ui->ui_labels->get_train_labels();
	if (!trainlabels)
		SG_ERROR("No trainlabels available.\n")

	CKernel* kernel=ui->ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	bool success=ui->ui_kernel->init_kernel("TRAIN");

	if (!success || !ui->ui_kernel->is_initialized() || !kernel->has_features())
		SG_ERROR("Kernel not initialized / no train features available.\n")

	int32_t num_vec=kernel->get_num_vec_lhs();
	if (trainlabels->get_num_labels() != num_vec)
		SG_ERROR("Number of train labels (%d) and training vectors (%d) differs!\n", trainlabels->get_num_labels(), num_vec)

	SG_INFO("Starting MC-MKL training on %ld vectors using C1=%lf C2=%lf epsilon=%lf\n", num_vec, svm_C1, svm_C2, svm_epsilon)

	mkl->set_mkl_epsilon(svm_weight_epsilon);
	mkl->set_mkl_norm(mkl_norm);
	//mkl->set_max_num_mkliters(-1);
	mkl->set_solver_type(solver_type);
	mkl->set_bias_enabled(svm_use_bias);
	mkl->set_epsilon(svm_epsilon);
	mkl->set_max_train_time(max_train_time);
	mkl->set_tube_epsilon(svm_tube_epsilon);
	mkl->set_nu(svm_nu);
	mkl->set_C(svm_C1);
	mkl->set_qpsize(svm_qpsize);
	mkl->set_shrinking_enabled(svm_use_shrinking);
	mkl->set_linadd_enabled(svm_use_linadd);
	mkl->set_batch_computation_enabled(svm_use_batch_computation);

	((CKernelMulticlassMachine*) mkl)->set_labels(trainlabels);
	((CKernelMulticlassMachine*) mkl)->set_kernel(kernel);

	return mkl->train();
}

bool CGUIClassifier::train_mkl()
{
	CMKL* mkl= (CMKL*) classifier;
	if (!mkl)
		SG_ERROR("No SVM available.\n")

	bool oneclass=(mkl->get_classifier_type()==CT_LIBSVMONECLASS);
	CLabels* trainlabels=NULL;
	if(!oneclass)
		trainlabels=ui->ui_labels->get_train_labels();
	else
		SG_INFO("Training one class mkl.\n")
	if (!trainlabels && !oneclass)
		SG_ERROR("No trainlabels available.\n")

	CKernel* kernel=ui->ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	bool success=ui->ui_kernel->init_kernel("TRAIN");
	if (!success || !ui->ui_kernel->is_initialized() || !kernel->has_features())
		SG_ERROR("Kernel not initialized.\n")

	int32_t num_vec=kernel->get_num_vec_lhs();
	if (!oneclass && trainlabels->get_num_labels() != num_vec)
		SG_ERROR("Number of train labels (%d) and training vectors (%d) differs!\n", trainlabels->get_num_labels(), num_vec)

	SG_INFO("Starting SVM training on %ld vectors using C1=%lf C2=%lf epsilon=%lf\n", num_vec, svm_C1, svm_C2, svm_epsilon)

	if (constraint_generator)
		mkl->set_constraint_generator(constraint_generator);
	mkl->set_solver_type(solver_type);
	mkl->set_bias_enabled(svm_use_bias);
	mkl->set_epsilon(svm_epsilon);
	mkl->set_max_train_time(max_train_time);
	mkl->set_tube_epsilon(svm_tube_epsilon);
	mkl->set_nu(svm_nu);
	mkl->set_C(svm_C1, svm_C2);
	mkl->set_qpsize(svm_qpsize);
	mkl->set_shrinking_enabled(svm_use_shrinking);
	mkl->set_linadd_enabled(svm_use_linadd);
	mkl->set_batch_computation_enabled(svm_use_batch_computation);
	mkl->set_mkl_epsilon(svm_weight_epsilon);
	mkl->set_mkl_norm(mkl_norm);
	mkl->set_elasticnet_lambda(ent_lambda);
	mkl->set_mkl_block_norm(mkl_block_norm);
	mkl->set_C_mkl(C_mkl);
	mkl->set_interleaved_optimization_enabled(mkl_use_interleaved);

	if (svm_do_auc_maximization)
	{
		CAUCKernel* auc_kernel = new CAUCKernel(10, kernel);
		CLabels* auc_labels= auc_kernel->setup_auc_maximization(trainlabels);
		((CKernelMachine*) mkl)->set_labels(auc_labels);
		((CKernelMachine*) mkl)->set_kernel(auc_kernel);
		SG_UNREF(auc_labels);
	}
	else
	{
		if(!oneclass)
			((CKernelMachine*) mkl)->set_labels(trainlabels);
		((CKernelMachine*) mkl)->set_kernel(kernel);
	}

	bool result=mkl->train();

	return result;
}

bool CGUIClassifier::train_svm()
{
	EMachineType type = classifier->get_classifier_type();

	if (!classifier)
		SG_ERROR("No SVM available.\n")

	bool oneclass=(type==CT_LIBSVMONECLASS);
	CLabels* trainlabels=NULL;
	if(!oneclass)
		trainlabels=ui->ui_labels->get_train_labels();
	else
		SG_INFO("Training one class svm.\n")
	if (!trainlabels && !oneclass)
		SG_ERROR("No trainlabels available.\n")

	CKernel* kernel=ui->ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	bool success=ui->ui_kernel->init_kernel("TRAIN");

	if (!success || !ui->ui_kernel->is_initialized() || !kernel->has_features())
		SG_ERROR("Kernel not initialized / no train features available.\n")

	int32_t num_vec=kernel->get_num_vec_lhs();
	if (!oneclass && trainlabels->get_num_labels() != num_vec)
		SG_ERROR("Number of train labels (%d) and training vectors (%d) differs!\n", trainlabels->get_num_labels(), num_vec)

	SG_INFO("Starting SVM training on %ld vectors using C1=%lf C2=%lf epsilon=%lf\n", num_vec, svm_C1, svm_C2, svm_epsilon)

	if (type==CT_LARANK || type==CT_GMNPSVM || type==CT_LIBSVMMULTICLASS)
	{
		CMulticlassSVM* svm = (CMulticlassSVM*)classifier;
		svm->set_solver_type(solver_type);
		svm->set_bias_enabled(svm_use_bias);
		svm->set_epsilon(svm_epsilon);
		svm->set_max_train_time(max_train_time);
		svm->set_tube_epsilon(svm_tube_epsilon);
		svm->set_nu(svm_nu);
		svm->set_C(svm_C1);
		svm->set_qpsize(svm_qpsize);
		svm->set_shrinking_enabled(svm_use_shrinking);
		svm->set_linadd_enabled(svm_use_linadd);
		svm->set_batch_computation_enabled(svm_use_batch_computation);
	}
	else
	{
		CSVM* svm = (CSVM*)classifier;
		svm->set_solver_type(solver_type);
		svm->set_bias_enabled(svm_use_bias);
		svm->set_epsilon(svm_epsilon);
		svm->set_max_train_time(max_train_time);
		svm->set_tube_epsilon(svm_tube_epsilon);
		svm->set_nu(svm_nu);
		svm->set_C(svm_C1, svm_C2);
		svm->set_qpsize(svm_qpsize);
		svm->set_shrinking_enabled(svm_use_shrinking);
		svm->set_linadd_enabled(svm_use_linadd);
		svm->set_batch_computation_enabled(svm_use_batch_computation);
	}

	if (type==CT_MKLMULTICLASS)
	{
		((CMKLMulticlass *)classifier)->set_mkl_epsilon(svm_weight_epsilon);
	}

	if (svm_do_auc_maximization)
	{
		CAUCKernel* auc_kernel = new CAUCKernel(10, kernel);
		CLabels* auc_labels = auc_kernel->setup_auc_maximization(trainlabels);
		((CKernelMachine*)classifier)->set_labels(auc_labels);
		((CKernelMachine*)classifier)->set_kernel(auc_kernel);
		SG_UNREF(auc_labels);
	}
	else
	{
		if (type==CT_LARANK || type==CT_GMNPSVM || type==CT_LIBSVMMULTICLASS)
		{
			((CKernelMulticlassMachine*)classifier)->set_labels(trainlabels);
			((CKernelMulticlassMachine*)classifier)->set_kernel(kernel);
		}
		else
		{
			if(!oneclass)
				((CKernelMachine*)classifier)->set_labels(trainlabels);

			((CKernelMachine*)classifier)->set_kernel(kernel);
		}
	}

	bool result = classifier->train();

	return result;
}

bool CGUIClassifier::train_clustering(int32_t k, int32_t max_iter)
{
	bool result=false;
	CDistance* distance=ui->ui_distance->get_distance();

	if (!distance)
		SG_ERROR("No distance available\n")

	if (!ui->ui_distance->init_distance("TRAIN"))
		SG_ERROR("Initializing distance with train features failed.\n")

	((CDistanceMachine*) classifier)->set_distance(distance);

	EMachineType type=classifier->get_classifier_type();
	switch (type)
	{
		case CT_KMEANS:
		{
			((CKMeans*) classifier)->set_k(k);
			((CKMeans*) classifier)->set_max_iter(max_iter);
			result=((CKMeans*) classifier)->train();
			break;
		}
		case CT_HIERARCHICAL:
		{
			((CHierarchical*) classifier)->set_merges(k);
			result=((CHierarchical*) classifier)->train();
			break;
		}
		default:
			SG_ERROR("Unknown clustering type %d\n", type)
	}

	return result;
}

bool CGUIClassifier::train_knn(int32_t k)
{
	CLabels* trainlabels=ui->ui_labels->get_train_labels();
	CDistance* distance=ui->ui_distance->get_distance();

	bool result=false;

	if (trainlabels)
	{
		if (distance)
		{
			if (!ui->ui_distance->init_distance("TRAIN"))
				SG_ERROR("Initializing distance with train features failed.\n")
			((CKNN*) classifier)->set_labels(trainlabels);
			((CKNN*) classifier)->set_distance(distance);
			((CKNN*) classifier)->set_k(k);
			result=((CKNN*) classifier)->train();
		}
		else
			SG_ERROR("No distance available.\n")
	}
	else
		SG_ERROR("No labels available\n")

	return result;
}

bool CGUIClassifier::train_krr()
{
#ifdef HAVE_LAPACK
	CKernelRidgeRegression* krr= (CKernelRidgeRegression*) classifier;
	if (!krr)
		SG_ERROR("No SVM available.\n")

	CLabels* trainlabels=NULL;
	trainlabels=ui->ui_labels->get_train_labels();
	if (!trainlabels)
		SG_ERROR("No trainlabels available.\n")

	CKernel* kernel=ui->ui_kernel->get_kernel();
	if (!kernel)
		SG_ERROR("No kernel available.\n")

	bool success=ui->ui_kernel->init_kernel("TRAIN");

	if (!success || !ui->ui_kernel->is_initialized() || !kernel->has_features())
		SG_ERROR("Kernel not initialized / no train features available.\n")

	int32_t num_vec=kernel->get_num_vec_lhs();
	if (trainlabels->get_num_labels() != num_vec)
		SG_ERROR("Number of train labels (%d) and training vectors (%d) differs!\n", trainlabels->get_num_labels(), num_vec)


	// Set training labels and kernel
	krr->set_labels(trainlabels);
	krr->set_kernel(kernel);

	bool result=krr->train();
	return result;
#else
	return false;
#endif
}

bool CGUIClassifier::train_linear(float64_t gamma)
{
	ASSERT(classifier)
	EMachineType ctype = classifier->get_classifier_type();
	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	CLabels* trainlabels=ui->ui_labels->get_train_labels();
	bool result=false;

	if (!trainfeatures)
		SG_ERROR("No trainfeatures available.\n")

	if (!trainfeatures->has_property(FP_DOT))
		SG_ERROR("Trainfeatures not based on DotFeatures.\n")

	if (!trainlabels)
		SG_ERROR("No labels available\n")

	if (ctype==CT_PERCEPTRON)
	{
		((CPerceptron*) classifier)->set_learn_rate(perceptron_learnrate);
		((CPerceptron*) classifier)->set_max_iter(perceptron_maxiter);
	}

#ifdef HAVE_EIGEN3
	if (ctype==CT_LDA)
	{
		if (trainfeatures->get_feature_type()!=F_DREAL ||
				trainfeatures->get_feature_class()!=C_DENSE)
		SG_ERROR("LDA requires train features of class SIMPLE type REAL.\n")
		((CLDA*) classifier)->set_gamma(gamma);
	}
#endif //HAVE_EIGEN3

	if (ctype==CT_SVMOCAS)
		((CSVMOcas*) classifier)->set_C(svm_C1, svm_C2);
#ifdef HAVE_LAPACK
	else if (ctype==CT_LIBLINEAR)
		((CLibLinear*) classifier)->set_C(svm_C1, svm_C2);
#endif
	else if (ctype==CT_SVMLIN)
		((CSVMLin*) classifier)->set_C(svm_C1, svm_C2);
	else if (ctype==CT_SVMSGD)
		((CSVMSGD*) classifier)->set_C(svm_C1, svm_C2);
	else if (ctype==CT_LPM || ctype==CT_LPBOOST)
	{
		if (trainfeatures->get_feature_class()!=C_SPARSE ||
				trainfeatures->get_feature_type()!=F_DREAL)
			SG_ERROR("LPM and LPBOOST require trainfeatures of class SPARSE type REAL.\n")
	}

	((CLinearMachine*) classifier)->set_labels(trainlabels);
	((CLinearMachine*) classifier)->set_features((CDenseFeatures<float64_t>*) trainfeatures);
	result=((CLinearMachine*) classifier)->train();

	return result;
}

bool CGUIClassifier::train_wdocas()
{
	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	CLabels* trainlabels=ui->ui_labels->get_train_labels();

	bool result=false;

	if (!trainfeatures)
		SG_ERROR("No trainfeatures available.\n")

	if (trainfeatures->get_feature_class()!=C_STRING ||
			trainfeatures->get_feature_type()!=F_BYTE )
		SG_ERROR("Trainfeatures are not of class STRING type BYTE.\n")

	if (!trainlabels)
		SG_ERROR("No labels available.\n")

	((CWDSVMOcas*) classifier)->set_labels(trainlabels);
	((CWDSVMOcas*) classifier)->set_features((CStringFeatures<uint8_t>*) trainfeatures);
	result=((CWDSVMOcas*) classifier)->train();

	return result;
}

bool CGUIClassifier::load(char* filename, char* type)
{
	bool result=false;

	if (new_classifier(type))
	{
		FILE* model_file=fopen(filename, "r");
		CSerializableAsciiFile* ascii_file = new CSerializableAsciiFile(model_file,'r');

		if (ascii_file)
		{
			if (classifier && classifier->load_serializable(ascii_file))
			{
				SG_DEBUG("file successfully read.\n")
				result=true;
			}
			else
				SG_ERROR("SVM/Classifier creation/loading failed on file %s.\n", filename)

			delete ascii_file;
		}
		else
			SG_ERROR("Opening file %s failed.\n", filename)

		return result;
	}
	else
		SG_ERROR("Type %s of SVM/Classifier unknown.\n", type)

	return false;
}

bool CGUIClassifier::save(char* param)
{
	bool result=false;
	param=SGIO::skip_spaces(param);

	if (classifier)
	{
		FILE* file=fopen(param, "w");
		CSerializableAsciiFile* ascii_file = new CSerializableAsciiFile(file,'w');

		if ((!ascii_file) || (!classifier->save_serializable(ascii_file)))
			printf("writing to file %s failed!\n", param);
		else
		{
			printf("successfully written classifier into \"%s\" !\n", param);
			result=true;
		}

		if (ascii_file)
			delete ascii_file;
	}
	else
		SG_ERROR("create classifier first\n")

	return result;
}

bool CGUIClassifier::set_perceptron_parameters(
	float64_t learnrate, int32_t maxiter)
{
	if (learnrate<=0)
		perceptron_learnrate=0.01;
	else
		perceptron_learnrate=learnrate;

	if (maxiter<=0)
		perceptron_maxiter=1000;
	else
		perceptron_maxiter=maxiter;
	SG_INFO("Setting to perceptron parameters (learnrate %f and maxiter: %d\n", perceptron_learnrate, perceptron_maxiter)

	return true;
}

bool CGUIClassifier::set_svm_epsilon(float64_t epsilon)
{
	if (epsilon<0)
		svm_epsilon=1e-4;
	else
		svm_epsilon=epsilon;
	SG_INFO("Set to svm_epsilon=%f.\n", svm_epsilon)

	return true;
}

bool CGUIClassifier::set_max_train_time(float64_t max)
{
	if (max>0)
	{
		max_train_time=max;
		SG_INFO("Set to max_train_time=%f.\n", max_train_time)
	}
	else
		SG_INFO("Disabling max_train_time.\n")

	return true;
}

bool CGUIClassifier::set_svr_tube_epsilon(float64_t tube_epsilon)
{
	if (!classifier)
		SG_ERROR("No regression method allocated\n")

	if (classifier->get_classifier_type() != CT_LIBSVR &&
			classifier->get_classifier_type() != CT_SVRLIGHT &&
			classifier->get_classifier_type() != CT_MKLREGRESSION )
	{
		SG_ERROR("Underlying method not capable of SV-regression\n")
	}

	if (tube_epsilon<0)
		svm_tube_epsilon=1e-2;
	svm_tube_epsilon=tube_epsilon;

	((CSVM*) classifier)->set_tube_epsilon(svm_tube_epsilon);
	SG_INFO("Set to svr_tube_epsilon=%f.\n", svm_tube_epsilon)

	return true;
}

bool CGUIClassifier::set_svm_nu(float64_t nu)
{
	if (nu<0 || nu>1)
		nu=0.5;

	svm_nu=nu;
	SG_INFO("Set to nu=%f.\n", svm_nu)

	return true;
}

bool CGUIClassifier::set_svm_mkl_parameters(
	float64_t weight_epsilon, float64_t C, float64_t norm)
{
	if (weight_epsilon<0)
		weight_epsilon=1e-4;
	if (C<0)
		C=0;
	if (norm<0)
		SG_ERROR("MKL norm >= 0\n")

	svm_weight_epsilon=weight_epsilon;
	C_mkl=C;
	mkl_norm=norm;

	SG_INFO("Set to weight_epsilon=%f.\n", svm_weight_epsilon)
	SG_INFO("Set to C_mkl=%f.\n", C_mkl)
	SG_INFO("Set to mkl_norm=%f.\n", mkl_norm)

	return true;
}

bool CGUIClassifier::set_elasticnet_lambda(float64_t lambda)
{
  if (lambda<0 || lambda>1)
    SG_ERROR("0 <= ent_lambda <= 1\n")

  ent_lambda = lambda;
  return true;
}

bool CGUIClassifier::set_mkl_block_norm(float64_t mkl_bnorm)
{
  if (mkl_bnorm<1)
    SG_ERROR("1 <= mkl_block_norm <= inf\n")

  mkl_block_norm=mkl_bnorm;
  return true;
}


bool CGUIClassifier::set_svm_C(float64_t C1, float64_t C2)
{
	if (C1<0)
		svm_C1=1.0;
	else
		svm_C1=C1;

	if (C2<0)
		svm_C2=svm_C1;
	else
		svm_C2=C2;

	SG_INFO("Set to C1=%f C2=%f.\n", svm_C1, svm_C2)

	return true;
}

bool CGUIClassifier::set_svm_qpsize(int32_t qpsize)
{
	if (qpsize<2)
		svm_qpsize=41;
	else
		svm_qpsize=qpsize;
	SG_INFO("Set qpsize to svm_qpsize=%d.\n", svm_qpsize)

	return true;
}

bool CGUIClassifier::set_svm_max_qpsize(int32_t max_qpsize)
{
	if (max_qpsize<50)
		svm_max_qpsize=50;
	else
		svm_max_qpsize=max_qpsize;
	SG_INFO("Set max qpsize to svm_max_qpsize=%d.\n", svm_max_qpsize)

	return true;
}

bool CGUIClassifier::set_svm_bufsize(int32_t bufsize)
{
	if (svm_bufsize<0)
		svm_bufsize=3000;
	else
		svm_bufsize=bufsize;
	SG_INFO("Set bufsize to svm_bufsize=%d.\n", svm_bufsize)

	return true ;
}

bool CGUIClassifier::set_svm_shrinking_enabled(bool enabled)
{
	svm_use_shrinking=enabled;
	if (svm_use_shrinking)
		SG_INFO("Enabling shrinking optimization.\n")
	else
		SG_INFO("Disabling shrinking optimization.\n")

	return true;
}

bool CGUIClassifier::set_svm_batch_computation_enabled(bool enabled)
{
	svm_use_batch_computation=enabled;
	if (svm_use_batch_computation)
		SG_INFO("Enabling batch computation.\n")
	else
		SG_INFO("Disabling batch computation.\n")

	return true;
}

bool CGUIClassifier::set_svm_linadd_enabled(bool enabled)
{
	svm_use_linadd=enabled;
	if (svm_use_linadd)
		SG_INFO("Enabling LINADD optimization.\n")
	else
		SG_INFO("Disabling LINADD optimization.\n")

	return true;
}

bool CGUIClassifier::set_svm_bias_enabled(bool enabled)
{
	svm_use_bias=enabled;
	if (svm_use_bias)
		SG_INFO("Enabling svm bias.\n")
	else
		SG_INFO("Disabling svm bias.\n")

	return true;
}

bool CGUIClassifier::set_mkl_interleaved_enabled(bool enabled)
{
	mkl_use_interleaved=enabled;
	if (mkl_use_interleaved)
		SG_INFO("Enabling mkl interleaved optimization.\n")
	else
		SG_INFO("Disabling mkl interleaved optimization.\n")

	return true;
}

bool CGUIClassifier::set_do_auc_maximization(bool do_auc)
{
	svm_do_auc_maximization=do_auc;

	if (svm_do_auc_maximization)
		SG_INFO("Enabling AUC maximization.\n")
	else
		SG_INFO("Disabling AUC maximization.\n")

	return true;
}


CLabels* CGUIClassifier::classify()
{
	ASSERT(classifier)

	switch (classifier->get_classifier_type())
	{
		case CT_LIGHT:
		case CT_LIGHTONECLASS:
		case CT_LIBSVM:
		case CT_SCATTERSVM:
		case CT_MPD:
		case CT_GPBT:
		case CT_CPLEXSVM:
		case CT_GMNPSVM:
		case CT_GNPPSVM:
		case CT_LIBSVR:
		case CT_LIBSVMMULTICLASS:
		case CT_LIBSVMONECLASS:
		case CT_SVRLIGHT:
		case CT_MKLCLASSIFICATION:
		case CT_MKLMULTICLASS:
		case CT_MKLREGRESSION:
		case CT_MKLONECLASS:
		case CT_KERNELRIDGEREGRESSION:
			return classify_kernelmachine();
		case CT_KNN:
			return classify_distancemachine();
		case CT_PERCEPTRON:
		case CT_LDA:
			return classify_linear();
		case CT_SVMLIN:
		case CT_SVMPERF:
		case CT_SVMOCAS:
		case CT_SVMSGD:
		case CT_LPM:
		case CT_LPBOOST:
		case CT_LIBLINEAR:
			return classify_linear();
		case CT_WDSVMOCAS:
			return classify_byte_linear();
		default:
			SG_ERROR("unknown classifier type\n")
			break;
	};

	return NULL;
}

CLabels* CGUIClassifier::classify_kernelmachine()
{
	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!classifier)
		SG_ERROR("No kernelmachine available.\n")

	bool success=true;

	REQUIRE(ui->ui_kernel->get_kernel(), "No kernel set");
	if (ui->ui_kernel->get_kernel()->get_kernel_type()!=K_CUSTOM)
	{
		if (ui->ui_kernel->get_kernel()->get_kernel_type()==K_COMBINED
				&& ( !trainfeatures || !testfeatures ))
		{
			SG_DEBUG("skipping initialisation of combined kernel "
					"as train/test features are unavailable\n")
		}
		else
		{
			if (!trainfeatures)
				SG_ERROR("No training features available.\n")
			if (!testfeatures)
				SG_ERROR("No test features available.\n")

			success=ui->ui_kernel->init_kernel("TEST");
		}
	}

	if (!success || !ui->ui_kernel->is_initialized())
		SG_ERROR("Kernel not initialized.\n")

	EMachineType type = classifier->get_classifier_type();
	if (type==CT_LARANK || type==CT_GMNPSVM || type==CT_LIBSVMMULTICLASS ||
		type==CT_MKLMULTICLASS)
	{
		CKernelMulticlassMachine* kmcm = (CKernelMulticlassMachine*) classifier;
		kmcm->set_kernel(ui->ui_kernel->get_kernel());
	}
	else
	{
		CKernelMachine* km=(CKernelMachine*) classifier;
		km->set_kernel(ui->ui_kernel->get_kernel());
		km->set_batch_computation_enabled(svm_use_batch_computation);
	}

	SG_INFO("Starting kernel machine testing.\n")
	return classifier->apply();
}

bool CGUIClassifier::get_trained_classifier(
	float64_t* &weights, int32_t &rows, int32_t &cols, float64_t*& bias,
	int32_t& brows, int32_t& bcols,
	int32_t idx) // which SVM for Multiclass
{
	ASSERT(classifier)

	switch (classifier->get_classifier_type())
	{
		case CT_SCATTERSVM:
		case CT_GNPPSVM:
		case CT_LIBSVMMULTICLASS:
		case CT_LIGHT:
		case CT_LIGHTONECLASS:
		case CT_LIBSVM:
		case CT_MPD:
		case CT_GPBT:
		case CT_CPLEXSVM:
		case CT_GMNPSVM:
		case CT_LIBSVR:
		case CT_LIBSVMONECLASS:
		case CT_SVRLIGHT:
		case CT_MKLCLASSIFICATION:
		case CT_MKLREGRESSION:
		case CT_MKLONECLASS:
		case CT_MKLMULTICLASS:
		case CT_KERNELRIDGEREGRESSION:
			return get_svm(weights, rows, cols, bias, brows, bcols, idx);
			break;
		case CT_PERCEPTRON:
		case CT_LDA:
		case CT_LPM:
		case CT_LPBOOST:
		case CT_SVMOCAS:
		case CT_SVMSGD:
		case CT_SVMLIN:
		case CT_SVMPERF:
		case CT_LIBLINEAR:
			return get_linear(weights, rows, cols, bias, brows, bcols);
			break;
		case CT_KMEANS:
		case CT_HIERARCHICAL:
			return get_clustering(weights, rows, cols, bias, brows, bcols);
			break;
		case CT_KNN:
			SG_ERROR("not implemented")
			break;
		default:
			SG_ERROR("unknown classifier type\n")
			break;
	};
	return false;
}


int32_t CGUIClassifier::get_num_svms()
{
	ASSERT(classifier)
	return ((CMulticlassSVM*) classifier)->get_num_machines();
}

bool CGUIClassifier::get_svm(
	float64_t* &weights, int32_t& rows, int32_t& cols, float64_t*& bias,
	int32_t& brows, int32_t& bcols, int32_t idx)
{
	CSVM* svm=(CSVM*) classifier;

	if (idx>-1) // should be MulticlassSVM
		svm=((CMulticlassSVM*) svm)->get_svm(idx);

	if (svm)
	{
		brows=1;
		bcols=1;
		bias=SG_MALLOC(float64_t, 1);
		*bias=svm->get_bias();

		rows=svm->get_num_support_vectors();
		cols=2;
		weights=SG_MALLOC(float64_t, rows*cols);

		for (int32_t i=0; i<rows; i++)
		{
			weights[i]=svm->get_alpha(i);
			weights[i+rows]=svm->get_support_vector(i);
		}

		return true;
	}

	return false;
}

bool CGUIClassifier::get_clustering(
	float64_t* &centers, int32_t& rows, int32_t& cols, float64_t*& radi,
	int32_t& brows, int32_t& bcols)
{
	if (!classifier)
		return false;

	switch (classifier->get_classifier_type())
	{
		case CT_KMEANS:
		{
			CKMeans* clustering=(CKMeans*) classifier;

			bcols=1;
			SGVector<float64_t> r=clustering->get_radiuses();
			brows=r.vlen;
			radi=SG_MALLOC(float64_t, brows);
			memcpy(radi, r.vector, sizeof(float64_t)*brows);

			cols=1;
			SGMatrix<float64_t> c=clustering->get_cluster_centers();
			rows=c.num_rows;
			cols=c.num_cols;
			centers=SG_MALLOC(float64_t, rows*cols);
			memcpy(centers, c.matrix, sizeof(float64_t)*rows*cols);
			break;
		}

		case CT_HIERARCHICAL:
		{
			CHierarchical* clustering=(CHierarchical*) classifier;

			// radi == merge_distances, centers == pairs
			bcols=1;
			SGVector<float64_t> r=clustering->get_merge_distances();
			brows=r.vlen;
			radi=SG_MALLOC(float64_t, brows);
			memcpy(radi, r.vector, sizeof(float64_t)*brows);

			SGMatrix<int32_t> p=clustering->get_cluster_pairs();
			rows=p.num_rows;
			cols=p.num_cols;
			centers=SG_MALLOC(float64_t, rows*cols);
			for (int32_t i=0; i<rows*cols; i++)
				centers[i]=(float64_t) p.matrix[i];

			break;
		}

		default:
			SG_ERROR("internal error - unknown clustering type\n")
	}

	return true;
}

bool CGUIClassifier::get_linear(
	float64_t* &weights, int32_t& rows, int32_t& cols, float64_t*& bias,
	int32_t& brows, int32_t& bcols)
{
	CLinearMachine* linear=(CLinearMachine*) classifier;

	if (!linear)
		return false;

	bias=SG_MALLOC(float64_t, 1);
	*bias=linear->get_bias();
	brows=1;
	bcols=1;

	SGVector<float64_t> w=linear->get_w();
	cols=1;
	rows=w.vlen;

	weights= SG_MALLOC(float64_t, w.vlen);
	memcpy(weights, w.vector, sizeof(float64_t)*w.vlen);

	return true;
}

CLabels* CGUIClassifier::classify_distancemachine()
{
	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!classifier)
	{
		SG_ERROR("no kernelmachine available\n")
		return NULL;
	}
	if (!trainfeatures)
	{
		SG_ERROR("no training features available\n")
		return NULL;
	}

	if (!testfeatures)
	{
		SG_ERROR("no test features available\n")
		return NULL;
	}

	bool success=ui->ui_distance->init_distance("TEST");

	if (!success || !ui->ui_distance->is_initialized())
	{
		SG_ERROR("distance not initialized\n")
		return NULL;
	}

	((CDistanceMachine*) classifier)->set_distance(
		ui->ui_distance->get_distance());
	SG_INFO("starting distance machine testing\n")
	return classifier->apply();
}


CLabels* CGUIClassifier::classify_linear()
{
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!classifier)
	{
		SG_ERROR("no classifier available\n")
		return NULL;
	}
	if (!testfeatures)
	{
		SG_ERROR("no test features available\n")
		return NULL;
	}
	if (!(testfeatures->has_property(FP_DOT)))
	{
		SG_ERROR("testfeatures not based on DotFeatures\n")
		return NULL;
	}

	((CLinearMachine*) classifier)->set_features((CDotFeatures*) testfeatures);
	SG_INFO("starting linear classifier testing\n")
	return classifier->apply();
}

CLabels* CGUIClassifier::classify_byte_linear()
{
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!classifier)
	{
		SG_ERROR("no svm available\n")
		return NULL;
	}
	if (!testfeatures)
	{
		SG_ERROR("no test features available\n")
		return NULL;
	}
	if (testfeatures->get_feature_class() != C_STRING ||
			testfeatures->get_feature_type() != F_BYTE )
	{
		SG_ERROR("testfeatures not of class STRING type BYTE\n")
		return NULL;
	}

	((CWDSVMOcas*) classifier)->set_features((CStringFeatures<uint8_t>*) testfeatures);
	SG_INFO("starting linear classifier testing\n")
	return classifier->apply();
}

bool CGUIClassifier::classify_example(int32_t idx, float64_t &result)
{
	CFeatures* trainfeatures=ui->ui_features->get_train_features();
	CFeatures* testfeatures=ui->ui_features->get_test_features();

	if (!classifier)
	{
		SG_ERROR("no svm available\n")
		return false;
	}

	if (!ui->ui_kernel->is_initialized())
	{
		SG_ERROR("kernel not initialized\n")
		return false;
	}

	if (!ui->ui_kernel->get_kernel() ||
		ui->ui_kernel->get_kernel()->get_kernel_type()!=K_CUSTOM)
	{
		if (!trainfeatures)
		{
			SG_ERROR("no training features available\n")
			return false;
		}

		if (!testfeatures)
		{
			SG_ERROR("no test features available\n")
			return false;
		}
	}

	((CKernelMachine*) classifier)->set_kernel(
		ui->ui_kernel->get_kernel());

	result=((CKernelMachine*)classifier)->apply_one(idx);
	return true ;
}


bool CGUIClassifier::set_krr_tau(float64_t tau)
{
#ifdef HAVE_LAPACK
	krr_tau=tau;
	((CKernelRidgeRegression*) classifier)->set_tau(krr_tau);
	SG_INFO("Set to krr_tau=%f.\n", krr_tau)

	return true;
#else
	return false;
#endif
}

bool CGUIClassifier::set_solver(char* solver)
{
	ESolverType s=ST_AUTO;

	if (strncmp(solver,"NEWTON", 6)==0)
	{
		SG_INFO("Using NEWTON solver.\n")
		s=ST_NEWTON;
	}
	else if (strncmp(solver,"DIRECT", 6)==0)
	{
		SG_INFO("Using DIRECT solver\n")
		s=ST_DIRECT;
	}
	else if (strncmp(solver,"BLOCK_NORM", 9)==0)
	{
		SG_INFO("Using BLOCK_NORM solver\n")
		s=ST_BLOCK_NORM;
	}
	else if (strncmp(solver,"ELASTICNET", 10)==0)
	{
		SG_INFO("Using ELASTICNET solver\n")
		s=ST_ELASTICNET;
	}
	else if (strncmp(solver,"AUTO", 4)==0)
	{
		SG_INFO("Automagically determining solver.\n")
		s=ST_AUTO;
	}
#ifdef USE_CPLEX
	else if (strncmp(solver, "CPLEX", 5)==0)
	{
		SG_INFO("USING CPLEX METHOD selected\n")
		s=ST_CPLEX;
	}
#endif
#ifdef USE_GLPK
	else if (strncmp(solver,"GLPK", 4)==0)
	{
		SG_INFO("Using GLPK solver\n")
		s=ST_GLPK;
	}
#endif
	else
		SG_ERROR("Unknown solver type, %s (not compiled in?)\n", solver)


	solver_type=s;
	return true;
}

bool CGUIClassifier::set_constraint_generator(char* name)
{
	if (strcmp(name,"LIBSVM_ONECLASS")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator = new CLibSVMOneClass();
		SG_INFO("created SVMlibsvm object for oneclass\n")
	}
	else if (strcmp(name,"LIBSVM_NU")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CLibSVM(LIBSVM_NU_SVC);
		SG_INFO("created SVMlibsvm object\n")
	}
	else if (strcmp(name,"LIBSVM")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CLibSVM();
		SG_INFO("created SVMlibsvm object\n")
	}
#ifdef USE_SVMLIGHT
	else if ((strcmp(name,"LIGHT")==0) || (strcmp(name,"SVMLIGHT")==0))
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CSVMLight();
		SG_INFO("created SVMLight object\n")
	}
	else if (strcmp(name,"SVMLIGHT_ONECLASS")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CSVMLightOneClass();
		SG_INFO("created SVMLightOneClass object\n")
	}
	else if (strcmp(name,"SVRLIGHT")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CSVRLight();
		SG_INFO("created SVRLight object\n")
	}
#endif //USE_SVMLIGHT
	else if (strcmp(name,"GPBTSVM")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CGPBTSVM();
		SG_INFO("created GPBT-SVM object\n")
	}
	else if (strcmp(name,"MPDSVM")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CMPDSVM();
		SG_INFO("created MPD-SVM object\n")
	}
	else if (strcmp(name,"GNPPSVM")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CGNPPSVM();
		SG_INFO("created GNPP-SVM object\n")
	}
	else if (strcmp(name,"LIBSVR")==0)
	{
		SG_UNREF(constraint_generator);
		constraint_generator= new CLibSVR();
		SG_INFO("created SVRlibsvm object\n")
	}
	else
	{
		SG_ERROR("Unknown SV-classifier %s.\n", name)
		return false;
	}
	SG_REF(constraint_generator);

	return (constraint_generator!=NULL);
}
