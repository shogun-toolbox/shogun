/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GUICLASSIFIER_H__
#define _GUICLASSIFIER_H__ 

#include "lib/config.h"
#include "base/SGObject.h"

#ifndef HAVE_SWIG
#include "classifier/Classifier.h"

class CGUI ;

class CGUIClassifier : public CSGObject
{

public:
	CGUIClassifier(CGUI*);
	~CGUIClassifier();

	bool new_classifier(CHAR* param);
	bool train(CHAR* param);
	bool test(CHAR* param);
	bool load(CHAR* param);
	bool save(CHAR* param);
	CLabels* classify(CLabels* output=NULL);
	CLabels* classify_kernelmachine(CLabels* output=NULL);
	CLabels* classify_distancemachine(CLabels* output=NULL);
	CLabels* classify_linear(CLabels* output=NULL);
	bool classify_example(INT idx, DREAL& result);
	inline CClassifier* get_classifier() { return classifier; }

	/// perceptron learnrate maxiter
	bool set_perceptron_parameters(CHAR* param);

	/// SVM functions
	bool set_svm_C(CHAR* param);
	bool set_svm_qpsize(CHAR* param);
	bool set_svm_mkl_enabled(CHAR* param);
	bool set_svm_linadd_enabled(CHAR* param);
	bool set_svm_epsilon(CHAR* param);
	bool set_svr_tube_epsilon(CHAR* param);
	bool set_svm_mkl_parameters(CHAR* param) ;
	bool set_svm_precompute_enabled(CHAR* param) ;
	bool train_svm(CHAR* param, bool auc_maximization);
	bool train_knn(CHAR* param);
	bool train_linear(CHAR* param);


 protected:
	CGUI* gui;
	CClassifier* classifier;

	double perceptron_learnrate;
	int perceptron_maxiter;

	int svm_qpsize;
	double svm_weight_epsilon;
	double svm_epsilon;
	double svm_tube_epsilon;
	double svm_C1;
	double svm_C2;
	double svm_C_mkl;
	bool svm_use_mkl;
	bool svm_use_linadd;
	bool svm_use_precompute;
	bool svm_use_precompute_subkernel;
	bool svm_use_precompute_subkernel_light;
};
#endif //HAVE_SWIG
#endif
