/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 */
#include <shogun/base/init.h>
#include <shogun/features/RandomFourierDotFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/io/LibSVMFile.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/evaluation/PRCEvaluation.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/lib/Time.h>

#include <stdio.h>

using namespace shogun;

const char* filepath = 0;
const char* testpath = 0;
int32_t D = 300;
float64_t C = 0.1;
float64_t epsilon = 0.01;
float64_t width = 8;
int32_t correct_dimension = -1;

SGSparseMatrix<float64_t> load_data(const char* filepath, float64_t*& label_vec)
{
	FILE* data_file = fopen(filepath, "r");
	SGSparseMatrix<float64_t> sparse_data;

	CLibSVMFile* file_reader = new CLibSVMFile(data_file);
	file_reader->get_sparse_matrix(sparse_data.sparse_matrix, sparse_data.num_features, sparse_data.num_vectors,
			label_vec);

	if (correct_dimension!=-1)
		sparse_data.num_features = correct_dimension;

	SG_UNREF(file_reader);

	return sparse_data;
}

void print_help_message()
{
	SG_SPRINT("Usage : ./rf_classify --dataset path_to_data [--testset path_to_test_data] [-D number_of_samples]\n");
	SG_SPRINT("		[-C C_for_SVM] [--epsilon SVM_epsilon] [--width gaussian_kernel_width] [--dimension feature_dimension]\n");
	SG_SPRINT("\nPerforms binary classification on provided data using Random Fourier features with a linear SVM solver,\n");
	SG_SPRINT("namely SVMOcas.\nParameter explanation :\n");
	SG_SPRINT("\ndataset  : Path to data in LibSVM format. Required.");
	SG_SPRINT("\ntestset  : Path to test data in LibSVM format. Optional.");
	SG_SPRINT("\nD         : Number of samples for the Random Fourier features. Default value = 300");
	SG_SPRINT("\nC         : SVM parameter C. Default value = 0.1");
	SG_SPRINT("\nepsilon   : SVM epsilon. Default value = 0.01");
	SG_SPRINT("\nwidth     : Gaussian Kernel width parameter. Default value = 8");
	SG_SPRINT("\ndimension : Correct feature dimension. Optional\n");
}

void parse_arguments(int argv, char** argc)
{
	if (argv%2!=1)
	{
		print_help_message();
		exit_shogun();
		exit(0);
	}

	for (index_t i=1; i<argv; i++)
	{
		if (strcmp(argc[i],"--dataset")==0)
			filepath = argc[++i];
		else if (strcmp(argc[i],"--testset")==0)
			testpath = argc[++i];
		else if (strcmp(argc[i],"-D")==0)
			D = atoi(argc[++i]);
		else if (strcmp(argc[i],"-C")==0)
			C = atof(argc[++i]);
		else if (strcmp(argc[i],"--epsilon")==0)
			epsilon = atof(argc[++i]);
		else if (strcmp(argc[i],"--width")==0)
			width = atof(argc[++i]);
		else if (strcmp(argc[i],"--dimension")==0)
			correct_dimension = atoi(argc[++i]);
	}

	if (filepath==0)
	{
		print_help_message();
		exit_shogun();
		exit(0);
	}
}

int main(int argv, char** argc)
{
	init_shogun_with_defaults();

	parse_arguments(argv, argc);

	/** Reading data */
	float64_t* label_vec = 0;
	SGSparseMatrix<float64_t> sparse_data = load_data(filepath, label_vec);
	SGVector<float64_t> label(label_vec, sparse_data.num_vectors);


	/** Creating features */
	CBinaryLabels* labels = new CBinaryLabels(label);
	SG_REF(labels);

	CSparseFeatures<float64_t>* s_feats = new CSparseFeatures<float64_t>(sparse_data);
	SGVector<float64_t> params(1);
	params[0] = width;
	CRandomFourierDotFeatures* r_feats = new CRandomFourierDotFeatures(
			s_feats, D, KernelName::GAUSSIAN, params);


	/** Training */
	CLibLinear* svm = new CLibLinear(C, r_feats, labels);
	//CSVMOcas* svm = new CSVMOcas(C, r_feats, labels);
	svm->set_epsilon(epsilon);
	SG_SPRINT("Starting training\n");
	CTime* timer = new CTime();
	svm->train();
	float64_t secs = timer->cur_runtime_diff_sec();
	timer->stop();
	SG_UNREF(timer);
	SG_SPRINT("Training completed, took %fs\n", secs);
	/** Training completed */

	/** Evaluating */
	CBinaryLabels* predicted = CLabelsFactory::to_binary(svm->apply());
	CPRCEvaluation* prc_evaluator = new CPRCEvaluation();
	CROCEvaluation* roc_evaluator = new CROCEvaluation();
	CAccuracyMeasure* accuracy_evaluator = new CAccuracyMeasure();

	float64_t auROC = roc_evaluator->evaluate(predicted, labels);
	float64_t auPRC = prc_evaluator->evaluate(predicted, labels);
	float32_t accuracy = accuracy_evaluator->evaluate(predicted, labels);
	SG_SPRINT("Training auPRC=%f, auROC=%f, accuracy=%f ( Incorrectly predicted=%f% )\n", auPRC, auROC,
				accuracy, (1-accuracy) * 100);

	SG_UNREF(predicted);
	SGMatrix<float64_t> w = r_feats->get_random_coefficients();
	svm->set_features(NULL);

	if (testpath!=0)
	{
		sparse_data = load_data(testpath, label_vec);
		label = SGVector<float64_t>(label_vec, sparse_data.num_vectors);

		s_feats = new CSparseFeatures<float64_t>(sparse_data);
		r_feats = new CRandomFourierDotFeatures(s_feats, D, KernelName::GAUSSIAN, width, w);
		CBinaryLabels* test_labels = new CBinaryLabels(label);

		predicted = CLabelsFactory::to_binary(svm->apply(r_feats));
		auROC = roc_evaluator->evaluate(predicted, test_labels);
		auPRC = prc_evaluator->evaluate(predicted, test_labels);
		accuracy = accuracy_evaluator->evaluate(predicted, test_labels);
		SG_SPRINT("Test auPRC=%f, auROC=%f, accuracy=%f ( Incorrectly predicted=%f% )\n", auPRC, auROC,
				accuracy, (1-accuracy) * 100);
		SG_UNREF(predicted);
		SG_UNREF(test_labels);

	}
	SG_UNREF(prc_evaluator);
	SG_UNREF(roc_evaluator);
	SG_UNREF(accuracy_evaluator);
	SG_UNREF(svm);
	SG_UNREF(labels);
	exit_shogun();
}
