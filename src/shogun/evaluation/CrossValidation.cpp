/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include <shogun/machine/KernelMachine.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/machine/Machine.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/evaluation/CrossValidationOutput.h>
#include <shogun/lib/List.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/kernel/GaussianKernel.h>

using namespace shogun;

CCrossValidation::CCrossValidation() : CMachineEvaluation()
{
	init();
}

CCrossValidation::CCrossValidation(CMachine* machine, CFeatures* features,
		CLabels* labels, CSplittingStrategy* splitting_strategy,
		CEvaluation* evaluation_criterion, bool autolock) :
		CMachineEvaluation(machine, features, labels, splitting_strategy,
		evaluation_criterion, autolock)
{
	init();
}

CCrossValidation::CCrossValidation(CMachine* machine, CLabels* labels,
		CSplittingStrategy* splitting_strategy,
		CEvaluation* evaluation_criterion, bool autolock) :
		CMachineEvaluation(machine, labels, splitting_strategy, evaluation_criterion,
		autolock)
{
	init();
}

CCrossValidation::~CCrossValidation()
{
	SG_UNREF(m_xval_outputs);
}

void CCrossValidation::init()
{
	m_num_runs=1;

	/* do reference counting for output objects */
	m_xval_outputs=new CList(true);

	SG_ADD(&m_num_runs, "num_runs", "Number of repetitions",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_xval_outputs, "m_xval_outputs", "List of output "
			"classes for intermediade cross-validation results",
			MS_NOT_AVAILABLE);
}

CEvaluationResult* CCrossValidation::evaluate()
{
	SG_DEBUG("entering %s::evaluate()\n", get_name())

	REQUIRE(m_machine, "%s::evaluate() is only possible if a machine is "
			"attached\n", get_name());

	REQUIRE(m_features, "%s::evaluate() is only possible if features are "
			"attached\n", get_name());

	REQUIRE(m_labels, "%s::evaluate() is only possible if labels are "
			"attached\n", get_name());

	/* if for some reason the do_unlock_frag is set, unlock */
	if (m_do_unlock)
	{
		m_machine->data_unlock();
		m_do_unlock=false;
	}

	/* set labels in any case (no locking needs this) */
	//m_machine->set_labels(m_labels);
	int32_t lo=0;
	if (m_autolock)
	{
		/* if machine supports locking try to do so */
		if (m_machine->supports_locking() && lo)
		{
			/* only lock if machine is not yet locked */
			if (!m_machine->is_data_locked())
			{
				m_machine->data_lock(m_labels, m_features);
				m_do_unlock=true;
			}
		}
		else
		{
			SG_WARNING("%s does not support locking. Autolocking is skipped. "
					"Set autolock flag to false to get rid of warning.\n",
					m_machine->get_name());
		}
	}

	SGVector<float64_t> results(m_num_runs);

	/* evtl. update xvalidation output class */
	CCrossValidationOutput* current=(CCrossValidationOutput*)
			m_xval_outputs->get_first_element();
	while (current)
	{
		current->init_num_runs(m_num_runs);
		current->init_num_folds(m_splitting_strategy->get_num_subsets());
		current->init_expose_labels(m_labels);
		current->post_init();
		SG_UNREF(current);
		current=(CCrossValidationOutput*)
				m_xval_outputs->get_next_element();
	}

	/* perform all the x-val runs */
	SG_DEBUG("starting %d runs of cross-validation\n", m_num_runs)
	for (index_t i=0; i <m_num_runs; ++i)
	{

		/* evtl. update xvalidation output class */
		current=(CCrossValidationOutput*)m_xval_outputs->get_first_element();
		while (current)
		{
			current->update_run_index(i);
			SG_UNREF(current);
			current=(CCrossValidationOutput*)
					m_xval_outputs->get_next_element();
		}

		SG_DEBUG("entering cross-validation run %d \n", i)
		results[i]=evaluate_one_run();
		SG_DEBUG("result of cross-validation run %d is %f\n", i, results[i])
	}

	/* construct evaluation result */
	CCrossValidationResult* result = new CCrossValidationResult();
	result->mean=CStatistics::mean(results);
	if (m_num_runs>1)
		result->std_dev=CStatistics::std_deviation(results);
	else
		result->std_dev=0;

	/* unlock machine if it was locked in this method */
	if (m_machine->is_data_locked() && m_do_unlock)
	{
		m_machine->data_unlock();
		m_do_unlock=false;
	}

	SG_DEBUG("leaving %s::evaluate()\n", get_name())

	SG_REF(result);
	return result;
}

void CCrossValidation::set_num_runs(int32_t num_runs)
{
	if (num_runs <1)
		SG_ERROR("%d is an illegal number of repetitions\n", num_runs)

	m_num_runs=num_runs;
}

float64_t CCrossValidation::evaluate_one_run()
{
	SG_DEBUG("entering %s::evaluate_one_run()\n", get_name())
	index_t num_subsets=m_splitting_strategy->get_num_subsets();

	SG_DEBUG("building index sets for %d-fold cross-validation\n", num_subsets)

	/* build index sets */
	m_splitting_strategy->build_subsets();

	/* results array */
	SGVector<float64_t> results(num_subsets);
	int32_t flag=0;

	/* different behavior whether data is locked or not */
	if (m_machine->is_data_locked() && flag)
	{
		SG_DEBUG("starting locked evaluation\n", get_name())
		/* do actual cross-validation */
		for (index_t i=0; i <num_subsets; ++i)
		{
			/* evtl. update xvalidation output class */
			CCrossValidationOutput* current=(CCrossValidationOutput*)
					m_xval_outputs->get_first_element();
			while (current)
			{
				current->update_fold_index(i);
				SG_UNREF(current);
				current=(CCrossValidationOutput*)
						m_xval_outputs->get_next_element();
			}

			/* index subset for training, will be freed below */
			SGVector<index_t> inverse_subset_indices =
					m_splitting_strategy->generate_subset_inverse(i);
			

			/* train machine on training features */
			m_machine->train_locked(inverse_subset_indices);

			/* feature subset for testing */
			SGVector<index_t> subset_indices =
					m_splitting_strategy->generate_subset_indices(i);
			//CMath::qsort(subset_indices);
			

			/* evtl. update xvalidation output class */
			current=(CCrossValidationOutput*)m_xval_outputs->get_first_element();
			while (current)
			{
				current->update_train_indices(inverse_subset_indices, "\t");
				current->update_trained_machine(m_machine, "\t");
				SG_UNREF(current);
				current=(CCrossValidationOutput*)
						m_xval_outputs->get_next_element();
			}

			/* produce output for desired indices */
			CLabels* result_labels=m_machine->apply_locked(subset_indices);
			SG_REF(result_labels);

			/* set subset for testing labels */
			m_labels->add_subset(subset_indices);

			/* evaluate against own labels */
			m_evaluation_criterion->set_indices(subset_indices);
			results[i]=m_evaluation_criterion->evaluate(result_labels, m_labels);

			/* evtl. update xvalidation output class */
			current=(CCrossValidationOutput*)m_xval_outputs->get_first_element();
			while (current)
			{
				current->update_test_indices(subset_indices, "\t");
				current->update_test_result(result_labels, "\t");
				current->update_test_true_result(m_labels, "\t");
				current->post_update_results();
				current->update_evaluation_result(results[i], "\t");
				SG_UNREF(current);
				current=(CCrossValidationOutput*)
						m_xval_outputs->get_next_element();
			}

			/* remove subset to prevent side effects */
			m_labels->remove_subset();

			/* clean up */
			SG_UNREF(result_labels);

			SG_DEBUG("done locked evaluation\n", get_name())
		}
	}
	else
	{
		SG_DEBUG("starting unlocked evaluation\n", get_name())
		/* tell machine to store model internally
		 * (otherwise changing subset of features will kaboom the classifier) */
		m_machine->set_store_model_features(true);

		/* do actual cross-validation */
#pragma omp parallel for
		for (index_t i=0; i <num_subsets; ++i)
		{
			CMachine* machine;
			CKernel* k;
			if (m_machine->get_classifier_type()==CT_LIBSVM)
			{
				machine=new CLibSVM();
				SG_REF(machine);
				((CLibSVM*)machine)->set_C(((CLibSVM*)m_machine)->get_C1(), ((CLibSVM*)m_machine)->get_C2());
				CKernel* ker=((CLibSVM*)m_machine)->get_kernel();
				if (ker->get_kernel_type() == K_GAUSSIAN)
				{
					//k=(CKernel*)ker->shallow_copy();
					k=new CGaussianKernel(((CGaussianKernel*)ker)->get_cache_size(), ((CGaussianKernel*)ker)->get_width());
					SG_REF(k);
				}
				((CLibSVM*)machine)->set_kernel(k);
			}
			machine->set_store_model_features(true);
			
			/* evtl. update xvalidation output class */
			CCrossValidationOutput* current=(CCrossValidationOutput*)
					m_xval_outputs->get_first_element();
			while (current)
			{
				current->update_fold_index(i);
				SG_UNREF(current);
				current=(CCrossValidationOutput*)
						m_xval_outputs->get_next_element();
			}
			
			/* set feature subset for training */
			SGVector<index_t> inverse_subset_indices=
					m_splitting_strategy->generate_subset_inverse(i);
			//CMath::qsort(inverse_subset_indices);						
			
			CFeatures* temp_feat=CFeatures::create_shallow_copy(m_features);
			SG_REF(temp_feat);
			k->init(temp_feat, temp_feat);

			temp_feat->add_subset(inverse_subset_indices);
			for (index_t p=0; p<temp_feat->get_num_preprocessors(); p++)
			{
				CPreprocessor* preprocessor = temp_feat->get_preprocessor(p);
				preprocessor->init(temp_feat);
				SG_UNREF(preprocessor);
			}
			
			CLabels* lab;
			/* set label subset for training */
			if (m_labels->get_label_type()==LT_BINARY)
			{	
				lab= new CBinaryLabels();
				SG_REF(lab);		
				SGVector<float64_t> vec=((CBinaryLabels*)m_labels)->get_labels();
				((CBinaryLabels*)lab)->set_labels(vec);
			}
			else
			{
				lab=m_labels;
				//SG_REF(m_labels);
			}
			machine->set_labels(lab);	

			lab->add_subset(inverse_subset_indices);
			SG_DEBUG("training set %d:\n", i)
			if (io->get_loglevel()==MSG_DEBUG)
			{
				SGVector<index_t>::display_vector(inverse_subset_indices.vector,
						inverse_subset_indices.vlen, "training indices");
			}

			/* train machine on training features and remove subset */
			SG_DEBUG("starting training\n")
			machine->train(temp_feat);
			SG_DEBUG("finished training\n")
	
			/* evtl. update xvalidation output class */
			current=(CCrossValidationOutput*)m_xval_outputs->get_first_element();
			while (current)
			{
				current->update_train_indices(inverse_subset_indices, "\t");
				current->update_trained_machine(machine, "\t");
				SG_UNREF(current);
				current=(CCrossValidationOutput*)
						m_xval_outputs->get_next_element();
			}

			temp_feat->remove_subset();
			lab->remove_subset();

			/* set feature subset for testing (subset method that stores pointer) */
			SGVector<index_t> subset_indices =
					m_splitting_strategy->generate_subset_indices(i);
			temp_feat->add_subset(subset_indices);

			/* set label subset for testing */
			lab->add_subset(subset_indices);

			SG_DEBUG("test set %d:\n", i)
			if (io->get_loglevel()==MSG_DEBUG)
			{
				SGVector<index_t>::display_vector(subset_indices.vector,
						subset_indices.vlen, "test indices");
			}

			/* apply machine to test features and remove subset */
			SG_DEBUG("starting evaluation\n")
			SG_DEBUG("%p\n", m_features)
			CLabels* result_labels=machine->apply(temp_feat);
			SG_DEBUG("finished evaluation\n")
			temp_feat->remove_subset();
			SG_REF(result_labels);

			/* evaluate */
			results[i]=m_evaluation_criterion->evaluate(result_labels, lab);
			SG_DEBUG("result on fold %d is %f\n", i, results[i])

			/* evtl. update xvalidation output class */
			current=(CCrossValidationOutput*)m_xval_outputs->get_first_element();
			while (current)
			{
				current->update_test_indices(subset_indices, "\t");
				current->update_test_result(result_labels, "\t");
				current->update_test_true_result(lab, "\t");
				current->post_update_results();
				current->update_evaluation_result(results[i], "\t");
				SG_UNREF(current);
				current=(CCrossValidationOutput*)
						m_xval_outputs->get_next_element();
			}

			/* clean up, remove subsets */
			lab->remove_subset();
			SG_UNREF(machine);
			SG_UNREF(k);						
			SG_UNREF(temp_feat);
			SG_UNREF(lab);
			SG_UNREF(result_labels);
		}

		SG_DEBUG("done unlocked evaluation\n", get_name())
	}

	/* build arithmetic mean of results */
	float64_t mean=CStatistics::mean(results);

	SG_DEBUG("leaving %s::evaluate_one_run()\n", get_name())
	return mean;
}

void CCrossValidation::add_cross_validation_output(
			CCrossValidationOutput* cross_validation_output)
{
	m_xval_outputs->append_element(cross_validation_output);
}
