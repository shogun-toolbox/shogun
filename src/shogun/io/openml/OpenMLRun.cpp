/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/evaluation/CrossValidationStorage.h>
#include <shogun/io/openml/OpenMLRun.h>
#include <shogun/io/openml/ShogunOpenML.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/Machine.h>

using namespace shogun;

std::shared_ptr<OpenMLRun> OpenMLRun::run_model_on_task(
    std::shared_ptr<CSGObject> model, std::shared_ptr<OpenMLTask> task)
{
	SG_SNOTIMPLEMENTED
	return std::shared_ptr<OpenMLRun>();
}

std::shared_ptr<OpenMLRun> OpenMLRun::run_flow_on_task(
    std::shared_ptr<OpenMLFlow> flow, std::shared_ptr<OpenMLTask> task)
{
	auto data = task->get_dataset();
	std::shared_ptr<CFeatures> features = nullptr;
	std::shared_ptr<CLabels> labels = nullptr;

	auto model = ShogunOpenML::flow_to_model(std::move(flow), true);

	labels = data->get_labels();
	features = data->get_features(data->get_default_target_attribute());

	auto storage = std::make_shared<CrossValidationStorage>();

	if (task->get_split()->contains_splits())
	{
		auto machine = std::dynamic_pointer_cast<CMachine>(model);
		if (!machine)
		{
			SG_SERROR("INTERNAL ERROR: failed to cast model to machine!\n")
		}
		auto train_idx = task->get_train_indices();
		auto test_idx = task->get_test_indices();

		auto xval_storage = std::make_shared<CrossValidationStorage>();
		xval_storage->set_num_folds(task->get_num_fold());
		xval_storage->set_num_runs(task->get_num_repeats());
		machine->set_store_model_features(true);

		// copied/adapted from crossvalidation
		for (auto repeat_idx : range(task->get_num_repeats()))
		{
			for (auto fold_idx : range(task->get_num_fold()))
			{
				auto* fold = new CrossValidationFoldStorage();
				SG_REF(fold)

				auto cloned_machine = (CMachine*)machine->clone();

				// TODO while these are not used through const interfaces,
				//  we unfortunately have to clone, even though these could be
				//  shared
				auto features_clone = (CFeatures*)features->clone();
				auto labels_clone = (CLabels*)labels->clone();
				//				auto evaluation_criterion =
				//						(CEvaluation*)m_evaluation_criterion->clone();

				/* evtl. update xvalidation output class */
				fold->set_run_index(repeat_idx);
				fold->set_fold_index(fold_idx);

				auto train_fold_idx = SGVector<index_t>(
				    train_idx[repeat_idx][fold_idx].data(),
				    train_idx[repeat_idx][fold_idx].size(), false);

				features_clone->add_subset(train_fold_idx);

				/* set label subset for training */
				labels_clone->add_subset(train_fold_idx);

				SG_SDEBUG(
				    "train set repeat %d fold %d: %s\n", repeat_idx, fold_idx,
				    train_fold_idx.to_string().c_str())

				/* train machine on training features and remove subset */
				SG_SDEBUG("starting training\n")
				cloned_machine->set_labels(labels_clone);
				cloned_machine->train(features_clone);
				SG_SDEBUG("finished training\n")

				/* evtl. update xvalidation output class */
				fold->set_train_indices(train_fold_idx);
				auto fold_machine = (CMachine*)cloned_machine->clone();
				fold->set_trained_machine(fold_machine);
				SG_UNREF(fold_machine)

				features_clone->remove_subset();
				labels_clone->remove_subset();

				/* set feature subset for testing (subset method that stores
				 * pointer) */
				auto test_fold_idx = SGVector<index_t>(
				    test_idx[repeat_idx][fold_idx].data(),
				    test_idx[repeat_idx][fold_idx].size(), false);
				features_clone->add_subset(test_fold_idx);

				/* set label subset for testing */
				labels_clone->add_subset(test_fold_idx);

				SG_SDEBUG(
				    "test set repeat %d fold %d: %s\n", repeat_idx, fold_idx,
				    test_fold_idx.to_string().c_str())

				/* apply machine to test features and remove subset */
				SG_SDEBUG("starting evaluation\n")
				SG_SDEBUG("%p\n", features_clone)
				CLabels* result_labels = cloned_machine->apply(features_clone);
				SG_SDEBUG("finished evaluation\n")
				features_clone->remove_subset();
				SG_REF(result_labels);

				/* evaluate */
				//				results[i] =
				//				    evaluation_criterion->evaluate(result_labels,
				//labels); 				SG_DEBUG("result on fold %d is %f\n", i, results[i])

				/* evtl. update xvalidation output class */
				//				fold->set_test_indices(test_fold_idx);
				//				fold->set_test_result(result_labels);
				//				auto* true_labels = (CLabels*)labels->clone();
				//				fold->set_test_true_result(true_labels);
				//				SG_UNREF(true_labels)
				//				fold->post_update_results();
				//				fold->set_evaluation_result(results[i]);

				storage->append_fold_result(fold);
				//
				//				/* clean up, remove subsets */
				//				labels->remove_subset();
				SG_UNREF(cloned_machine);
				SG_UNREF(features_clone);
				SG_UNREF(labels_clone);
				// SG_UNREF(evaluation_criterion);
				//				SG_UNREF(result_labels);
				SG_UNREF(fold)
			}
		}
	}
	else
	{
		// ensures delete is called by shared ptr destructor
		SG_REF(labels.get())
		SG_REF(features.get())
		if (auto machine = std::dynamic_pointer_cast<CMachine>(model))
		{
			auto result = ShogunOpenML::run_model_on_fold(
			    machine, task, features, 0, 0, labels,
				nullptr);
			SG_SDEBUG(result->to_string().c_str());
		}
		else
			SG_SERROR("INTERNAL ERROR: failed to cast model to machine!\n")
	}
	return std::shared_ptr<OpenMLRun>();
}

std::shared_ptr<OpenMLRun>
OpenMLRun::from_filesystem(const std::string& directory)
{
	SG_SNOTIMPLEMENTED
	return nullptr;
}

void OpenMLRun::to_filesystem(const std::string& directory) const
{
	SG_SNOTIMPLEMENTED
}

void OpenMLRun::publish() const
{
	SG_SNOTIMPLEMENTED
}
