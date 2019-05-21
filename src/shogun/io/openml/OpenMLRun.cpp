/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/evaluation/CrossValidationStorage.h>
#include <shogun/io/openml/OpenMLFile.h>
#include <shogun/io/openml/OpenMLRun.h>
#include <shogun/io/openml/ShogunOpenML.h>
#include <shogun/io/openml/utils.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/Machine.h>

using namespace shogun;
using namespace shogun::openml_detail;
using namespace rapidjson;

std::shared_ptr<OpenMLRun> OpenMLRun::run_model_on_task(
    std::shared_ptr<CSGObject> model, std::shared_ptr<OpenMLTask> task)
{
	SG_SNOTIMPLEMENTED
	return std::shared_ptr<OpenMLRun>();
}

std::shared_ptr<OpenMLRun> OpenMLRun::run_flow_on_task(
    std::shared_ptr<OpenMLFlow> flow, std::shared_ptr<OpenMLTask> task,
    bool avoid_duplicate_runs)
{
	if (avoid_duplicate_runs && flow->exists_on_server())
	{
		auto flow_from_server =
		    OpenMLFlow::download_flow(flow->get_flow_id(), "");
	}

	auto data = task->get_dataset();

	auto model = ShogunOpenML::flow_to_model(flow, true);
	flow->set_model(model);

	auto labels = data->get_labels();
	auto features = data->get_features(data->get_default_target_attribute());

	auto machine = std::dynamic_pointer_cast<CMachine>(model);
	if (!machine)
	{
		SG_SERROR("INTERNAL ERROR: failed to cast model to machine!\n")
	}

	auto* xval_storage = new CrossValidationStorage();

	if (task->get_split()->contains_splits())
	{
		auto train_idx = task->get_train_indices();
		auto test_idx = task->get_test_indices();

		xval_storage->set_num_runs(task->get_num_repeats());
		xval_storage->set_num_folds(task->get_num_fold());

		machine->set_store_model_features(true);

		for (auto repeat_idx : range(task->get_num_repeats()))
		{
			for (auto fold_idx : range(task->get_num_fold()))
			{
				SGVector<index_t> train_i_idx(
				    train_idx[repeat_idx][fold_idx].data(),
				    train_idx[repeat_idx][fold_idx].size());
				SGVector<index_t> test_i_idx(
				    train_idx[repeat_idx][fold_idx].data(),
				    train_idx[repeat_idx][fold_idx].size());
				xval_storage->append_fold_result(
				    ShogunOpenML::run_model_on_fold(
				        machine, task, features, labels, train_i_idx,
				        test_i_idx, repeat_idx, fold_idx)
				        .release());
			}
		}
	}
	else
	{
		xval_storage->set_num_runs(0);
		xval_storage->set_num_folds(0);
		xval_storage->append_fold_result(
		    ShogunOpenML::run_model_on_fold(machine, task, features, labels)
		        .release());
	}
	SG_SDEBUG("End of openml run: %s\n", xval_storage->to_string().c_str());

	return std::make_shared<OpenMLRun>(
	    nullptr,                    // uploader
	    nullptr,                    // uploader_name
	    nullptr,                    // setup_id
	    nullptr,                    // setup_string
	    nullptr,                    // parameter_settings
	    std::vector<float64_t>{},   // evaluations
	    std::vector<float64_t>{},   // fold_evaluations
	    std::vector<float64_t>{},   // sample_evaluations
	    nullptr,                    // data_content
	    std::vector<std::string>{}, // output_files
	    task,                       // task
	    flow,                       // flow
	    nullptr,                    // run_id
	    model,                      // model
	    std::vector<std::string>{}, // tags
	    nullptr                     // predictions_url
	);
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
