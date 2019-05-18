/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

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
	std::shared_ptr<CFeatures> train_features = nullptr, test_features = nullptr;
	std::shared_ptr<CLabels> train_labels = nullptr, test_labels = nullptr;

	if (task->get_split()->contains_splits())
		SG_SNOTIMPLEMENTED
	else
	{
		train_labels = data->get_labels();
		train_features =
				data->get_features(data->get_default_target_attribute());
		// ensures delete is called by shared ptr destructor
		SG_REF(train_labels.get())
		SG_REF(train_features.get())
		auto model = ShogunOpenML::flow_to_model(std::move(flow), true);

		if (auto machine = std::dynamic_pointer_cast<CMachine>(model))
		{
			auto result = ShogunOpenML::run_model_on_fold(
					machine, task, train_features, 0, 0, train_labels,
					test_features);
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
