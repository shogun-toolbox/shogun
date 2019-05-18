/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLTASK_H
#define SHOGUN_OPENMLTASK_H

#include <shogun/io/openml/OpenMLData.h>
#include <shogun/io/openml/OpenMLSplit.h>

namespace shogun
{
	/**
	 * Handles OpenML tasks. A task contains all the information
	 * required to train and test a model.
	 */
	class OpenMLTask
	{
	public:
		enum class TaskType
		{
			SUPERVISED_CLASSIFICATION = 0,
			SUPERVISED_REGRESSION = 1,
			LEARNING_CURVE = 2,
			SUPERVISED_DATASTREAM_CLASSIFICATION = 3,
			CLUSTERING = 4,
			MACHINE_LEARNING_CHALLENGE = 5,
			SURVIVAL_ANALYSIS = 6,
			SUBGROUP_DISCOVERY = 7
		};

		enum class TaskEvaluation
		{

		};

		OpenMLTask(
		    const std::string& task_id, const std::string task_name,
		    TaskType task_type, const std::string& task_type_id,
		    std::unordered_map<std::string, std::string> evaluation_measures,
		    std::shared_ptr<OpenMLSplit> split,
		    std::shared_ptr<OpenMLData> data)
		    : m_task_id(task_id), m_task_name(task_name),
		      m_task_type(task_type), m_task_type_id(task_type_id),
		      m_evaluation_measures(std::move(evaluation_measures)),
		      m_split(std::move(split)), m_data(std::move(data))
		{
		}

		static std::shared_ptr<OpenMLTask>
		get_task(const std::string& task_id, const std::string& api_key);

		std::shared_ptr<OpenMLData> get_dataset() const noexcept
		{
			return m_data;
		}

		std::shared_ptr<OpenMLSplit> get_split() const noexcept
		{
			return m_split;
		}

		std::vector<std::vector<int64_t>> get_train_indices() const;

		std::vector<std::vector<int64_t>> get_test_indices() const;

#ifndef SWIG
		SG_FORCED_INLINE TaskType

		get_task_type() const noexcept
		{
			return m_task_type;
		}

#endif // SWIG

	private:
		static TaskType get_task_from_string(const std::string& task_type);

		std::vector<std::vector<int64_t>>
		get_indices(const std::vector<std::vector<int64_t>>& idx) const;

		std::string m_task_id;
		std::string m_task_name;
		TaskType m_task_type;
		std::string m_task_type_id;
		std::unordered_map<std::string, std::string> m_evaluation_measures;
		std::shared_ptr<OpenMLSplit> m_split;
		std::shared_ptr<OpenMLData> m_data;
	};
} // namespace shogun

#endif // SHOGUN_OPENMLTASK_H
