/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#ifndef SHOGUN_PARAMETEROBSERVERCV_H
#define SHOGUN_PARAMETEROBSERVERCV_H

#include <shogun/evaluation/CrossValidationStorage.h>
#include <shogun/lib/observers/ParameterObserver.h>
#include <shogun/lib/observers/observers_utils.h>

namespace shogun
{

	/**
	 * Base ParameterObserver class for CrossValidation.
	 */
	class ParameterObserverCV : public ParameterObserver
	{

	public:
		ParameterObserverCV();

		ParameterObserverCV(std::vector<std::string>& parameters);

		ParameterObserverCV(std::vector<ParameterProperties>& properties);

		ParameterObserverCV(
		    std::vector<std::string>& parameters,
		    std::vector<ParameterProperties>& properties);

		ParameterObserverCV(
		    const std::string& filename, std::vector<std::string>& parameters,
		    std::vector<ParameterProperties>& properties);

		~ParameterObserverCV() override;
		void on_error(std::exception_ptr ptr) override;
		void on_complete() override;

		/**
		* Get class name.
		* @return class name
		*/
		const char* get_name() const override
		{
			return "ParameterObserverCV";
		}

	private:
		/**
		 * Print data contained into a CrossValidationStorage object.
		 * @param value CrossValidationStorage object
		 */
		void print_observed_value(const std::shared_ptr<CrossValidationStorage>& value) const;

		/**
		 * Print information of a machine
		 * @param machine given machine
		 */
		void print_machine_information(const std::shared_ptr<Machine>& machine) const;

	protected:
		void on_next_impl(const TimedObservedValue& value) override;
	};
}

#endif // SHOGUN_PARAMETEROBSERVERCV_H
