/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#ifndef SHOGUN_PARAMETEROBSERVER_H
#define SHOGUN_PARAMETEROBSERVER_H

#include <stdexcept>
#include <vector>

#include <shogun/base/SGObject.h>
#include <shogun/lib/any.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/observers/observers_utils.h>

namespace shogun
{

	/**
	 * Interface for the parameter observer classes
	 */
	class ParameterObserver : public SGObject
	{

	public:
		/**
		* Default constructor
		*/
		ParameterObserver();

		/**
		 * Constructor
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserver(std::vector<std::string>& parameters);

		/**
		 * Constructor
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserver(std::vector<ParameterProperties>& properties);

		/**
		 * Constructor
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserver(
		    std::vector<std::string>& parameters,
		    std::vector<ParameterProperties>& properties);

		/**
		 * Constructor
		 * @param filename name of the generated output file
		 * @param parameters list of parameters which we want to watch over
		 * @param properties list of properties which we want to watch over
		 */
		ParameterObserver(
		    const std::string& filename, std::vector<std::string>& parameters,
		    std::vector<ParameterProperties>& properties);

		/**
		 * Virtual destructor
		 */
		virtual ~ParameterObserver();

		/**
		 * Filter function, check if the parameter name supplied is what
		 * we want to monitor
		 * @param param the param name
		 * @return true if param is found inside of m_parameters list
		 */
		virtual bool observes(const std::string& param);

		virtual bool observes(const AnyParameterProperties& property);

		/**
		 * Return a single observation from the received ones (not SG_REF).
		 * @tparam T the type of the observation
		 * @param i the index
		 * @return the observation casted to the requested type
		 */
		ObservedValue* get_observation(index_t i)
		{
			require(
			    i >= 0 && i < this->get_num_observations(),
			    "Observation index ({}) is out of bound (total observations "
			    "{})",
			    i, this->get_num_observations());
			return this->m_observations[i].get();
		};

		/**
		 * Erase all observations registered so far by the observer.
		 */
		virtual void clear()
		{
			m_observations.clear();
		};

		/**
		 * Method which will be called when the parameter observable emits a
		 * value.
		 * @param value the value emitted by the parameter observable
		 */
		void on_next(const TimedObservedValue& value)
		{
			m_observations.push_back(value.first);
			on_next_impl(value);
		};

		/**
		 * Method which will be called on errors
		 */
		virtual void on_error(std::exception_ptr) = 0;
		/**
		 * Method which will be called on completion
		 */
		virtual void on_complete() = 0;

		/**
		 * Get the name of this class
		 * @return name as a string
		 */
		virtual const char* get_name() const
		{
			return "ParameterObserver";
		}

	protected:
		/**
		 * Get the total number of observation received.
		 * @return number of obsevation received.
		 */
		index_t get_num_observations() const;

		/**
		 * Implementation of the on_next method which will be needed
		 * in order to process the observed value
		 * @param value the observed value
		 */
		virtual void on_next_impl(const TimedObservedValue& value) = 0;

		/**
		 * List of parameter's names we want to monitor
		 */
		std::vector<std::string> m_observed_parameters;

		std::vector<ParameterProperties> m_observed_properties;

		/**
		 * Observations recorded each time we compute on_next()
		 */
		std::vector<std::shared_ptr<ObservedValue>> m_observations;

		/**
		 * Subscription id set when I subscribe to a machine
		 */
		int64_t m_subscription_id;
	};
}

#endif // SHOGUN_PARAMETEROBSERVER_H
