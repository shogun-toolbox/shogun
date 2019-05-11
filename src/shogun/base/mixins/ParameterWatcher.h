#ifndef __PARAMETERWATCHER_H__
#define __PARAMETERWATCHER_H__

#include <shogun/base/AnyParameter.h>
#include <shogun/base/base_types.h>
#include <shogun/base/macros.h>
#include <shogun/base/some.h>
#include <shogun/base/unique.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/any.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/lib/exception/ShogunException.h>
#include <shogun/lib/tag.h>
#include <shogun/util/mixins.h>

#include <rxcpp/operators/rx-filter.hpp>
#include <rxcpp/rx-lite.hpp>

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shogun
{
	class ObservedValue;
	class ParameterObserver;

	template <typename M>
	class HouseKeeper;
	template <typename M>
	class ParameterHandler;

	template <typename M>
	IGNORE_IN_CLASSLIST class ParameterWatcher
	    : public mixin<M, requires<HouseKeeper, ParameterHandler>>
	{
		using Derived = typename M::derived_t;

		/** Definition of observed subject */
		typedef rxcpp::subjects::subject<Some<ObservedValue>> SGSubject;
		/** Definition of observable */
		typedef rxcpp::observable<
		    Some<ObservedValue>, rxcpp::dynamic_observable<Some<ObservedValue>>>
		    SGObservable;
		/** Definition of subscriber */
		typedef rxcpp::subscriber<
		    Some<ObservedValue>,
		    rxcpp::observer<Some<ObservedValue>, void, void, void, void>>
		    SGSubscriber;

	public:
		/** default constructor */
		ParameterWatcher();

		/** copy constructor */
		ParameterWatcher(const ParameterWatcher<M>& orig);

		/** destructor */
		virtual ~ParameterWatcher();

		/** Puts a pointer to some parameter into the parameter map.
		 *
		 * @param name name of the parameter
		 * @param value pointer to the parameter value
		 * @param properties properties of the parameter (e.g. if model
		 * selection is supported)
		 */
		template <typename T>
		void watch_param(
		    const std::string& name, T* value,
		    AnyParameterProperties properties = AnyParameterProperties())
		{
			BaseTag tag(name);
			param_handler.create_parameter(
			    tag, AnyParameter(make_any_ref(value), properties));
		}

		/** Puts a pointer to some parameter into the parameter map.
		 * The parameter is expected to be initialised at runtime
		 * using the provided lambda.
		 *
		 * @param name name of the parameter
		 * @param value pointer to the parameter value
		 * @param auto_init AutoInit object to initialise the value of the
		 * parameter
		 * @param properties properties of the parameter (e.g. if model
		 * selection is supported)
		 */
		template <typename T>
		void watch_param(
		    const std::string& name, T* value,
		    std::shared_ptr<params::AutoInit> auto_init,
		    AnyParameterProperties properties = AnyParameterProperties())
		{
			BaseTag tag(name);
			param_handler.create_parameter(
			    tag,
			    AnyParameter(
			        make_any_ref(value), properties, std::move(auto_init)));
		}

		/** Puts a pointer to some parameter array into the parameter map.
		 *
		 * @param name name of the parameter array
		 * @param value pointer to the first element of the parameter array
		 * @param len number of elements in the array
		 * @param properties properties of the parameter (e.g. if model
		 * selection is supported)
		 */
		template <typename T, typename S>
		void watch_param(
		    const std::string& name, T** value, S* len,
		    AnyParameterProperties properties = AnyParameterProperties())
		{
			BaseTag tag(name);
			param_handler.create_parameter(
			    tag, AnyParameter(make_any_ref(value, len), properties));
		}

		/** Puts a pointer to some 2d parameter array (i.e. a matrix) into the
		 * parameter map.
		 *
		 * @param name name of the parameter array
		 * @param value pointer to the first element of the parameter array
		 * @param rows number of rows in the array
		 * @param cols number of columns in the array
		 * @param properties properties of the parameter (e.g. if model
		 * selection is supported)
		 */
		template <typename T, typename S>
		void watch_param(
		    const std::string& name, T** value, S* rows, S* cols,
		    AnyParameterProperties properties = AnyParameterProperties())
		{
			BaseTag tag(name);
			param_handler.create_parameter(
			    tag, AnyParameter(make_any_ref(value, rows, cols), properties));
		}

#ifndef SWIG
		/** Puts a pointer to a (lazily evaluated) function into the parameter
		 * map.
		 *
		 * @param name name of the parameter
		 * @param method pointer to the method
		 */
		template <typename T, typename S>
		void watch_method(const std::string& name, T (S::*method)() const)
		{
			BaseTag tag(name);
			AnyParameterProperties properties(
			    "Dynamic parameter", ParameterProperties::HYPER |
			                             ParameterProperties::GRADIENT |
			                             ParameterProperties::MODEL);
			std::function<T()> bind_method =
			    std::bind(method, dynamic_cast<const S*>(this));
			param_handler.create_parameter(
			    tag, AnyParameter(make_any(bind_method), properties));
		}

		/**
		 * Observe a parameter value and emit them to observer.
		 * @param value Observed parameter's value
		 */
		void observe(const Some<ObservedValue> value) const
		{
			m_subscriber_params->on_next(value);
		}

		/**
		 * Observe a parameter value given some information
		 * @tparam T value of the parameter
		 * @param step step
		 * @param name name of the observed value
		 * @param description description
		 * @param value observed value
		 */
		template <class T>
		void observe(
		    const int64_t step, const std::string& name,
		    const std::string& description, const T value) const;

		/**
		 * Observe a registered tag.
		 * @tparam T type of the tag
		 * @param step step
		 * @param name tag's name
		 */
		template <class T>
		void observe(const int64_t step, const std::string& name) const;

		/**
		 * Get parameters observable
		 * @return RxCpp observable
		 */
		SGObservable* get_parameters_observable()
		{
			return m_observable_params;
		};
#endif
	protected:
		/**
		 * Return total subscriptions
		 * @return total number of subscriptions
		 */
		index_t get_num_subscriptions() const
		{
			return static_cast<index_t>(m_subscriptions.size());
		}

		/**
		 * Register which params this object can emit.
		 * @param name the param name
		 * @param type the param type
		 * @param description a user oriented description
		 */
		void register_observable(
		    const std::string& name, const std::string& description);

	private:
		class ParameterObserverList;
		Unique<ParameterObserverList> param_obs_list;

		/** Subject used to create the params observer */
		SGSubject* m_subject_params;

		/** Parameter Observable */
		SGObservable* m_observable_params;

		/** Subscriber used to call onNext, onComplete etc.*/
		SGSubscriber* m_subscriber_params;

		/** List of subscription for this SGObject */
		std::map<int64_t, rxcpp::subscription> m_subscriptions;
		int64_t m_next_subscription_index;

		// mixins
		typename M::template requirement_t<HouseKeeper>& house_keeper;
		typename M::template requirement_t<ParameterHandler>& param_handler;
		SGIO*& io;
	};

	template <class T>
	class ObservedValueTemplated;

	/**
	 * Observed value which is emitted by algorithms.
	 */
	class ObservedValue
	    : public composition<
	          ObservedValue, HouseKeeper, ParameterHandler, ParameterWatcher>
	{
	public:
		/**
		 * Constructor
		 * @param step step
		 * @param name name of the observed value
		 */
		ObservedValue(const int64_t step, const std::string& name);

		/**
		 * Destructor
		 */
		~ObservedValue(){};

#ifndef SWIG
		/**
		 * Return a any version of the stored type.
		 * @return the any value.
		 */
		virtual Any get_any() const
		{
			return m_any_value;
		}
#endif

		/** @return object name */
		virtual const char* get_name() const
		{
			return "ObservedValue";
		}

	protected:
		/** ObservedValue step (used by Tensorboard to print graphs) */
		int64_t m_step;
		/** Parameter's name */
		std::string m_name;
		/** Untyped value */
		Any m_any_value;
	};

	/**
	 * Templated specialisation of ObservedValue that stores the actual data.
	 * @tparam T the type of the observed value
	 */
	template <class T>
	class ObservedValueTemplated : public ObservedValue
	{

	public:
		/**
		 * Constructor
		 * @param step step
		 * @param name the observed value's name
		 * @param value the observed value
		 */
		ObservedValueTemplated(
		    const int64_t step, const std::string& name,
		    const std::string& description, const T value)
		    : ObservedValue(step, name), m_observed_value(value)
		{
			this->watch_param(
			    name, &m_observed_value,
			    AnyParameterProperties(
			        description, ParameterProperties::READONLY));
			m_any_value = make_any(m_observed_value);
		}

		/**
		 * Constructor which takes AnyParameterProperties for the observed value
		 * @param step step
		 * @param name the observed value's name
		 * @param value the observed value
		 * @param properties properties of that observed value
		 */
		ObservedValueTemplated(
		    const int64_t step, const std::string& name, const T value,
		    const AnyParameterProperties properties)
		    : ObservedValue(step, name), m_observed_value(value)
		{
			this->watch_param(name, &m_observed_value, properties);
			m_any_value = make_any(m_observed_value);
		}

		/**
		 * Destructor
		 */
		~ObservedValueTemplated(){};

	private:
		/**
		 * Templated observed value
		 */
		T m_observed_value;
	};

	template <class M>
	template <class T>
	void ParameterWatcher<M>::observe(
	    const int64_t step, const std::string& name,
	    const std::string& description, const T value) const
	{
		auto obs =
		    some<ObservedValueTemplated<T>>(step, name, description, value);
		this->observe(obs);
	}

	template <class M>
	template <class T>
	void ParameterWatcher<M>::observe(
	    const int64_t step, const std::string& name) const
	{
		auto param = param_handler.get_parameter(BaseTag(name));
		auto obs = some<ObservedValueTemplated<T>>(
		    step, name, any_cast<T>(param.get_value()), param.get_properties());
		this->observe(obs);
	}

} // namespace shogun

#endif // __PARAMETERWATCHER_H__
