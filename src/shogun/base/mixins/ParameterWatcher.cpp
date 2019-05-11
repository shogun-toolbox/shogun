#include <shogun/base/DynArray.h>
#include <shogun/base/mixins/ParameterWatcher.h>
#include <shogun/base/mixins/SGObjectBase.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/observers/ParameterObserver.h>

#include <rxcpp/operators/rx-filter.hpp>
#include <rxcpp/rx-lite.hpp>
#include <rxcpp/rx-subscription.hpp>

#include <algorithm>
#include <unordered_map>
#include <utility>

using namespace shogun;

namespace shogun
{
	typedef std::unordered_map<std::string, std::string> ObsParamsList;
}

template <typename Derived>
ParameterWatcher<Derived>::ParameterWatcher(
    const ParameterWatcher<Derived>& orig)
    : ParameterWatcher()
{
}

template <typename Derived>
ParameterWatcher<Derived>::ParameterWatcher()
    : param_obs_list(),
      house_keeper((HouseKeeper<Derived>&)(*(Derived*)this)), // FIXME
      param_handler((ParameterHandler<Derived>&)(*(Derived*)this)),
      io(house_keeper.io)
{
	m_subject_params = new SGSubject();
	m_observable_params = new SGObservable(m_subject_params->get_observable());
	m_subscriber_params = new SGSubscriber(m_subject_params->get_subscriber());
	m_next_subscription_index = 0;

	watch_method(
	    "num_subscriptions", &ParameterWatcher<Derived>::get_num_subscriptions);
}

template <typename Derived>
ParameterWatcher<Derived>::~ParameterWatcher()
{
	delete m_subject_params;
	delete m_observable_params;
	delete m_subscriber_params;
}

template <typename Derived>
void ParameterWatcher<Derived>::subscribe(ParameterObserver* obs)
{
	auto sub = rxcpp::make_subscriber<TimedObservedValue>(
	    [obs](TimedObservedValue e) { obs->on_next(e); },
	    [obs](std::exception_ptr ep) { obs->on_error(ep); },
	    [obs]() { obs->on_complete(); });

	// Create an observable which emits values only if they are about
	// parameters selected by the observable.
	rxcpp::subscription subscription =
	    m_observable_params
	        ->filter([obs](Some<ObservedValue> v) {
		        return obs->filter(v->get<std::string>("name"));
	        })
	        .timestamp()
	        .subscribe(sub);

	// Insert the subscription in the list
	m_subscriptions.insert(std::make_pair<int64_t, rxcpp::subscription>(
	    std::move(m_next_subscription_index), std::move(subscription)));

	obs->put("subscription_id", m_next_subscription_index);

	m_next_subscription_index++;
}

template <typename Derived>
void ParameterWatcher<Derived>::unsubscribe(ParameterObserver* obs)
{
	int64_t index = obs->get<int64_t>("subscription_id");

	// Check if we have such subscription
	auto it = m_subscriptions.find(index);
	if (it == m_subscriptions.end())
		SG_ERROR(
		    "The object %s does not have any registered parameter observer "
		    "with index %i",
		    this->house_keeper.get_name(), index);

	it->second.unsubscribe();
	m_subscriptions.erase(index);

	obs->put("subscription_id", static_cast<int64_t>(-1));
}

template <typename Derived>
class ParameterWatcher<Derived>::ParameterObserverList
{
public:
	void register_param(const std::string& name, const std::string& description)
	{
		m_list_obs_params[name] = description;
	}

	ObsParamsList get_list() const
	{
		return m_list_obs_params;
	}

private:
	/** List of observable parameters (name, description) */
	ObsParamsList m_list_obs_params;
};

template <typename Derived>
void ParameterWatcher<Derived>::register_observable(
    const std::string& name, const std::string& description)
{
	param_obs_list->register_param(name, description);
}

template <typename Derived>
std::vector<std::string> ParameterWatcher<Derived>::observable_names()
{
	std::vector<std::string> list;
	std::transform(
	    param_obs_list->get_list().begin(), param_obs_list->get_list().end(),
	    list.begin(), [](auto const& x) { return x.first; });
	return list;
}

ObservedValue::ObservedValue(const int64_t step, const std::string& name)
    : HouseKeeper<ObservedValue>(), ParameterHandler<ObservedValue>(),
      ParameterWatcher<ObservedValue>(), CSGObjectBase<ObservedValue>(),
      m_step(step), m_name(name), m_any_value(Any())
{
	SG_ADD(&m_step, "step", "Step");
	this->watch_param(
	    "name", &m_name, AnyParameterProperties("Name of the observed value"));
}

namespace shogun
{
	template class ParameterWatcher<CSGObject>;
	template class ParameterWatcher<ObservedValue>;
	template class HouseKeeper<ObservedValue>;
} // namespace shogun