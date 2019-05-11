#include <shogun/base/mixins/ParameterHandler.h>

#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>

#include <shogun/base/DynArray.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/observers/ParameterObserver.h>

#include <rxcpp/operators/rx-filter.hpp>
#include <rxcpp/rx-lite.hpp>
#include <rxcpp/rx-subscription.hpp>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>

using namespace shogun;

namespace shogun
{
	typedef std::unordered_map<std::string, std::string> ObsParamsList;
	typedef std::map<BaseTag, AnyParameter> ParametersMap;

	template <typename M>
	class ParameterHandler<M>::Self
	{
	public:
		void create(const BaseTag& tag, const AnyParameter& parameter)
		{
			if (has(tag))
			{
				SG_SERROR("Can not register %s twice", tag.name().c_str())
			}
			map[tag] = parameter;
		}

		void update(const BaseTag& tag, const Any& value)
		{
			if (!has(tag))
			{
				SG_SERROR(
				    "Can not update unregistered parameter %s",
				    tag.name().c_str())
			}
			map.at(tag).set_value(value);
		}

		AnyParameter get(const BaseTag& tag) const
		{
			if (!has(tag))
				return AnyParameter();
			return map.at(tag);
		}

		bool has(const BaseTag& tag) const
		{
			return map.find(tag) != map.end();
		}

		ParametersMap filter(ParameterProperties pprop) const
		{
			ParametersMap result;
			std::copy_if(
			    map.cbegin(), map.cend(), std::inserter(result, result.end()),
			    [&pprop](const std::pair<BaseTag, AnyParameter>& each) {
				    return each.second.get_properties().has_property(pprop);
			    });
			return result;
		}

		ParametersMap map;
	}; // namespace shogun
} // namespace shogun

template <typename M>
ParameterHandler<M>::ParameterHandler(const ParameterHandler<M>& orig)
    : ParameterHandler()
{
}

template <typename M>
ParameterHandler<M>::ParameterHandler()
    : self(), house_keeper(M::template mutate<HouseKeeper>()),
      io(house_keeper.io)
{
	m_subject_params = new SGSubject();
	m_observable_params = new SGObservable(m_subject_params->get_observable());
	m_subscriber_params = new SGSubscriber(m_subject_params->get_subscriber());
	m_next_subscription_index = 0;

	watch_method(
	    "num_subscriptions", &ParameterWatcher<M>::get_num_subscriptions);
}

template <typename M>
ParameterHandler<M>::~ParameterHandler()
{
	delete m_subject_params;
	delete m_observable_params;
	delete m_subscriber_params;
}

template <typename M>
typename ParameterHandler<M>::Derived* ParameterHandler<M>::clone() const
{
	SG_DEBUG("Starting to clone %s at %p.\n", house_keeper.get_name(), this);
	SG_DEBUG(
	    "Constructing an empty instance of %s.\n", house_keeper.get_name());
	Derived* clone = create_empty();
	SG_DEBUG(
	    "Empty instance of %s created at %p.\n", house_keeper.get_name(),
	    clone);

	REQUIRE(
	    clone,
	    "Could not create empty instance of %s. The reason for "
	    "this usually is that get_name() of the class returns something "
	    "wrong, that a class has a wrongly set generic type, or that it "
	    "lies outside the main source tree and does not have "
	    "CSGObject::create_empty() overridden.\n",
	    house_keeper.get_name());

	for (const auto& it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.cloneable())
		{
			SG_SDEBUG(
			    "Skipping clone of %s::%s of type %s.\n",
			    this->house_keeper.get_name(), tag.name().c_str(),
			    own.type().c_str());
			continue;
		}

		SG_SDEBUG(
		    "Cloning parameter %s::%s of type %s.\n",
		    this->house_keeper.get_name(), tag.name().c_str(),
		    own.type().c_str());

		clone->get_parameter(tag).get_value().clone_from(own);
	}

	SG_DEBUG(
	    "Done cloning %s at %p, new object at %p.\n", house_keeper.get_name(),
	    this, clone);
	return clone;
}

template <typename M>
std::string ParameterHandler<M>::to_string() const
{
	std::stringstream ss;
	std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));
	ss << house_keeper.get_name();
	ss << "(";
	for (auto it = self->map.begin(); it != self->map.end(); ++it)
	{
		ss << it->first.name() << "=";
		auto value = it->second.get_value();
		if (m_string_to_enum_map.find(it->first.name()) !=
		    m_string_to_enum_map.end())
		{
			ss << string_enum_reverse_lookup(
			    it->first.name(), any_cast<machine_int_t>(value));
		}
		else if (value.visitable())
		{
			value.visit(visitor.get());
		}
		else
		{
			ss << "{function}";
		}
		if (std::next(it) != (self->map.end()))
		{
			ss << ",";
		}
	}
	ss << ")";
	return ss.str();
}

template <typename M>
bool ParameterHandler<M>::has(const std::string& name) const
{
	return has_parameter(BaseTag(name));
}

template <typename M>
void ParameterHandler<M>::create_parameter(
    const BaseTag& _tag, const AnyParameter& parameter)
{
	self->create(_tag, parameter);
}

template <typename M>
void ParameterHandler<M>::update_parameter(
    const BaseTag& _tag, const Any& value)
{
	if (!self->map[_tag].get_properties().has_property(
	        ParameterProperties::READONLY))
		self->update(_tag, value);
	else
	{
		SG_ERROR(
		    "%s::%s is marked as read-only and cannot be modified",
		    house_keeper.get_name(), _tag.name().c_str());
	}
	self->map[_tag].get_properties().remove_property(ParameterProperties::AUTO);
}

template <typename M>
AnyParameter ParameterHandler<M>::get_parameter(const BaseTag& _tag) const
{
	const auto& parameter = self->get(_tag);
	if (parameter.get_value().empty())
	{
		SG_ERROR(
		    "There is no parameter called \"%s\" in %s\n", _tag.name().c_str(),
		    house_keeper.get_name());
	}
	return parameter;
}

template <typename M>
bool ParameterHandler<M>::has_parameter(const BaseTag& _tag) const
{
	return self->has(_tag);
}

template <typename M>
std::map<std::string, std::shared_ptr<const AnyParameter>>
ParameterHandler<M>::get_params() const
{
	std::map<std::string, std::shared_ptr<const AnyParameter>> result;
	for (auto const& each : self->map)
	{
		result.emplace(
		    each.first.name(),
		    std::make_shared<const AnyParameter>(each.second));
	}
	return result;
}

template <typename M>
bool ParameterHandler<M>::equals(const Derived* other) const
{
	if (other == this)
		return true;

	if (other == nullptr)
	{
		SG_DEBUG("No object to compare to provided.\n");
		return false;
	}

	/* Assumption: can use SGObject::get_name to distinguish types */
	if (strcmp(this->house_keeper.get_name(), other->get_name()))
	{
		SG_DEBUG(
		    "Own type %s differs from provided %s.\n", house_keeper.get_name(),
		    other->get_name());
		return false;
	}

	/* Assumption: objects of same type have same set of tags. */
	for (const auto& it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.visitable())
		{
			SG_SDEBUG(
			    "Skipping comparison of %s::%s of type %s as it is "
			    "non-visitable.\n",
			    this->house_keeper.get_name(), tag.name().c_str(),
			    own.type().c_str());
			continue;
		}

		const Any& given = other->get_parameter(tag).get_value();

		SG_SDEBUG(
		    "Comparing parameter %s::%s of type %s.\n",
		    this->house_keeper.get_name(), tag.name().c_str(),
		    own.type().c_str());
		if (own != given)
		{
			if (house_keeper.io->get_loglevel() <= MSG_DEBUG)
			{
				std::stringstream ss;
				std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));

				ss << "Own parameter " << this->house_keeper.get_name()
				   << "::" << tag.name() << "=";
				own.visit(visitor.get());

				ss << " different from provided " << other->get_name()
				   << "::" << tag.name() << "=";
				given.visit(visitor.get());

				SG_SDEBUG("%s\n", ss.str().c_str());
			}

			return false;
		}
	}

	SG_SDEBUG("All parameters of %s equal.\n", this->house_keeper.get_name());
	return true;
}

template <typename M>
void ParameterHandler<M>::init_auto_params()
{
	auto params = self->filter(ParameterProperties::AUTO);
	for (const auto& param : params)
	{
		update_parameter(
		    param.first, param.second.get_init_function()->operator()());
	}
}

template <typename M>
std::string ParameterHandler<M>::string_enum_reverse_lookup(
    const std::string& param, machine_int_t value) const
{
	auto param_enum_map = m_string_to_enum_map.at(param);
	auto enum_value = value;
	auto enum_map_it = std::find_if(
	    param_enum_map.begin(), param_enum_map.end(),
	    [&enum_value](const std::pair<std::string, machine_int_t>& p) {
		    return p.second == enum_value;
	    });
	return enum_map_it->first;
}

template <typename M>
typename ParameterHandler<M>::Derived*
ParameterHandler<M>::get(const std::string& name, index_t index) const
{
	auto* result = sgo_details::get_by_tag(
	    (Derived*)this, name, std::move(sgo_details::GetByNameIndex(index)));
	if (!result && has(name))
	{
		SG_ERROR(
		    "Cannot get array parameter %s::%s[%d] of type %s as object.\n",
		    house_keeper.get_name(), name.c_str(), index,
		    self->map[BaseTag(name)].get_value().type().c_str());
	}
	return result;
}

template <typename M>
typename ParameterHandler<M>::Derived*
ParameterHandler<M>::get(const std::string& name, std::nothrow_t) const noexcept
{
	return sgo_details::get_by_tag(
	    (Derived*)this, name, std::move(sgo_details::GetByName()));
}

template <typename M>
typename ParameterHandler<M>::Derived*
ParameterHandler<M>::get(const std::string& name) const noexcept(false)
{
	if (!has(name))
	{
		SG_ERROR(
		    "Parameter %s::%s does not exist.\n", house_keeper.get_name(),
		    name.c_str())
	}
	if (auto* result = get(name, std::nothrow))
	{
		return result;
	}
	SG_ERROR(
	    "Cannot get parameter %s::%s of type %s as object.\n",
	    house_keeper.get_name(), name.c_str(),
	    self->map[BaseTag(name)].get_value().type().c_str());
	return nullptr;
}

namespace shogun
{
	template class ParameterHandler<mutator<
	    CSGObject, ParameterHandler, CSGObjectBase, HouseKeeper,
	    ParameterHandler, ParameterWatcher>>;
	template class ParameterHandler<mutator<
	    ObservedValue, ParameterHandler, HouseKeeper, ParameterHandler,
	    ParameterWatcher>>;
} // namespace shogun