/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein,
 *          Giovanni De Toni, Jacob Walker, Fernando Iglesias, Roman Votyakov,
 *          Soumyajit De, Evgeniy Andreev, Evangelos Anagnostopoulos,
 *          Leon Kuchenbecker, Sanuj Sharma, Wu Lin
 */

#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>

#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Version.h>
#include <shogun/base/class_list.h>
#include <shogun/io/visitors/ToStringVisitor.h>
#include <shogun/io/serialization/Serializer.h>
#include <shogun/io/serialization/Deserializer.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/observers/ParameterObserver.h>
#include <shogun/mathematics/RandomMixin.h>

#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>

#include <rxcpp/operators/rx-filter.hpp>
#include <rxcpp/rx-lite.hpp>
#include <rxcpp/rx-subscription.hpp>

#include <algorithm>
#include <memory>
#include <unordered_map>

#include <shogun/distance/Distance.h>
#include <shogun/evaluation/EvaluationResult.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/Machine.h>
#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/multiclass/ecoc/ECOCDecoder.h>
#include <shogun/multiclass/ecoc/ECOCEncoder.h>

#include <shogun/lib/observers/ObservedValue.h>
#include <shogun/util/visitors/FilterVisitor.h>

#include <shogun/util/hash.h>

namespace shogun
{

	typedef std::map<BaseTag, AnyParameter> ParametersMap;
	typedef std::unordered_map<std::string_view, std::string_view> ObsParamsList;

	class SGObject::Self
	{
	public:
		void create(const BaseTag& tag, const AnyParameter& parameter)
		{
			if (has(tag))
			{
				error("Can not register {} twice", tag.name().c_str());
			}
			map[tag] = parameter;
		}

		void update(const BaseTag& tag, const Any& value)
		{
			if (!has(tag))
			{
				error(
				    "Can not update unregistered parameter {}",
				    tag.name().c_str());
			}
			map.at(tag).set_value(value);
		}

		AnyParameter& at(const BaseTag& tag)
		{
			return map.at(tag);
		}

		const AnyParameter& at(const BaseTag& tag) const
		{
			return map.at(tag);
		}

		AnyParameter get(const BaseTag& tag) const
		{
			if(!has(tag))
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
			    [&pprop](const auto& each) {
					auto p = each.second.get_properties();
					// if the filter mask is ALL, also include parameters with no set properties (NONE)
				    return p.has_property(pprop) ||
								(pprop==ParameterProperties::ALL &&
								p.compare_mask(ParameterProperties::NONE));
			    });
			return result;
		}

		ParametersMap map;
	};

	class Parallel;

	template<> void SGObject::set_generic<bool>()
	{
		m_generic = PT_BOOL;
	}

	template<> void SGObject::set_generic<char>()
	{
		m_generic = PT_CHAR;
	}

	template<> void SGObject::set_generic<int8_t>()
	{
		m_generic = PT_INT8;
	}

	template<> void SGObject::set_generic<uint8_t>()
	{
		m_generic = PT_UINT8;
	}

	template<> void SGObject::set_generic<int16_t>()
	{
		m_generic = PT_INT16;
	}

	template<> void SGObject::set_generic<uint16_t>()
	{
		m_generic = PT_UINT16;
	}

	template<> void SGObject::set_generic<int32_t>()
	{
		m_generic = PT_INT32;
	}

	template<> void SGObject::set_generic<uint32_t>()
	{
		m_generic = PT_UINT32;
	}

	template<> void SGObject::set_generic<int64_t>()
	{
		m_generic = PT_INT64;
	}

	template<> void SGObject::set_generic<uint64_t>()
	{
		m_generic = PT_UINT64;
	}

	template<> void SGObject::set_generic<float32_t>()
	{
		m_generic = PT_FLOAT32;
	}

	template<> void SGObject::set_generic<float64_t>()
	{
		m_generic = PT_FLOAT64;
	}

	template<> void SGObject::set_generic<floatmax_t>()
	{
		m_generic = PT_FLOATMAX;
	}

	template<> void SGObject::set_generic<SGObject*>()
	{
		m_generic = PT_SGOBJECT;
	}

	template<> void SGObject::set_generic<complex128_t>()
	{
		m_generic = PT_COMPLEX128;
	}
	class SGObject::ParameterObserverList
	{
	public:
		void register_param(std::string_view name, std::string_view description)
		{
			m_list_obs_params[name.data()] = description;
		}

		ObsParamsList get_list() const
		{
			return m_list_obs_params;
		}

	private:
		ObsParamsList m_list_obs_params;
	};
} /* namespace shogun  */

using namespace shogun;

SGObject::SGObject() : self(), param_obs_list()
{
	init();

	SG_TRACE("SGObject created ({})", fmt::ptr(this));
}

SGObject::SGObject(const SGObject& orig)
    : self(), param_obs_list()
{
	init();

	SG_TRACE("SGObject copied ({})", fmt::ptr(this));
}

SGObject::~SGObject()
{
	SG_TRACE("SGObject destroyed ({})", fmt::ptr(this));

	delete m_parameters;
	delete m_model_selection_parameters;
	delete m_gradient_parameters;
}

std::shared_ptr<SGObject> SGObject::shallow_copy() const
{
	not_implemented(SOURCE_LOCATION);
	return NULL;
}

std::shared_ptr<SGObject> SGObject::deep_copy() const
{
	not_implemented(SOURCE_LOCATION);
	return NULL;
}

void SGObject::update_parameter_hash() const
{
	SG_TRACE("entering");

	m_hash = hash();

	SG_TRACE("leaving");
}

bool SGObject::parameter_hash_changed() const
{
	return (m_hash!=hash());
}

Parallel* SGObject::get_global_parallel()
{
	return env();
}

bool SGObject::is_generic(EPrimitiveType* generic) const
{
	*generic = m_generic;

	return m_generic != PT_NOT_GENERIC;
}

void SGObject::unset_generic()
{
	m_generic = PT_NOT_GENERIC;
}

bool SGObject::serialize(std::shared_ptr<io::Serializer> ser)
{
	require(ser != nullptr, "Serializer format object should be non-null");
	ser->write(shared_from_this());
	return true;
}

bool SGObject::deserialize(std::shared_ptr<io::Deserializer> deser)
{
	require(deser != nullptr, "Deserializer format object should be non-null");
	deser->read(shared_from_this());
	return true;
}

void SGObject::load_serializable_pre() noexcept(false)
{
	m_load_pre_called = true;
}

void SGObject::load_serializable_post() noexcept(false)
{
	m_load_post_called = true;
}

void SGObject::save_serializable_pre() noexcept(false)
{
	m_save_pre_called = true;
}

void SGObject::save_serializable_post() noexcept(false)
{
	m_save_post_called = true;
}

void SGObject::init()
{
	m_parameters = new Parameter();
	m_model_selection_parameters = new Parameter();
	m_gradient_parameters=new Parameter();
	m_generic = PT_NOT_GENERIC;
	m_load_pre_called = false;
	m_load_post_called = false;
	m_save_pre_called = false;
	m_save_post_called = false;
	m_hash = 0;

	m_subject_params = std::make_shared<SGSubject>();
	m_observable_params = std::make_shared<SGObservable>(m_subject_params->get_observable());
	m_subscriber_params = std::make_shared<SGSubscriber>(m_subject_params->get_subscriber());
	m_next_subscription_index = 0;

	watch_method("num_subscriptions", &SGObject::get_num_subscriptions);
}

#ifdef SWIG
std::string SGObject::get_description(const std::string& name) const
#else
std::string SGObject::get_description(std::string_view name) const
#endif
{
	auto it = self->map.find(BaseTag(name));
	if (it != self->map.end())
	{
		return std::string(it->second.get_properties().get_description());
	}
	else
	{
		error(
		    "There is no parameter called '{}' in {}", name.data(),
		    this->get_name());
		return "";
	}
}

void SGObject::print_modsel_params()
{
	io::print("parameters available for model selection for {}:\n", get_name());

	index_t num_param=m_model_selection_parameters->get_num_parameters();

	if (!num_param)
		io::print("\tnone\n");

	for (index_t i=0; i<num_param; i++)
	{
		TParameter* current=m_model_selection_parameters->get_parameter(i);
		index_t  l=200;
		char* type=SG_MALLOC(char, l);
		if (type)
		{
			current->m_datatype.to_string(type, l);
			io::print("\t{} ({}): {}\n", current->m_name, current->m_description,
					type);
			SG_FREE(type);
		}
	}
}

void SGObject::build_gradient_parameter_dictionary(std::shared_ptr<CMap<TParameter*, SGObject*>> dict)
{
	for (index_t i=0; i<m_gradient_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_gradient_parameters->get_parameter(i);
		dict->add(p, this);
	}

	for (index_t i=0; i<m_model_selection_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_model_selection_parameters->get_parameter(i);
		SGObject* child=*(SGObject**)(p->m_parameter);

		if ((p->m_datatype.m_ptype == PT_SGOBJECT) &&
				(p->m_datatype.m_ctype == CT_SCALAR) &&	child)
		{
			child->build_gradient_parameter_dictionary(dict);
		}
	}
}

std::shared_ptr<SGObject> SGObject::clone(ParameterProperties pp) const
{
	SG_DEBUG("Starting to clone {} at {}.", get_name(), fmt::ptr(this));
	SG_DEBUG("Constructing an empty instance of {}.", get_name());
	auto clone = create_empty();
	SG_DEBUG("Empty instance of {} created at {}.", get_name(), fmt::ptr(clone.get()));

	require(
	    clone, "Could not create empty instance of {}. The reason for "
	          "this usually is that get_name() of the class returns something "
	          "wrong, that a class has a wrongly set generic type, or that it "
	          "lies outside the main source tree and does not have "
	          "SGObject::create_empty() overridden.\n",
	    get_name());

	for (const auto &it : self->filter(pp))
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.cloneable())
		{
			SG_DEBUG(
			    "Skipping clone of {}::{} of type {}.", this->get_name(),
			    tag.name().c_str(), own.type().c_str());
			continue;
		}

		SG_DEBUG(
			"Cloning parameter {}::{} of type {}.", this->get_name(),
			tag.name().c_str(), own.type().c_str());

		clone->get_parameter(tag).get_value().clone_from(own);
	}

	SG_DEBUG("Done cloning {} at {}, new object at {}.", get_name(), fmt::ptr(this), fmt::ptr(clone.get()));
	return clone;
}

void SGObject::create_parameter(
    const BaseTag& _tag, const AnyParameter& parameter)
{
	self->create(_tag, parameter);
}

void SGObject::update_parameter(const BaseTag& _tag, const Any& value)
{
	auto& param = self->at(_tag);
	auto& pprop = param.get_properties();
	if (pprop.has_property(ParameterProperties::READONLY))
		error(
		    "{}::{} is marked as read-only and cannot be modified!",
		    get_name(), _tag.name().c_str());

	if (pprop.has_property(ParameterProperties::CONSTRAIN))
	{
		auto msg = self->map[_tag].get_constrain_function()(value);
		if (!msg.empty())
		{
			error(
					"{}::{} cannot be updated because it must be: {}!",
					get_name(), _tag.name().c_str(), msg.c_str());
		}
	}
	self->update(_tag, value);
	for (auto& method : param.get_callbacks())
		method();

	pprop.remove_property(ParameterProperties::AUTO);
}

AnyParameter SGObject::get_parameter(const BaseTag& _tag) const
{
	const auto& parameter = self->get(_tag);
	if (parameter.get_properties().has_property(
	        ParameterProperties::RUNFUNCTION))
	{
		error(
		    "The parameter {}::{} is registered as a function, "
		    "use the .run() method instead!\n",
		    get_name(), _tag.name().c_str());
	}
	if (parameter.get_value().empty())
	{
		error(
		    "There is no parameter called \"{}\" in {}", _tag.name().c_str(),
		    get_name());
	}
	return parameter;
}

AnyParameter SGObject::get_function(const BaseTag& _tag) const
{
	const auto& parameter = self->get(_tag);
	if (!parameter.get_properties().has_property(
	        ParameterProperties::RUNFUNCTION))
	{
		error(
		    "The parameter {}::{} is not registered as a function, "
		    "use the .get() method instead",
		    get_name(), _tag.name().c_str());
	}
	if (parameter.get_value().empty())
	{
		error(
		    "There is no parameter called \"{}\" in {}", _tag.name().c_str(),
		    get_name());
	}
	return parameter;
}

bool SGObject::has_parameter(const BaseTag& _tag) const
{
	return self->has(_tag);
}

void SGObject::add_callback_function(
    std::string_view name, std::function<void()> function)
{
	require(function, "Function object is not callable");
	BaseTag tag(name);
	require(
	    has_parameter(tag), "There is no parameter called \"{}\" in {}",
	    tag.name().c_str(), get_name());

	auto& param = self->at(tag);
	param.add_callback_function(std::move(function));
}

void SGObject::subscribe(std::shared_ptr<ParameterObserver> obs)
{
	auto sub = rxcpp::make_subscriber<TimedObservedValue>(
	    [obs](TimedObservedValue e) { obs->on_next(e); },
	    [obs](std::exception_ptr ep) { obs->on_error(ep); },
	    [obs]() { obs->on_complete(); });

	// Create an observable which emits values only if they are about
	// parameters selected by the observable.
	rxcpp::subscription subscription =
	    m_observable_params
	        ->filter([obs](std::shared_ptr<ObservedValue> v) {
		        return obs->filter(v->get<std::string>("name"));
		    })
	        .timestamp()
	        .subscribe(sub);

	// Insert the subscription in the list
	m_subscriptions.insert(
	    std::make_pair<int64_t, rxcpp::subscription>(
	        std::move(m_next_subscription_index), std::move(subscription)));

	obs->put("subscription_id", m_next_subscription_index);

	m_next_subscription_index++;
}

void SGObject::unsubscribe(std::shared_ptr<ParameterObserver> obs)
{

	int64_t index = obs->get<int64_t>("subscription_id");

	// Check if we have such subscription
	auto it = m_subscriptions.find(index);
	if (it == m_subscriptions.end())
		error(
		    "The object {} does not have any registered parameter observer "
		    "with index {}",
		    this->get_name(), index);

	it->second.unsubscribe();
	m_subscriptions.erase(index);

	obs->put("subscription_id", static_cast<int64_t>(-1));
}

void SGObject::observe(std::shared_ptr<ObservedValue> value) const
{
	m_subscriber_params->on_next(value);
}

void SGObject::register_observable(
    std::string_view name, std::string_view description)
{
	param_obs_list->register_param(name, description);
}

std::vector<std::string> SGObject::observable_names()
{
	std::vector<std::string> list;
	std::transform(
	    param_obs_list->get_list().begin(), param_obs_list->get_list().end(),
	    list.begin(), [](auto const& x) { return x.first; });
	return list;
}

#ifdef SWIG
bool SGObject::has(const std::string& name) const
#else
bool SGObject::has(std::string_view name) const
#endif
{
	return has_parameter(BaseTag(name));
}

std::string SGObject::to_string() const
{
	std::stringstream ss;
	std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));
	ss << get_name();
	ss << "(";
	for (auto it = self->map.begin(); it != self->map.end(); ++it)
	{
		ss << it->first.name() << "=";
		auto value = it->second.get_value();
		if (m_string_to_enum_map.count(it->first.name()))
		{
			ss << string_enum_reverse_lookup(it->first.name(), any_cast<machine_int_t>(value));
		}
		else if (value.safe_visitable())
		{
			value.visit(visitor.get());
		}
		else
		{
			ss << "{function}";
		}
		if (std::next(it) != (self->map.end()))
		{
			ss << ", ";
		}
	}
	ss << ")";
	return ss.str();
}

#ifndef SWIG // SWIG should skip this part
std::map<std::string, std::shared_ptr<const AnyParameter>> SGObject::get_params() const
{
	std::map<std::string, std::shared_ptr<const AnyParameter>> result;
	for (auto const& each: self->map) {
		result.emplace(each.first.name().data(), std::make_shared<const AnyParameter>(each.second));
	}
	return result;
}
#endif

bool SGObject::equals(const SGObject* other) const
{
	if (other == this)
		return true;

	if (other == nullptr)
	{
		SG_DEBUG("No object to compare to provided.");
		return false;
	}

	/* Assumption: can use SGObject::get_name to distinguish types */
	if (strcmp(this->get_name(), other->get_name()))
	{
		SG_DEBUG(
		    "Own type {} differs from provided {}.", get_name(),
		    other->get_name());
		return false;
	}

	/* Assumption: objects of same type have same set of tags. */
	for (const auto &it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.comparable())
		{
			SG_DEBUG(
			    "Skipping comparison of {}::{} of type {} as it is "
			    "non-visitable.",
			    this->get_name(), tag.name().c_str(), own.type().c_str());
			continue;
		}

		const Any& given = other->get_parameter(tag).get_value();

		SG_DEBUG(
		    "Comparing parameter {}::{} of type {}.", this->get_name(),
		    tag.name().c_str(), own.type().c_str());
		if (own != given)
		{
			if (env()->io()->get_loglevel() <= io::MSG_DEBUG)
			{
				std::stringstream ss;
				std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));

				ss << "Own parameter " << this->get_name() << "::" << tag.name()
				   << "=";
				own.visit(visitor.get());

				ss << " different from provided " << other->get_name()
				   << "::" << tag.name() << "=";
				given.visit(visitor.get());

				SG_DEBUG("{}", ss.str().c_str());
			}

			return false;
		}
	}

	SG_DEBUG("All parameters of {} equal.", this->get_name());
	return true;
}

template <typename T>
void SGObject::for_each_param_of_type(
    std::function<void(const std::string&, T*)> operation)
{
	auto visitor = std::make_unique<FilterVisitor<T>>(operation);
	const auto& param_map = self->map;

	std::for_each(param_map.begin(), param_map.end(), [&](auto& pair) {
		Any any_param = pair.second.get_value();
		if (any_param.safe_visitable())
		{
			const std::string name = pair.first.name();
			visitor->set_name(name);
			any_param.visit(visitor.get());
		}
	});
}

template void shogun::SGObject::for_each_param_of_type<SGObject*>(
    std::function<void(const std::string&, SGObject**)>);

bool SGObject::equals(std::shared_ptr<const SGObject> other) const
{
	return this->equals(other.get());
}

std::shared_ptr<SGObject> SGObject::create_empty() const
{
	return create(this->get_name(), this->m_generic);
}

void SGObject::init_auto_params()
{
	auto params = self->filter(ParameterProperties::AUTO);
	for (const auto& param : params)
	{
		update_parameter(param.first, param.second.get_init_function()->operator()());
	}
}

#ifdef SWIG
std::shared_ptr<SGObject> SGObject::get(const std::string& name, index_t index) const
#else
std::shared_ptr<SGObject> SGObject::get(std::string_view name, index_t index) const
#endif
{
	auto result = sgo_details::get_by_tag(shared_from_this(), name, sgo_details::GetByNameIndex(index));
	if (!result && has(name))
	{
		error(
			"Cannot get array parameter {}::{}[{}] of type {} as object.",
			get_name(), name.data(), index,
			self->map[BaseTag(name)].get_value().type().c_str());
	}
	return result;
}

#ifndef SWIG
std::shared_ptr<SGObject> SGObject::get(std::string_view name, std::nothrow_t) const
    noexcept
{
	return sgo_details::get_by_tag(shared_from_this(), name, sgo_details::GetByName());
}
#endif

#ifdef SWIG
std::shared_ptr<SGObject> SGObject::get(const std::string& name) const noexcept(false)
#else
std::shared_ptr<SGObject> SGObject::get(std::string_view name) const noexcept(false)
#endif
{
	if (!has(name))
	{
		error("Parameter {}::{} does not exist.", get_name(), name.data());
	}
	if (auto result = get(name, std::nothrow))
	{
		return result;
	}
	error(
			"Cannot get parameter {}::{} of type {} as object.",
			get_name(), name.data(),
			self->map[BaseTag(name.data())].get_value().type().c_str());
	return nullptr;
}

std::string SGObject::string_enum_reverse_lookup(
    std::string_view param, machine_int_t value) const
{
	auto param_enum_map = m_string_to_enum_map.at(param);
	auto enum_value = value;
	auto enum_map_it = std::find_if(
	    param_enum_map.begin(), param_enum_map.end(),
	    [&enum_value](const std::pair<std::string_view, machine_int_t>& p) {
		    return p.second == enum_value;
	    });
	return std::string(enum_map_it->first);
}

void SGObject::visit_parameter(const BaseTag& _tag, AnyVisitor* v) const
{
	auto p = get_parameter(_tag);
	p.get_value().visit(v);
}

size_t SGObject::hash() const
{
	return std::hash<SGObject>{}(*this);
}
