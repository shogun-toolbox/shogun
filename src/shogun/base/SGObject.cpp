/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein,
 *          Giovanni De Toni, Jacob Walker, Fernando Iglesias, Roman Votyakov,
 *          Soumyajit De, Evgeniy Andreev, Evangelos Anagnostopoulos,
 *          Leon Kuchenbecker, Sanuj Sharma, Wu Lin
 */

#include <shogun/lib/RefCount.h>
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

	class CSGObject::Self
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

	template<> void CSGObject::set_generic<bool>()
	{
		m_generic = PT_BOOL;
	}

	template<> void CSGObject::set_generic<char>()
	{
		m_generic = PT_CHAR;
	}

	template<> void CSGObject::set_generic<int8_t>()
	{
		m_generic = PT_INT8;
	}

	template<> void CSGObject::set_generic<uint8_t>()
	{
		m_generic = PT_UINT8;
	}

	template<> void CSGObject::set_generic<int16_t>()
	{
		m_generic = PT_INT16;
	}

	template<> void CSGObject::set_generic<uint16_t>()
	{
		m_generic = PT_UINT16;
	}

	template<> void CSGObject::set_generic<int32_t>()
	{
		m_generic = PT_INT32;
	}

	template<> void CSGObject::set_generic<uint32_t>()
	{
		m_generic = PT_UINT32;
	}

	template<> void CSGObject::set_generic<int64_t>()
	{
		m_generic = PT_INT64;
	}

	template<> void CSGObject::set_generic<uint64_t>()
	{
		m_generic = PT_UINT64;
	}

	template<> void CSGObject::set_generic<float32_t>()
	{
		m_generic = PT_FLOAT32;
	}

	template<> void CSGObject::set_generic<float64_t>()
	{
		m_generic = PT_FLOAT64;
	}

	template<> void CSGObject::set_generic<floatmax_t>()
	{
		m_generic = PT_FLOATMAX;
	}

	template<> void CSGObject::set_generic<CSGObject*>()
	{
		m_generic = PT_SGOBJECT;
	}

	template<> void CSGObject::set_generic<complex128_t>()
	{
		m_generic = PT_COMPLEX128;
	}
	class CSGObject::ParameterObserverList
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

CSGObject::CSGObject() : self(), param_obs_list()
{
	init();
	m_refcount = new RefCount(0);

	SG_GCDEBUG("SGObject created ({})", fmt::ptr(this))
}

CSGObject::CSGObject(const CSGObject& orig)
    : self(), param_obs_list()
{
	init();
	m_refcount = new RefCount(0);

	SG_GCDEBUG("SGObject copied ({})", fmt::ptr(this))
}

CSGObject::~CSGObject()
{
	SG_GCDEBUG("SGObject destroyed ({})", fmt::ptr(this))

	delete m_parameters;
	delete m_model_selection_parameters;
	delete m_gradient_parameters;
	delete m_refcount;
	delete m_subject_params;
	delete m_observable_params;
	delete m_subscriber_params;
}

int32_t CSGObject::ref()
{
	int32_t count = m_refcount->ref();
	SG_GCDEBUG("ref() refcount {} obj {} ({}) increased", count, this->get_name(), fmt::ptr(this))
	return m_refcount->ref_count();
}

int32_t CSGObject::ref_count()
{
	int32_t count = m_refcount->ref_count();
	SG_GCDEBUG("ref_count(): refcount {}, obj {} ({})", count, this->get_name(), fmt::ptr(this))
	return m_refcount->ref_count();
}

int32_t CSGObject::unref()
{
	int32_t count = m_refcount->unref();
	if (count<=0)
	{
		SG_GCDEBUG("unref() refcount {}, obj {} ({}) destroying", count, this->get_name(), fmt::ptr(this))
		delete this;
		return 0;
	}
	else
	{
		SG_GCDEBUG("unref() refcount {} obj {} ({}) decreased", count, this->get_name(), fmt::ptr(this))
		return m_refcount->ref_count();
	}
}

CSGObject * CSGObject::shallow_copy() const
{
	not_implemented(SOURCE_LOCATION);
	return NULL;
}

CSGObject * CSGObject::deep_copy() const
{
	not_implemented(SOURCE_LOCATION);
	return NULL;
}

void CSGObject::update_parameter_hash() const
{
	SG_DEBUG("entering")

	m_hash = hash();

	SG_DEBUG("leaving")
}

bool CSGObject::parameter_hash_changed() const
{
	SG_DEBUG("entering")

	SG_DEBUG("leaving")
	return (m_hash!=hash());
}

Parallel* CSGObject::get_global_parallel()
{
	return env();
}

bool CSGObject::is_generic(EPrimitiveType* generic) const
{
	*generic = m_generic;

	return m_generic != PT_NOT_GENERIC;
}

void CSGObject::unset_generic()
{
	m_generic = PT_NOT_GENERIC;
}

bool CSGObject::serialize(io::CSerializer* ser)
{
	require(ser != nullptr, "Serializer format object should be non-null");
	ser->write(wrap(this));
	return true;
}

bool CSGObject::deserialize(io::CDeserializer* deser)
{
	require(deser != nullptr, "Deserializer format object should be non-null");
	deser->read(this);
	return true;
}

void CSGObject::load_serializable_pre() noexcept(false)
{
	m_load_pre_called = true;
}

void CSGObject::load_serializable_post() noexcept(false)
{
	m_load_post_called = true;
}

void CSGObject::save_serializable_pre() noexcept(false)
{
	m_save_pre_called = true;
}

void CSGObject::save_serializable_post() noexcept(false)
{
	m_save_post_called = true;
}

void CSGObject::init()
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

	m_subject_params = new SGSubject();
	m_observable_params = new SGObservable(m_subject_params->get_observable());
	m_subscriber_params = new SGSubscriber(m_subject_params->get_subscriber());
	m_next_subscription_index = 0;

	watch_method("num_subscriptions", &CSGObject::get_num_subscriptions);
}

#ifdef SWIG
std::string CSGObject::get_description(const std::string& name) const
#else
std::string CSGObject::get_description(std::string_view name) const
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

void CSGObject::print_modsel_params()
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

void CSGObject::build_gradient_parameter_dictionary(CMap<TParameter*, CSGObject*>* dict)
{
	for (index_t i=0; i<m_gradient_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_gradient_parameters->get_parameter(i);
		dict->add(p, this);
	}

	for (index_t i=0; i<m_model_selection_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_model_selection_parameters->get_parameter(i);
		CSGObject* child=*(CSGObject**)(p->m_parameter);

		if ((p->m_datatype.m_ptype == PT_SGOBJECT) &&
				(p->m_datatype.m_ctype == CT_SCALAR) &&	child)
		{
			child->build_gradient_parameter_dictionary(dict);
		}
	}
}

CSGObject* CSGObject::clone(ParameterProperties pp) const
{
	SG_DEBUG("Starting to clone {} at {}.", get_name(), fmt::ptr(this));
	SG_DEBUG("Constructing an empty instance of {}.", get_name());
	CSGObject* clone = create_empty();
	SG_DEBUG("Empty instance of {} created at {}.", get_name(), fmt::ptr(clone));

	require(
	    clone, "Could not create empty instance of {}. The reason for "
	          "this usually is that get_name() of the class returns something "
	          "wrong, that a class has a wrongly set generic type, or that it "
	          "lies outside the main source tree and does not have "
	          "CSGObject::create_empty() overridden.\n",
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

	SG_DEBUG("Done cloning {} at {}, new object at {}.", get_name(), fmt::ptr(this), fmt::ptr(clone));
	return clone;
}

void CSGObject::create_parameter(
    const BaseTag& _tag, const AnyParameter& parameter)
{
	self->create(_tag, parameter);
}

void CSGObject::update_parameter(const BaseTag& _tag, const Any& value)
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

AnyParameter CSGObject::get_parameter(const BaseTag& _tag) const
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

AnyParameter CSGObject::get_function(const BaseTag& _tag) const
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

bool CSGObject::has_parameter(const BaseTag& _tag) const
{
	return self->has(_tag);
}

void CSGObject::add_callback_function(
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

void CSGObject::subscribe(ParameterObserver* obs)
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
	m_subscriptions.insert(
	    std::make_pair<int64_t, rxcpp::subscription>(
	        std::move(m_next_subscription_index), std::move(subscription)));

	obs->put("subscription_id", m_next_subscription_index);

	m_next_subscription_index++;
}

void CSGObject::unsubscribe(ParameterObserver* obs)
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

void CSGObject::observe(const Some<ObservedValue> value) const
{
	m_subscriber_params->on_next(value);
}

void CSGObject::observe(ObservedValue* value) const
{
	auto somed_value = Some<ObservedValue>::from_raw(value);
	m_subscriber_params->on_next(somed_value);
}

void CSGObject::register_observable(
    std::string_view name, std::string_view description)
{
	param_obs_list->register_param(name, description);
}

std::vector<std::string> CSGObject::observable_names()
{
	std::vector<std::string> list;
	std::transform(
	    param_obs_list->get_list().begin(), param_obs_list->get_list().end(),
	    list.begin(), [](auto const& x) { return x.first; });
	return list;
}

#ifdef SWIG
bool CSGObject::has(const std::string& name) const
#else
bool CSGObject::has(std::string_view name) const
#endif
{
	return has_parameter(BaseTag(name));
}

std::string CSGObject::to_string() const
{
	std::stringstream ss;
	std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));
	ss << get_name();
	ss << "(";
	for (auto it = self->map.begin(); it != self->map.end(); ++it)
	{
		ss << it->first.name() << "=";
		auto value = it->second.get_value();
		if (m_string_to_enum_map.find(it->first.name()) !=
		    m_string_to_enum_map.end())
		{
			ss << string_enum_reverse_lookup(it->first.name(), any_cast<machine_int_t>(value));
		}
		else if (value.visitable())
		{
			try
			{
				value.visit(visitor.get());
			}
			catch(...)
			{
				ss << "null";
			}
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
std::map<std::string, std::shared_ptr<const AnyParameter>> CSGObject::get_params() const
{
	std::map<std::string, std::shared_ptr<const AnyParameter>> result;
	for (auto const& each: self->map) {
		result.emplace(each.first.name().data(), std::make_shared<const AnyParameter>(each.second));
	}
	return result;
}
#endif

bool CSGObject::equals(const CSGObject* other) const
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
void CSGObject::for_each_param_of_type(
    std::function<void(const std::string&, T*)> operation)
{
	auto visitor = std::make_unique<FilterVisitor<T>>(operation);
	const auto& param_map = self->map;

	std::for_each(param_map.begin(), param_map.end(), [&](auto& pair) {
		Any any_param = pair.second.get_value();
		if (any_param.visitable() && !pair.second.get_properties().has_property(ParameterProperties::RUNFUNCTION) && pair.first.name() != RandomMixin<CSGObject>::kSetRandomSeed)
		{
			const std::string name = pair.first.name();
			visitor->set_name(name);
			any_param.visit(visitor.get());
		}
	});
}

template void shogun::CSGObject::for_each_param_of_type<CSGObject*>(
    std::function<void(const std::string&, CSGObject**)>);

CSGObject* CSGObject::create_empty() const
{
	CSGObject* object = create(this->get_name(), this->m_generic);
	SG_REF(object);
	return object;
}

void CSGObject::init_auto_params()
{
	auto params = self->filter(ParameterProperties::AUTO);
	for (const auto& param : params)
	{
		update_parameter(param.first, param.second.get_init_function()->operator()());
	}
}

#ifdef SWIG
CSGObject* CSGObject::get(const std::string& name, index_t index) const
#else
CSGObject* CSGObject::get(std::string_view name, index_t index) const
#endif
{
	auto* result = sgo_details::get_by_tag(this, name, sgo_details::GetByNameIndex(index));
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
CSGObject* CSGObject::get(std::string_view name, std::nothrow_t) const
    noexcept
{
	return sgo_details::get_by_tag(this, name, sgo_details::GetByName());
}
#endif

#ifdef SWIG
CSGObject* CSGObject::get(const std::string& name) const noexcept(false)
#else
CSGObject* CSGObject::get(std::string_view name) const noexcept(false)
#endif
{
	if (!has(name))
	{
		error("Parameter {}::{} does not exist.", get_name(), name.data());
	}
	if (auto* result = get(name, std::nothrow))
	{
		return result;
	}
	error(
			"Cannot get parameter {}::{} of type {} as object.",
			get_name(), name.data(),
			self->map[BaseTag(name.data())].get_value().type().c_str());
	return nullptr;
}

std::string CSGObject::string_enum_reverse_lookup(
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

void CSGObject::visit_parameter(const BaseTag& _tag, AnyVisitor* v) const
{
	auto p = get_parameter(_tag);
	p.get_value().visit(v);
}

size_t CSGObject::hash() const
{
	return std::hash<CSGObject>{}(*this);
}
