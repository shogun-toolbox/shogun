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
#include <shogun/base/SGObject.h>
#include <shogun/base/Version.h>
#include <shogun/base/class_list.h>
#include <shogun/io/visitors/ToStringVisitor.h>
#include <shogun/io/serialization/Serializer.h>
#include <shogun/io/serialization/Deserializer.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/observers/ParameterObserver.h>

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

namespace shogun
{

	typedef std::map<BaseTag, AnyParameter> ParametersMap;
	typedef std::unordered_map<std::string, std::string> ObsParamsList;

	class CSGObject::Self
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
			    [&pprop](const std::pair<BaseTag, AnyParameter>& each) {
				    return each.second.get_properties().has_property(pprop);
			    });
			return result;
		}

		ParametersMap map;
	};

	class Parallel;

	extern Parallel* sg_parallel;
	extern SGIO* sg_io;
	extern Version* sg_version;

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
		void register_param(const std::string& name, const std::string& description)
		{
			m_list_obs_params[name] = description;
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
	set_global_objects();
	m_refcount = new RefCount(0);

	SG_SGCDEBUG("SGObject created (%p)\n", this)
}

CSGObject::CSGObject(const CSGObject& orig)
    : self(), param_obs_list(), io(orig.io), parallel(orig.parallel),
      version(orig.version)
{
	init();
	set_global_objects();
	m_refcount = new RefCount(0);

	SG_SGCDEBUG("SGObject copied (%p)\n", this)
}

CSGObject::~CSGObject()
{
	SG_SGCDEBUG("SGObject destroyed (%p)\n", this)

	unset_global_objects();
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
	SG_SGCDEBUG("ref() refcount %ld obj %s (%p) increased\n", count, this->get_name(), this)
	return m_refcount->ref_count();
}

int32_t CSGObject::ref_count()
{
	int32_t count = m_refcount->ref_count();
	SG_SGCDEBUG("ref_count(): refcount %d, obj %s (%p)\n", count, this->get_name(), this)
	return m_refcount->ref_count();
}

int32_t CSGObject::unref()
{
	int32_t count = m_refcount->unref();
	if (count<=0)
	{
		SG_SGCDEBUG("unref() refcount %ld, obj %s (%p) destroying\n", count, this->get_name(), this)
		delete this;
		return 0;
	}
	else
	{
		SG_SGCDEBUG("unref() refcount %ld obj %s (%p) decreased\n", count, this->get_name(), this)
		return m_refcount->ref_count();
	}
}

CSGObject * CSGObject::shallow_copy() const
{
	SG_NOTIMPLEMENTED
	return NULL;
}

CSGObject * CSGObject::deep_copy() const
{
	SG_NOTIMPLEMENTED
	return NULL;
}

void CSGObject::set_global_objects()
{
	if (!sg_io || !sg_parallel || !sg_version)
	{
		fprintf(stderr, "call init_shogun() before using the library, dying.\n");
		exit(1);
	}

	SG_REF(sg_io);
	SG_REF(sg_parallel);
	SG_REF(sg_version);

	io=sg_io;
	parallel=sg_parallel;
	version=sg_version;
}

void CSGObject::unset_global_objects()
{
	SG_UNREF(version);
	SG_UNREF(parallel);
	SG_UNREF(io);
}

void CSGObject::set_global_io(SGIO* new_io)
{
	SG_REF(new_io);
	SG_UNREF(sg_io);
	sg_io=new_io;
}

SGIO* CSGObject::get_global_io()
{
	SG_REF(sg_io);
	return sg_io;
}

void CSGObject::set_global_parallel(Parallel* new_parallel)
{
	SG_REF(new_parallel);
	SG_UNREF(sg_parallel);
	sg_parallel=new_parallel;
}

void CSGObject::update_parameter_hash()
{
	SG_DEBUG("entering\n")

	uint32_t carry=0;
	uint32_t length=0;

	m_hash=0;
	get_parameter_incremental_hash(m_hash, carry, length);
	m_hash=CHash::FinalizeIncrementalMurmurHash3(m_hash, carry, length);

	SG_DEBUG("leaving\n")
}

bool CSGObject::parameter_hash_changed()
{
	SG_DEBUG("entering\n")

	uint32_t hash=0;
	uint32_t carry=0;
	uint32_t length=0;

	get_parameter_incremental_hash(hash, carry, length);
	hash=CHash::FinalizeIncrementalMurmurHash3(hash, carry, length);

	SG_DEBUG("leaving\n")
	return (m_hash!=hash);
}

Parallel* CSGObject::get_global_parallel()
{
	SG_REF(sg_parallel);
	return sg_parallel;
}

void CSGObject::set_global_version(Version* new_version)
{
	SG_REF(new_version);
	SG_UNREF(sg_version);
	sg_version=new_version;
}

Version* CSGObject::get_global_version()
{
	SG_REF(sg_version);
	return sg_version;
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
	REQUIRE(ser != nullptr, "Serializer format object should be non-null\n");
	ser->write(wrap(this));
	return true;
}

bool CSGObject::deserialize(io::CDeserializer* deser)
{
	REQUIRE(deser != nullptr, "Deserializer format object should be non-null\n");
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

	io = NULL;
	parallel = NULL;
	version = NULL;
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

std::string CSGObject::get_description(const std::string& name) const
{
	auto it = this->get_params().find(name);
	if (it != this->get_params().end())
	{
		return it->second.get()->get_properties().get_description();
	}
	else
	{
		SG_SERROR(
		    "There is no parameter called '%s' in %s", name.c_str(),
		    this->get_name());
	}
}

void CSGObject::print_modsel_params()
{
	SG_PRINT("parameters available for model selection for %s:\n", get_name())

	index_t num_param=m_model_selection_parameters->get_num_parameters();

	if (!num_param)
		SG_PRINT("\tnone\n")

	for (index_t i=0; i<num_param; i++)
	{
		TParameter* current=m_model_selection_parameters->get_parameter(i);
		index_t  l=200;
		char* type=SG_MALLOC(char, l);
		if (type)
		{
			current->m_datatype.to_string(type, l);
			SG_PRINT("\t%s (%s): %s\n", current->m_name, current->m_description,
					type);
			SG_FREE(type);
		}
	}
}

SGStringList<char> CSGObject::get_modelsel_names()
{
    index_t num_param=m_model_selection_parameters->get_num_parameters();

    SGStringList<char> result(num_param, -1);

	index_t max_string_length=-1;

    for (index_t i=0; i<num_param; i++)
    {
        char* name=m_model_selection_parameters->get_parameter(i)->m_name;
        index_t len=strlen(name);
		// +1 to have a zero terminated string
        result.strings[i]=SGString<char>(name, len+1);

        if (len>max_string_length)
            max_string_length=len;
    }

	result.max_string_length=max_string_length;

    return result;
}

char* CSGObject::get_modsel_param_descr(const char* param_name)
{
	index_t index=get_modsel_param_index(param_name);

	if (index<0)
	{
		SG_ERROR("There is no model selection parameter called \"%s\" for %s",
				param_name, get_name());
	}

	return m_model_selection_parameters->get_parameter(index)->m_description;
}

index_t CSGObject::get_modsel_param_index(const char* param_name)
{
	/* use fact that names extracted from below method are in same order than
	 * in m_model_selection_parameters variable */
	SGStringList<char> names=get_modelsel_names();

	/* search for parameter with provided name */
	index_t index=-1;
	for (index_t i=0; i<names.num_strings; i++)
	{
		TParameter* current=m_model_selection_parameters->get_parameter(i);
		if (!strcmp(param_name, current->m_name))
		{
			index=i;
			break;
		}
	}

	return index;
}

void CSGObject::get_parameter_incremental_hash(uint32_t& hash, uint32_t& carry,
		uint32_t& total_length)
{
	for (index_t i=0; i<m_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_parameters->get_parameter(i);

		SG_DEBUG("Updating hash for parameter %s.%s\n", get_name(), p->m_name);

		if (p->m_datatype.m_ptype == PT_SGOBJECT)
		{
			if (p->m_datatype.m_ctype == CT_SCALAR)
			{
				CSGObject* child = *((CSGObject**)(p->m_parameter));

				if (child)
				{
					child->get_parameter_incremental_hash(hash, carry,
							total_length);
				}
			}
			else if (p->m_datatype.m_ctype==CT_VECTOR ||
					p->m_datatype.m_ctype==CT_SGVECTOR)
			{
				CSGObject** child=(*(CSGObject***)(p->m_parameter));

				for (index_t j=0; j<*(p->m_datatype.m_length_y); j++)
				{
					if (child[j])
					{
						child[j]->get_parameter_incremental_hash(hash, carry,
								total_length);
					}
				}
			}
		}
		else
			p->get_incremental_hash(hash, carry, total_length);
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

CSGObject* CSGObject::clone() const
{
	SG_DEBUG("Starting to clone %s at %p.\n", get_name(), this);
	SG_DEBUG("Constructing an empty instance of %s.\n", get_name());
	CSGObject* clone = create_empty();
	SG_DEBUG("Empty instance of %s created at %p.\n", get_name(), clone);

	REQUIRE(
	    clone, "Could not create empty instance of %s. The reason for "
	          "this usually is that get_name() of the class returns something "
	          "wrong, that a class has a wrongly set generic type, or that it "
	          "lies outside the main source tree and does not have "
	          "CSGObject::create_empty() overridden.\n",
	    get_name());

	for (const auto &it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.cloneable())
		{
			SG_SDEBUG(
			    "Skipping clone of %s::%s of type %s.\n", this->get_name(),
			    tag.name().c_str(), own.type().c_str());
			continue;
		}

		SG_SDEBUG(
			"Cloning parameter %s::%s of type %s.\n", this->get_name(),
			tag.name().c_str(), own.type().c_str());

		clone->get_parameter(tag).get_value().clone_from(own);
	}

	SG_DEBUG("Done cloning %s at %p, new object at %p.\n", get_name(), this, clone);
	return clone;
}

void CSGObject::create_parameter(
    const BaseTag& _tag, const AnyParameter& parameter)
{
	self->create(_tag, parameter);
}

void CSGObject::update_parameter(const BaseTag& _tag, const Any& value)
{
	auto& pprop = self->map[_tag].get_properties();
	if (pprop.has_property(ParameterProperties::READONLY))
		SG_ERROR(
		    "%s::%s is marked as read-only and cannot be modified!\n",
		    get_name(), _tag.name().c_str());
	if (pprop.has_property(ParameterProperties::CONSTRAIN))
	{
		auto msg = self->map[_tag].get_constrain_function()(value);
		if (!msg.empty())
		{
			SG_ERROR(
					"%s::%s cannot be updated because it must be: %s!\n",
					get_name(), _tag.name().c_str(), msg.c_str())
		}
	}
	self->update(_tag, value);
	pprop.remove_property(ParameterProperties::AUTO);
}

AnyParameter CSGObject::get_parameter(const BaseTag& _tag) const
{
	const auto& parameter = self->get(_tag);
	if (parameter.get_properties().has_property(
	        ParameterProperties::RUNFUNCTION))
	{
		SG_ERROR(
		    "The parameter %s::%s is registered as a function, "
		    "use the .run() method instead!\n",
		    get_name(), _tag.name().c_str())
	}
	if (parameter.get_value().empty())
	{
		SG_ERROR(
		    "There is no parameter called \"%s\" in %s\n", _tag.name().c_str(),
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
		SG_ERROR(
		    "The parameter %s::%s is not registered as a function, "
		    "use the .get() method instead",
		    get_name(), _tag.name().c_str())
	}
	if (parameter.get_value().empty())
	{
		SG_ERROR(
		    "There is no parameter called \"%s\" in %s\n", _tag.name().c_str(),
		    get_name());
	}
	return parameter;
}

bool CSGObject::has_parameter(const BaseTag& _tag) const
{
	return self->has(_tag);
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
		SG_ERROR(
		    "The object %s does not have any registered parameter observer "
		    "with index %i",
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
    const std::string& name, const std::string& description)
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

bool CSGObject::has(const std::string& name) const
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

#ifndef SWIG // SWIG should skip this part
std::map<std::string, std::shared_ptr<const AnyParameter>> CSGObject::get_params() const
{
	std::map<std::string, std::shared_ptr<const AnyParameter>> result;
	for (auto const& each: self->map) {
		result.emplace(each.first.name(), std::make_shared<const AnyParameter>(each.second));
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
		SG_DEBUG("No object to compare to provided.\n");
		return false;
	}

	/* Assumption: can use SGObject::get_name to distinguish types */
	if (strcmp(this->get_name(), other->get_name()))
	{
		SG_DEBUG(
		    "Own type %s differs from provided %s.\n", get_name(),
		    other->get_name());
		return false;
	}

	/* Assumption: objects of same type have same set of tags. */
	for (const auto &it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.visitable())
		{
			SG_SDEBUG(
			    "Skipping comparison of %s::%s of type %s as it is "
			    "non-visitable.\n",
			    this->get_name(), tag.name().c_str(), own.type().c_str());
			continue;
		}

		const Any& given = other->get_parameter(tag).get_value();

		SG_SDEBUG(
		    "Comparing parameter %s::%s of type %s.\n", this->get_name(),
		    tag.name().c_str(), own.type().c_str());
		if (own != given)
		{
			if (io->get_loglevel() <= MSG_DEBUG)
			{
				std::stringstream ss;
				std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));

				ss << "Own parameter " << this->get_name() << "::" << tag.name()
				   << "=";
				own.visit(visitor.get());

				ss << " different from provided " << other->get_name()
				   << "::" << tag.name() << "=";
				given.visit(visitor.get());

				SG_SDEBUG("%s\n", ss.str().c_str());
			}

			return false;
		}
	}

	SG_SDEBUG("All parameters of %s equal.\n", this->get_name());
	return true;
}

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

CSGObject* CSGObject::get(const std::string& name, index_t index) const
{
	auto* result = sgo_details::get_by_tag(this, name, sgo_details::GetByNameIndex(index));
	if (!result && has(name))
	{
		SG_ERROR(
			"Cannot get array parameter %s::%s[%d] of type %s as object.\n",
			get_name(), name.c_str(), index,
			self->map[BaseTag(name)].get_value().type().c_str());
	}
	return result;
}

CSGObject* CSGObject::get(const std::string& name, std::nothrow_t) const
    noexcept
{
	return sgo_details::get_by_tag(this, name, sgo_details::GetByName());
}

CSGObject* CSGObject::get(const std::string& name) const noexcept(false)
{
	if (!has(name))
	{
		SG_ERROR("Parameter %s::%s does not exist.\n", get_name(), name.c_str())
	}
	if (auto* result = get(name, std::nothrow))
	{
		return result;
	}
	SG_ERROR(
			"Cannot get parameter %s::%s of type %s as object.\n",
			get_name(), name.c_str(),
			self->map[BaseTag(name)].get_value().type().c_str());
	return nullptr;
}

std::string CSGObject::string_enum_reverse_lookup(
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

void CSGObject::visit_parameter(const BaseTag& _tag, AnyVisitor* v) const
{
	auto p = get_parameter(_tag);
	p.get_value().visit(v);
}
