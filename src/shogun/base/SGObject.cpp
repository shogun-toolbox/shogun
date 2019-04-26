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
#include <shogun/io/SerializableFile.h>
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
#include <utility>

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

namespace shogun
{

	typedef std::map<BaseTag, AnyParameter> ParametersMap;
	typedef std::unordered_map<std::string, std::string> ObsParamsList;

	template <typename M>
	class CSGObjectBase<M>::Self
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

} /* namespace shogun  */

using namespace shogun;

template <typename M>
CSGObjectBase<M>::CSGObjectBase(const CSGObjectBase<M>& orig)
	: CSGObjectBase() {}

template <typename M>
CSGObjectBase<M>::CSGObjectBase()
    : self(), param_obs_list(), house_keeper(M::template mutate<HouseKeeper>()),
      io(house_keeper.io)
{
	init();
	SG_SGCDEBUG("SGObject created (%p)\n", this)
}

template <typename M>
CSGObjectBase<M>::~CSGObjectBase()
{
	delete m_parameters;
	delete m_model_selection_parameters;
	delete m_gradient_parameters;
	delete m_subject_params;
	delete m_observable_params;
	delete m_subscriber_params;
}

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;

template <typename M>
void CSGObjectBase<M>::list_memory_allocs()
{
	shogun::list_memory_allocs();
}
#endif

template <typename M>
void CSGObjectBase<M>::update_parameter_hash()
{
	SG_DEBUG("entering\n")

	uint32_t carry=0;
	uint32_t length=0;

	m_hash=0;
	get_parameter_incremental_hash(m_hash, carry, length);
	m_hash=CHash::FinalizeIncrementalMurmurHash3(m_hash, carry, length);

	SG_DEBUG("leaving\n")
}

template <typename M>
bool CSGObjectBase<M>::parameter_hash_changed()
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

template <typename M>
void CSGObjectBase<M>::print_serializable(const char* prefix)
{
	SG_PRINT("\n%s\n================================================================================\n", house_keeper.get_name())
	m_parameters->print(prefix);
}

template <typename M>
bool CSGObjectBase<M>::save_serializable(CSerializableFile* file,
		const char* prefix)
{
	SG_DEBUG("START SAVING CSGObject '%s'\n", house_keeper.get_name())
	try
	{
		save_serializable_pre();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::save_serializable_pre(): ShogunException: "
				   "%s\n", prefix, house_keeper.get_name(), e.what());
		return false;
	}

	if (!m_save_pre_called)
	{
		SG_SWARNING("%s%s::save_serializable_pre(): Implementation "
				   "error: BASE_CLASS::SAVE_SERIALIZABLE_PRE() not "
				   "called!\n", prefix, house_keeper.get_name());
		return false;
	}

	if (!m_parameters->save(file, prefix))
		return false;

	try
	{
		save_serializable_post();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::save_serializable_post(): ShogunException: "
				   "%s\n", prefix, house_keeper.get_name(), e.what());
		return false;
	}

	if (!m_save_post_called)
	{
		SG_SWARNING("%s%s::save_serializable_post(): Implementation "
				   "error: BASE_CLASS::SAVE_SERIALIZABLE_POST() not "
				   "called!\n", prefix, house_keeper.get_name());
		return false;
	}

	if (prefix == NULL || *prefix == '\0')
		file->close();

	SG_DEBUG("DONE SAVING CSGObject '%s' (%p)\n", house_keeper.get_name(), this)

	return true;
}

template <typename M>
bool CSGObjectBase<M>::load_serializable(CSerializableFile* file,
		const char* prefix)
{
	REQUIRE(file != NULL, "Serializable file object should be != NULL\n");

	SG_DEBUG("START LOADING CSGObject '%s'\n", house_keeper.get_name())
	try
	{
		load_serializable_pre();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::load_serializable_pre(): ShogunException: "
				   "%s\n", prefix, house_keeper.get_name(), e.what());
		return false;
	}
	if (!m_load_pre_called)
	{
		SG_SWARNING("%s%s::load_serializable_pre(): Implementation "
				   "error: BASE_CLASS::LOAD_SERIALIZABLE_PRE() not "
				   "called!\n", prefix, house_keeper.get_name());
		return false;
	}

	if (!m_parameters->load(file, prefix))
		return false;

	try
	{
		load_serializable_post();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::load_serializable_post(): ShogunException: "
		            "%s\n", prefix, house_keeper.get_name(), e.what());
		return false;
	}

	if (!m_load_post_called)
	{
		SG_SWARNING("%s%s::load_serializable_post(): Implementation "
		            "error: BASE_CLASS::LOAD_SERIALIZABLE_POST() not "
		            "called!\n", prefix, house_keeper.get_name());
		return false;
	}
	SG_DEBUG("DONE LOADING CSGObject '%s' (%p)\n", house_keeper.get_name(), this)

	return true;
}

template <typename M>
void CSGObjectBase<M>::load_serializable_pre() throw (ShogunException)
{
	m_load_pre_called = true;
}

template <typename M>
void CSGObjectBase<M>::load_serializable_post() throw (ShogunException)
{
	m_load_post_called = true;
}

template <typename M>
void CSGObjectBase<M>::save_serializable_pre() throw (ShogunException)
{
	m_save_pre_called = true;
}

template <typename M>
void CSGObjectBase<M>::save_serializable_post() throw (ShogunException)
{
	m_save_post_called = true;
}

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;
#endif

template <typename M>
void CSGObjectBase<M>::init()
{
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
	{
		int32_t idx=sg_mallocs->index_of(this);
		if (idx>-1)
		{
			MemoryBlock* b=sg_mallocs->get_element_ptr(idx);
			b->set_sgobject();
		}
	}
#endif

	m_parameters = new Parameter();
	m_model_selection_parameters = new Parameter();
	m_gradient_parameters=new Parameter();
	m_load_pre_called = false;
	m_load_post_called = false;
	m_save_pre_called = false;
	m_save_post_called = false;
	m_hash = 0;

	m_subject_params = new SGSubject();
	m_observable_params = new SGObservable(m_subject_params->get_observable());
	m_subscriber_params = new SGSubscriber(m_subject_params->get_subscriber());
	m_next_subscription_index = 0;

	watch_method("num_subscriptions", &CSGObjectBase<M>::get_num_subscriptions);
}

template <typename M>
void CSGObjectBase<M>::print_modsel_params()
{
	SG_PRINT("parameters available for model selection for %s:\n", house_keeper.get_name())

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

template <typename M>
SGStringList<char> CSGObjectBase<M>::get_modelsel_names()
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

template <typename M>
char* CSGObjectBase<M>::get_modsel_param_descr(const char* param_name)
{
	index_t index=get_modsel_param_index(param_name);

	if (index<0)
	{
		SG_ERROR("There is no model selection parameter called \"%s\" for %s",
				param_name, house_keeper.get_name());
	}

	return m_model_selection_parameters->get_parameter(index)->m_description;
}

template <typename M>
index_t CSGObjectBase<M>::get_modsel_param_index(const char* param_name)
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

template <typename M>
void CSGObjectBase<M>::get_parameter_incremental_hash(uint32_t& hash, uint32_t& carry,
		uint32_t& total_length)
{
	for (index_t i=0; i<m_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_parameters->get_parameter(i);

		SG_DEBUG("Updating hash for parameter %s.%s\n", house_keeper.get_name(), p->m_name);

		if (p->m_datatype.m_ptype == PT_SGOBJECT)
		{
			if (p->m_datatype.m_ctype == CT_SCALAR)
			{
				Derived* child = *((Derived**)(p->m_parameter));

				if (child)
				{
					child->get_parameter_incremental_hash(hash, carry,
							total_length);
				}
			}
			else if (p->m_datatype.m_ctype==CT_VECTOR ||
					p->m_datatype.m_ctype==CT_SGVECTOR)
			{
				Derived** child=(*(Derived***)(p->m_parameter));

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

template <typename M>
void CSGObjectBase<M>::build_gradient_parameter_dictionary(CMap<TParameter*, Derived*>* dict)
{
	for (index_t i=0; i<m_gradient_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_gradient_parameters->get_parameter(i);
		dict->add(p, (Derived*) this);
	}

	for (index_t i=0; i<m_model_selection_parameters->get_num_parameters(); i++)
	{
		TParameter* p=m_model_selection_parameters->get_parameter(i);
		Derived* child=*(Derived**)(p->m_parameter);

		if ((p->m_datatype.m_ptype == PT_SGOBJECT) &&
				(p->m_datatype.m_ctype == CT_SCALAR) &&	child)
		{
			child->build_gradient_parameter_dictionary(dict);
		}
	}
}

template <typename M>
typename CSGObjectBase<M>::Derived* CSGObjectBase<M>::clone() const
{
	SG_DEBUG("Starting to clone %s at %p.\n", house_keeper.get_name(), this);
	SG_DEBUG("Constructing an empty instance of %s.\n", house_keeper.get_name());
	Derived* clone = create_empty();
	SG_DEBUG("Empty instance of %s created at %p.\n", house_keeper.get_name(), clone);

	REQUIRE(
	    clone, "Could not create empty instance of %s. The reason for "
	          "this usually is that get_name() of the class returns something "
	          "wrong, that a class has a wrongly set generic type, or that it "
	          "lies outside the main source tree and does not have "
	          "CSGObject::create_empty() overridden.\n",
	    house_keeper.get_name());

	for (const auto &it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.cloneable())
		{
			SG_SDEBUG(
			    "Skipping clone of %s::%s of type %s.\n", this->house_keeper.get_name(),
			    tag.name().c_str(), own.type().c_str());
			continue;
		}

		SG_SDEBUG(
			"Cloning parameter %s::%s of type %s.\n", this->house_keeper.get_name(),
			tag.name().c_str(), own.type().c_str());

		clone->get_parameter(tag).get_value().clone_from(own);
	}

	SG_DEBUG("Done cloning %s at %p, new object at %p.\n", house_keeper.get_name(), this, clone);
	return clone;
}

template <typename M>
void CSGObjectBase<M>::create_parameter(
	const BaseTag& _tag, const AnyParameter& parameter)
{
	self->create(_tag, parameter);
}

template <typename M>
void CSGObjectBase<M>::update_parameter(const BaseTag& _tag, const Any& value)
{
	if (!self->map[_tag].get_properties().has_property(
            ParameterProperties::READONLY))
		self->update(_tag, value);
	else
	{
		SG_ERROR(
		    "%s::%s is marked as read-only and cannot be modified", house_keeper.get_name(),
		    _tag.name().c_str());
	}
	self->map[_tag].get_properties().remove_property(ParameterProperties::AUTO);
}

template <typename M>
AnyParameter CSGObjectBase<M>::get_parameter(const BaseTag& _tag) const
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
bool CSGObjectBase<M>::has_parameter(const BaseTag& _tag) const
{
	return self->has(_tag);
}

template <typename M>
void CSGObjectBase<M>::subscribe(ParameterObserver* obs)
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

template <typename M>
void CSGObjectBase<M>::unsubscribe(ParameterObserver* obs)
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

template <typename M>
void CSGObjectBase<M>::observe(const Some<ObservedValue> value) const
{
	m_subscriber_params->on_next(value);
}

template <typename M>
class CSGObjectBase<M>::ParameterObserverList
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

template <typename M>
void CSGObjectBase<M>::register_observable(
	const std::string& name, const std::string& description)
{
	param_obs_list->register_param(name, description);
}

template <typename M>
std::vector<std::string> CSGObjectBase<M>::observable_names()
{
	std::vector<std::string> list;
	std::transform(
	    param_obs_list->get_list().begin(), param_obs_list->get_list().end(),
	    list.begin(), [](auto const& x) { return x.first; });
	return list;
}

template <typename M>
bool CSGObjectBase<M>::has(const std::string& name) const
{
	return has_parameter(BaseTag(name));
}

class ToStringVisitor : public AnyVisitor
{
public:
	ToStringVisitor(std::stringstream* ss) : AnyVisitor(), m_stream(ss)
	{
	}

	virtual void on(bool* v)
	{
		stream() << (*v ? "true" : "false");
	}
	virtual void on(int32_t* v)
	{
		stream() << *v;
	}
	virtual void on(int64_t* v)
	{
		stream() << *v;
	}
	virtual void on(float* v)
	{
		stream() << *v;
	}
	virtual void on(double* v)
	{
		stream() << *v;
	}
	virtual void on(long double* v)
	{
		stream() << *v;
	}
	virtual void on(CSGObject** v)
	{
		if (*v)
		{
			stream() << (*v)->get_name() << "(...)";
		}
		else
		{
			stream() << "null";
		}
	}
	virtual void on(SGVector<int>* v)
	{
		to_string(v);
	}
	virtual void on(SGVector<float>* v)
	{
		to_string(v);
	}
	virtual void on(SGVector<double>* v)
	{
		to_string(v);
	}
	virtual void on(SGMatrix<int>* mat)
	{
		to_string(mat);
	}
	virtual void on(SGMatrix<float>* mat)
	{
		to_string(mat);
	}
	virtual void on(SGMatrix<double>* mat)
	{
		to_string(mat);
	}

private:
	std::stringstream& stream()
	{
		return *m_stream;
	}

	template <class T>
	void to_string(SGMatrix<T>* m)
	{
		if (m)
		{
			stream() << "Matrix<" << demangled_type<T>() << ">(" << m->num_rows
			         << "," << m->num_cols << "): [";
			for (auto col : range(m->num_cols))
			{
				stream() << "[";
				for (auto row : range(m->num_rows))
				{
					stream() << (*m)(row, col);
					if (row < m->num_rows - 1)
						stream() << ",";
				}
				stream() << "]";
				if (col < m->num_cols)
					stream() << ",";
			}
			stream() << "]";
		}
	}

	template <class T>
	void to_string(SGVector<T>* v)
	{
		if (v)
		{
			stream() << "Vector<" << demangled_type<T>() << ">(" << v->vlen
			         << "): [";
			for (auto i : range(v->vlen))
			{
				stream() << (*v)[i];
				if (i < v->vlen - 1)
					stream() << ",";
			}
			stream() << "]";
		}
	}

private:
	std::stringstream* m_stream;
};

template <typename M>
std::string CSGObjectBase<M>::to_string() const
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
template <typename M>
std::map<std::string, std::shared_ptr<const AnyParameter>> CSGObjectBase<M>::get_params() const
{
	std::map<std::string, std::shared_ptr<const AnyParameter>> result;
	for (auto const& each: self->map) {
		result.emplace(each.first.name(), std::make_shared<const AnyParameter>(each.second));
	}
	return result;
}
#endif

template <typename M>
bool CSGObjectBase<M>::equals(const Derived* other) const
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
	for (const auto &it : self->map)
	{
		const BaseTag& tag = it.first;
		const Any& own = it.second.get_value();

		if (!own.visitable())
		{
			SG_SDEBUG(
			    "Skipping comparison of %s::%s of type %s as it is "
			    "non-visitable.\n",
			    this->house_keeper.get_name(), tag.name().c_str(), own.type().c_str());
			continue;
		}

		const Any& given = other->get_parameter(tag).get_value();

		SG_SDEBUG(
		    "Comparing parameter %s::%s of type %s.\n", this->house_keeper.get_name(),
		    tag.name().c_str(), own.type().c_str());
		if (own != given)
		{
			if (house_keeper.io->get_loglevel() <= MSG_DEBUG)
			{
				std::stringstream ss;
				std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));

				ss << "Own parameter " << this->house_keeper.get_name() << "::" << tag.name()
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

	SG_SDEBUG("All parameters of %s equal.\n", this->house_keeper.get_name());
	return true;
}

template <typename M>
typename CSGObjectBase<M>::Derived* CSGObjectBase<M>::create_empty() const
{
	Derived* object = create(this->house_keeper.get_name(), this->house_keeper.get_generic());
	SG_REF(object);
	return object;
}

template <typename M>
void CSGObjectBase<M>::init_auto_params()
{
	auto params = self->filter(ParameterProperties::AUTO);
	for (const auto& param : params)
	{
		update_parameter(param.first, param.second.get_init_function()->operator()());
	}
}

template <typename M>
typename CSGObjectBase<M>::Derived* CSGObjectBase<M>::get(const std::string& name, index_t index) const
{
	auto* result = sgo_details::get_by_tag((Derived*) this, name, std::move(sgo_details::GetByNameIndex(index)));
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
typename CSGObjectBase<M>::Derived* CSGObjectBase<M>::get(const std::string& name, std::nothrow_t) const
	noexcept
{
	return sgo_details::get_by_tag((Derived*) this, name, std::move(sgo_details::GetByName()));
}

template <typename M>
typename CSGObjectBase<M>::Derived* CSGObjectBase<M>::get(const std::string& name) const noexcept(false)
{
	if (!has(name))
	{
		SG_ERROR("Parameter %s::%s does not exist.\n", house_keeper.get_name(), name.c_str())
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

template <typename M>
std::string CSGObjectBase<M>::string_enum_reverse_lookup(
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

ObservedValue::ObservedValue(const int64_t step, const std::string& name)
    : CSGObject(), m_step(step), m_name(name), m_any_value(Any())
{
	SG_ADD(&m_step, "step", "Step");
	this->watch_param(
	    "name", &m_name, AnyParameterProperties("Name of the observed value"));
}

// ugly explicit template specialization
namespace shogun
{
	template class CSGObjectBase<mutator<CSGObject, CSGObjectBase, CSGObjectBase, HouseKeeper>>;
	template class HouseKeeper<mutator<CSGObject, HouseKeeper, CSGObjectBase, HouseKeeper>>;
}