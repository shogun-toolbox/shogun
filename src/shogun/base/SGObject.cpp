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
#include <shogun/lib/RefCount.h>

#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Version.h>
#include <shogun/io/SerializableFile.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/parameter_observers/ParameterObserverInterface.h>

#include <shogun/base/class_list.h>

#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>

#include <rxcpp/operators/rx-filter.hpp>
#include <rxcpp/rx-lite.hpp>

#include <algorithm>
#include <unordered_map>
#include <memory>

namespace shogun
{

	typedef std::map<BaseTag, AnyParameter> ParametersMap;
	typedef std::unordered_map<std::string,
	                           std::pair<SG_OBS_VALUE_TYPE, std::string>>
	    ObsParamsList;

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

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;

void CSGObject::list_memory_allocs()
{
	shogun::list_memory_allocs();
}
#endif

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

void CSGObject::print_serializable(const char* prefix)
{
	SG_PRINT("\n%s\n================================================================================\n", get_name())
	m_parameters->print(prefix);
}

bool CSGObject::save_serializable(CSerializableFile* file,
		const char* prefix)
{
	SG_DEBUG("START SAVING CSGObject '%s'\n", get_name())
	try
	{
		save_serializable_pre();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::save_serializable_pre(): ShogunException: "
				   "%s\n", prefix, get_name(),
				   e.get_exception_string());
		return false;
	}

	if (!m_save_pre_called)
	{
		SG_SWARNING("%s%s::save_serializable_pre(): Implementation "
				   "error: BASE_CLASS::SAVE_SERIALIZABLE_PRE() not "
				   "called!\n", prefix, get_name());
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
				   "%s\n", prefix, get_name(),
				   e.get_exception_string());
		return false;
	}

	if (!m_save_post_called)
	{
		SG_SWARNING("%s%s::save_serializable_post(): Implementation "
				   "error: BASE_CLASS::SAVE_SERIALIZABLE_POST() not "
				   "called!\n", prefix, get_name());
		return false;
	}

	if (prefix == NULL || *prefix == '\0')
		file->close();

	SG_DEBUG("DONE SAVING CSGObject '%s' (%p)\n", get_name(), this)

	return true;
}

bool CSGObject::load_serializable(CSerializableFile* file,
		const char* prefix)
{
	REQUIRE(file != NULL, "Serializable file object should be != NULL\n");

	SG_DEBUG("START LOADING CSGObject '%s'\n", get_name())
	try
	{
		load_serializable_pre();
	}
	catch (ShogunException& e)
	{
		SG_SWARNING("%s%s::load_serializable_pre(): ShogunException: "
				   "%s\n", prefix, get_name(),
				   e.get_exception_string());
		return false;
	}
	if (!m_load_pre_called)
	{
		SG_SWARNING("%s%s::load_serializable_pre(): Implementation "
				   "error: BASE_CLASS::LOAD_SERIALIZABLE_PRE() not "
				   "called!\n", prefix, get_name());
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
		            "%s\n", prefix, get_name(),
		            e.get_exception_string());
		return false;
	}

	if (!m_load_post_called)
	{
		SG_SWARNING("%s%s::load_serializable_post(): Implementation "
		            "error: BASE_CLASS::LOAD_SERIALIZABLE_POST() not "
		            "called!\n", prefix, get_name());
		return false;
	}
	SG_DEBUG("DONE LOADING CSGObject '%s' (%p)\n", get_name(), this)

	return true;
}

void CSGObject::load_serializable_pre() throw (ShogunException)
{
	m_load_pre_called = true;
}

void CSGObject::load_serializable_post() throw (ShogunException)
{
	m_load_post_called = true;
}

void CSGObject::save_serializable_pre() throw (ShogunException)
{
	m_save_pre_called = true;
}

void CSGObject::save_serializable_post() throw (ShogunException)
{
	m_save_post_called = true;
}

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;
#endif

void CSGObject::init()
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

CSGObject* CSGObject::clone()
{
	SG_DEBUG("Constructing an empty instance of %s\n", get_name());
	CSGObject* copy = create_empty();

	REQUIRE(
	    copy, "Could not create empty instance of %s. The reason for "
	          "this usually is that get_name() of the class returns something "
	          "wrong, that a class has a wrongly set generic type, or that it "
	          "lies outside the main source tree and does not have "
	          "CSGObject::create_empty() overridden.\n",
	    get_name());

	SG_DEBUG("Cloning all parameters of %s\n", get_name());
	if (!copy->clone_parameters(this))
	{
		SG_WARNING("Cloning parameters failed.\n");
	}
	else
	{
		SG_DEBUG("Done cloning.\n");
	}

	return copy;
}

bool CSGObject::clone_parameters(CSGObject* other)
{
	REQUIRE(other, "Provided instance must be non-empty.\n");
	index_t num_parameters = m_parameters->get_num_parameters();

	REQUIRE(other->m_parameters->get_num_parameters() == num_parameters,
		"Number of parameters of provided instance (%d) must match this instance (%d).\n",
		other->m_parameters->get_num_parameters(), num_parameters);
	REQUIRE(!strcmp(other->get_name(), get_name()),
		"Name of provided instance (%s) must match this instance (%s).\n",
		other->get_name(), get_name());

	for (index_t i=0; i<num_parameters; ++i)
	{
		SG_DEBUG("Cloning parameter \"%s\" at index %d into this instance\n",
				other->m_parameters->get_parameter(i)->m_name, i);

		if (!other->m_parameters->get_parameter(i)->copy(m_parameters->get_parameter(i)))
		{
			SG_WARNING("Cloning parameter \"%s\" (at index %d) of provided instance %s"
				" into parameter \"%s\" of this instance %s failed.\n",
				other->m_parameters->get_parameter(i)->m_name, i,
				other->get_name(), m_parameters->get_parameter(i)->m_name,
				get_name());
			return false;
		}
	}

	return true;
}

void CSGObject::create_parameter(
    const BaseTag& _tag, const AnyParameter& parameter)
{
	self->create(_tag, parameter);
}

void CSGObject::update_parameter(const BaseTag& _tag, const Any& value)
{
	self->update(_tag, value);
}

AnyParameter CSGObject::get_parameter(const BaseTag& _tag) const
{
	const auto& parameter = self->get(_tag);
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

void CSGObject::subscribe_to_parameters(ParameterObserverInterface* obs)
{
	auto sub = rxcpp::make_subscriber<TimedObservedValue>(
	    [obs](TimedObservedValue e) { obs->on_next(e); },
	    [obs](std::exception_ptr ep) { obs->on_error(ep); },
	    [obs]() { obs->on_complete(); });

	// Create an observable which emits values only if they are about
	// parameters selected by the observable.
	auto subscription = m_observable_params
	                        ->filter([obs](ObservedValue v) {
		                        return obs->filter(v.get_name());
		                    })
	                        .timestamp()
	                        .subscribe(sub);
}

void CSGObject::observe(const ObservedValue value)
{
	m_subscriber_params->on_next(value);
}

class CSGObject::ParameterObserverList
{
public:
	void register_param(
	    const std::string& name, const SG_OBS_VALUE_TYPE type,
	    const std::string& description)
	{
		m_list_obs_params[name] = std::make_pair(type, description);
	}

	std::string type_name(SG_OBS_VALUE_TYPE type)
	{
		std::string value;
		switch (type)
		{
		case TENSORBOARD:
			value = std::string("Tensorboard");
			break;
		case CROSSVALIDATION:
			value = std::string("CrossValidation");
			break;
		default:
			value = std::string("Unknown");
			break;
		}
		return value;
	}

	ObsParamsList get_list() const
	{
		return m_list_obs_params;
	}

private:
	/** List of observable parameters (name, description) */
	ObsParamsList m_list_obs_params;
};

void CSGObject::register_observable_param(
    const std::string& name, const SG_OBS_VALUE_TYPE type,
    const std::string& description)
{
	param_obs_list->register_param(name, type, description);
}

void CSGObject::list_observable_parameters()
{
	SG_INFO("List of observable parameters of object %s\n", get_name());
	SG_PRINT("------");
	for (auto const& x : param_obs_list->get_list())
	{
		SG_PRINT(
		    "%s [%s]: %s\n", x.first.c_str(),
		    param_obs_list->type_name(x.second.first).c_str(),
		    x.second.second.c_str());
	}
}

bool CSGObject::has(const std::string& name) const
{
	return has_parameter(BaseTag(name));
}

void CSGObject::ref_value(CSGObject* const* value)
{
	SG_REF(*value);
}

void CSGObject::ref_value(...)
{
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
	virtual void on(SGVector<int>*)
	{
		stream() << "[...]";
	}
	virtual void on(SGVector<float>*)
	{
		stream() << "[...]";
	}
	virtual void on(SGVector<double>*)
	{
		stream() << "[...]";
	}
	virtual void on(SGMatrix<int>*)
	{
		stream() << "[...]";
	}
	virtual void on(SGMatrix<float>*)
	{
		stream() << "[...]";
	}
	virtual void on(SGMatrix<double>* mat)
	{
		stream() << "Matrix(" << mat->num_rows << "," << mat->num_cols << ")";
	}

private:
	std::stringstream& stream()
	{
		return *m_stream;
	}

private:
	std::stringstream* m_stream;
};

std::string CSGObject::to_string() const
{
	std::stringstream ss;
	std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));
	ss << get_name();
	ss << "(";
	for (auto it = self->map.begin(); it != self->map.end(); ++it)
	{
		ss << it->first.name() << "=";
		it->second.get_value().visit(visitor.get());
		if (std::next(it) != (self->map.end()))
		{
			ss << ",";
		}
	}
	ss << ")";
	return ss.str();
}

std::vector<std::string> CSGObject::parameter_names() const
{
	std::vector<std::string> result;
	std::transform(self->map.cbegin(), self->map.cend(), std::back_inserter(result),
		// FIXME: const auto& each fails on gcc 4.8.4
		[](const std::pair<BaseTag, AnyParameter>& each) -> std::string { return each.first.name(); });
	return result;
}

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
	for (const auto it : self->map)
	{
		auto tag = it.first;
		auto own = it.second;
		auto given = other->get_parameter(tag);

		SG_SDEBUG(
		    "Comparing parameter %s::%s of type %s.\n", this->get_name(),
		    tag.name().c_str(), own.get_value().type().c_str());
		if (own != given)
		{
			if (io->get_loglevel() <= MSG_DEBUG)
			{
				std::stringstream ss;
				std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));

				ss << "Own parameter " << this->get_name() << "::" << tag.name()
				   << "=";
				own.get_value().visit(visitor.get());

				ss << " different from provided " << other->get_name()
				   << "::" << tag.name() << "=";
				given.get_value().visit(visitor.get());

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
