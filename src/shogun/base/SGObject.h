/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Giovanni De Toni, Jacob Walker, Thoralf Klein, Chiyuan Zhang,
 *          Fernando Iglesias, Sanuj Sharma, Roman Votyakov, Yuyu Zhang,
 *          Viktor Gal, Bjoern Esser, Evangelos Anagnostopoulos, Pan Deng,
 *          Gil Hoben
 */

#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include <shogun/base/AnyParameter.h>
#include <shogun/base/Version.h>
#include <shogun/base/base_types.h>
#include <shogun/base/macros.h>
#include <shogun/base/some.h>
#include <shogun/base/unique.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/RxCppHeader.h>
#include <shogun/lib/any.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/lib/exception/ShogunException.h>
#include <shogun/lib/parameter_observers/ObservedValue.h>
#include <shogun/lib/tag.h>

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

/** \namespace shogun
 * @brief all of classes and functions are contained in the shogun namespace
 */
namespace shogun
{
class RefCount;
class SGIO;
class Parallel;
class Parameter;
class CSerializableFile;
class ParameterObserverInterface;
class CDynamicObjectArray;

template <class T, class K> class CMap;

struct TParameter;
template <class T> class DynArray;
template <class T> class SGStringList;

using stringToEnumMapType = std::unordered_map<std::string, std::unordered_map<std::string, machine_int_t>>;

/*******************************************************************************
 * define reference counter macros
 ******************************************************************************/

#define SG_REF(x) { if (x) (x)->ref(); }
#define SG_UNREF(x) { if (x) { if ((x)->unref()==0) (x)=NULL; } }
#define SG_UNREF_NO_NULL(x) { if (x) { (x)->unref(); } }

/*******************************************************************************
 * Macros for registering parameter properties
 ******************************************************************************/

#define SG_ADD3(param, name, description)                                      \
	{                                                                          \
		this->m_parameters->add(param, name, description);                     \
		this->watch_param(name, param, AnyParameterProperties(description));   \
	}

#define SG_ADD4(param, name, description, param_properties)                    \
	{                                                                          \
		static_assert(                                                         \
		    !static_cast<bool>((param_properties)&ParameterProperties::AUTO),  \
		    "Expected a lambda when passing param with "                       \
		    "ParameterProperty::AUTO");                                        \
		AnyParameterProperties pprop =                                         \
		    AnyParameterProperties(description, param_properties);             \
		this->m_parameters->add(param, name, description);                     \
		this->watch_param(name, param, pprop);                                 \
		if (pprop.get_model_selection())                                       \
			this->m_model_selection_parameters->add(param, name, description); \
		if (pprop.get_gradient())                                              \
			this->m_gradient_parameters->add(param, name, description);        \
	}

#define SG_ADD5(param, name, description, param_properties, auto_init)         \
	{                                                                          \
		static_assert(                                                         \
		    static_cast<bool>((param_properties)&ParameterProperties::AUTO),   \
		    "Expected param to have ParameterProperty::AUTO");                 \
		AnyParameterProperties pprop =                                         \
		    AnyParameterProperties(description, param_properties);             \
		this->m_parameters->add(param, name, description);                     \
		this->watch_param(name, param, auto_init, pprop);                      \
		if (pprop.get_model_selection())                                       \
			this->m_model_selection_parameters->add(param, name, description); \
		if (pprop.get_gradient())                                              \
			this->m_gradient_parameters->add(param, name, description);        \
	}

#define SG_ADD(...) VARARG(SG_ADD, __VA_ARGS__)

/*******************************************************************************
 * End of macros for registering parameter properties
 ******************************************************************************/

/** @brief Class SGObject is the base class of all shogun objects.
 *
 * Apart from dealing with reference counting that is used to manage shogung
 * objects in memory (erase unused object, avoid cleaning objects when they are
 * still in use), it provides interfaces for:
 *
 * -# parallel - to determine the number of used CPUs for a method (cf. Parallel)
 * -# io - to output messages and general i/o (cf. IO)
 * -# version - to provide version information of the shogun version used (cf. Version)
 *
 * All objects can be cloned and compared (deep copy, recursively)
 */
class CSGObject
{
public:
	/** Definition of observed subject */
	typedef rxcpp::subjects::subject<ObservedValue> SGSubject;
	/** Definition of observable */
	typedef rxcpp::observable<ObservedValue,
		                      rxcpp::dynamic_observable<ObservedValue>>
		SGObservable;
	/** Definition of subscriber */
	typedef rxcpp::subscriber<
		ObservedValue, rxcpp::observer<ObservedValue, void, void, void, void>>
		SGSubscriber;

	/** default constructor */
	CSGObject();

	/** copy constructor */
	CSGObject(const CSGObject& orig);

	/** destructor */
	virtual ~CSGObject();

	/** increase reference counter
	 *
	 * @return reference count
	 */
	int32_t ref();

	/** display reference counter
	 *
	 * @return reference count
	 */
	int32_t ref_count();

	/** decrement reference counter and deallocate object if refcount is zero
	 * before or after decrementing it
	 *
	 * @return reference count
	 */
	int32_t unref();

#ifdef TRACE_MEMORY_ALLOCS
	static void list_memory_allocs();
#endif

	/** A shallow copy.
	 * All the SGObject instance variables will be simply assigned and SG_REF-ed.
	 */
	virtual CSGObject *shallow_copy() const;

	/** A deep copy.
	 * All the instance variables will also be copied.
	 */
	virtual CSGObject *deep_copy() const;

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name() const = 0;

	/** If the SGSerializable is a class template then TRUE will be
	 *  returned and GENERIC is set to the type of the generic.
	 *
	 *  @param generic set to the type of the generic if returning
	 *                 TRUE
	 *
	 *  @return TRUE if a class template.
	 */
	virtual bool is_generic(EPrimitiveType* generic) const;

	/** set generic type to T
	 */
	template<class T> void set_generic();

	/** Returns generic type.
	 * @return generic type of this object
	 */
	EPrimitiveType get_generic() const
	{
		return m_generic;
	}

	/** unset generic type
	 *
	 * this has to be called in classes specializing a template class
	 */
	void unset_generic();

	/** prints registered parameters out
	 *
	 *	@param prefix prefix for members
	 */
	virtual void print_serializable(const char* prefix="");

	/** Save this object to file.
	 *
	 * @param file where to save the object; will be closed during
	 * returning if PREFIX is an empty string.
	 * @param prefix prefix for members
	 * @return TRUE if done, otherwise FALSE
	 */
	virtual bool save_serializable(CSerializableFile* file,
			const char* prefix="");

	/** Load this object from file.  If it will fail (returning FALSE)
	 *  then this object will contain inconsistent data and should not
	 *  be used!
	 *
	 *  @param file where to load from
	 *  @param prefix prefix for members
	 *
	 *  @return TRUE if done, otherwise FALSE
	 */
	virtual bool load_serializable(CSerializableFile* file,
			const char* prefix="");

	/** set the io object
	 *
	 * @param io io object to use
	 */
	void set_global_io(SGIO* io);

	/** get the io object
	 *
	 * @return io object
	 */
	SGIO* get_global_io();

	/** set the parallel object
	 *
	 * @param parallel parallel object to use
	 */
	void set_global_parallel(Parallel* parallel);

	/** get the parallel object
	 *
	 * @return parallel object
	 */
	Parallel* get_global_parallel();

	/** set the version object
	 *
	 * @param version version object to use
	 */
	void set_global_version(Version* version);

	/** get the version object
	 *
	 * @return version object
	 */
	Version* get_global_version();

	/** @return vector of names of all parameters which are registered for model
	 * selection */
	SGStringList<char> get_modelsel_names();

	/** prints all parameter registered for model selection and their type */
	void print_modsel_params();

	/** Returns description of a given parameter string, if it exists. SG_ERROR
	 * otherwise
	 *
	 * @param param_name name of the parameter
	 * @return description of the parameter
	 */
	char* get_modsel_param_descr(const char* param_name);

	/** Returns index of model selection parameter with provided index
	 *
	 * @param param_name name of model selection parameter
	 * @return index of model selection parameter with provided name,
	 * -1 if there is no such
	 */
	index_t get_modsel_param_index(const char* param_name);

	/** Builds a dictionary of all parameters in SGObject as well of those
	 *  of SGObjects that are parameters of this object. Dictionary maps
	 *  parameters to the objects that own them.
	 *
	 * @param dict dictionary of parameters to be built.
	 */
	void build_gradient_parameter_dictionary(CMap<TParameter*, CSGObject*>* dict);

	/** Checks if object has a class parameter identified by a name.
	 *
	 * @param name name of the parameter
	 * @return true if the parameter exists with the input name
	 */
	bool has(const std::string& name) const;

	/** Checks if object has a class parameter identified by a Tag.
	 *
	 * @param tag tag of the parameter containing name and type information
	 * @return true if the parameter exists with the input tag
	 */
	template <typename T>
	bool has(const Tag<T>& tag) const
	{
		return has<T>(tag.name());
	}

	/** Checks if a type exists for a class parameter identified by a name.
	 *
	 * @param name name of the parameter
	 * @return true if the parameter exists with the input name and type
	 */
	template <typename T, typename U = void>
	bool has(const std::string& name) const throw(ShogunException)
	{
		BaseTag tag(name);
		if (!has_parameter(tag))
			return false;
		const Any value = get_parameter(tag).get_value();
		return value.has_type<T>();
	}

#ifndef SWIG
	/** Setter for a class parameter, identified by a Tag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name and type information of parameter
	 * @param value value of the parameter
	 */
	template <typename T, typename std::enable_if_t<!is_string<T>::value>* = nullptr>
	void put(const Tag<T>& _tag, const T& value) noexcept(false)
	{
		if (has_parameter(_tag))
		{
			auto parameter_value = get_parameter(_tag).get_value();
			if (!parameter_value.cloneable())
			{
				SG_ERROR(
					"Cannot put parameter %s::%s.\n", get_name(),
					_tag.name().c_str());
			}
			try
			{
				any_cast<T>(parameter_value);
			}
			catch (const TypeMismatchException& exc)
			{
				SG_ERROR(
					"Cannot put parameter %s::%s of type %s, incompatible "
					"provided type %s.\n",
					get_name(), _tag.name().c_str(), exc.actual().c_str(),
					exc.expected().c_str());
			}
			ref_value(value);
			update_parameter(_tag, make_any(value));
		}
		else
		{
			SG_ERROR(
				"Parameter %s::%s does not exist.\n", get_name(),
				_tag.name().c_str());
		}
	}

	/** Setter for a class parameter that has values of type string,
	 * identified by a Tag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name and type information of parameter
	 * @param value value of the parameter
	 */
	template <typename T, typename std::enable_if_t<is_string<T>::value>* = nullptr>
	void put(const Tag<T>& _tag, const T& value) noexcept(false)
	{
	    std::string val_string(value);

		if (m_string_to_enum_map.find(_tag.name()) == m_string_to_enum_map.end())
		{
			SG_ERROR(
					"There are no options for parameter %s::%s", get_name(),
					_tag.name().c_str());
		}

		auto string_to_enum = m_string_to_enum_map[_tag.name()];

		if (string_to_enum.find(val_string) == string_to_enum.end())
		{
			SG_ERROR(
					"Illegal option '%s' for parameter %s::%s",
                    val_string.c_str(), get_name(), _tag.name().c_str());
		}

		machine_int_t enum_value = string_to_enum[val_string];

		put(Tag<machine_int_t>(_tag.name()), enum_value);
	}
#endif

	/** Typed setter for an object class parameter of a Shogun base class type,
	 * identified by a name.
	 *
	 * @param name name of the parameter
	 * @param value value of the parameter
	 */
	template <class T,
		      class X = typename std::enable_if<is_sg_base<T>::value>::type,
		      class Z = void>
	void put(const std::string& name, T* value)
	{
		put(Tag<T*>(name), value);
	}

	/** Typed appender for an object class parameter of a Shogun base class
	* type, identified by a name.
	*
	* @param name name of the parameter
	* @param value value of the parameter
	*/
	template <class T,
		      class X = typename std::enable_if<is_sg_base<T>::value>::type>
	void add(const std::string& name, T* value)
	{
		REQUIRE(
			value, "Cannot add to %s::%s, no object provided.\n", get_name(),
			name.c_str());

		Tag<CDynamicObjectArray*> tag_array_sg(name);
		if (has(tag_array_sg))
		{
			auto array = get(tag_array_sg);
			if (!array)
			{
				SG_ERROR(
					"Cannot add object %s to parameters %s::%s, array "
					"is not instantiated.\n",
					value->get_name(), get_name(), name.c_str());
			}

			push_back(array, value);
			return;
		}

		Tag<std::vector<T*>> tag_vector(name);
		if (has(tag_vector))
		{
			// TODO this needs test once we have std::vector parameters
			SG_NOTIMPLEMENTED
			auto array = get(tag_vector);
			array.push_back(value);
			return;
		}

		SG_ERROR(
		    "Cannot add object %s to parameters %s::%s of type %s, there is no"
		    " such array of parameters.\n",
		    value->get_name(), get_name(), name.c_str(),
			demangled_type<T>().c_str());
	}

	/** Untyped getter for an object class parameter, identified by a name.
	 * Will attempt to get specified object of appropriate internal type.
	 * If this is not possible it will raise a ShogunException.
	 *
	 * @param name name of the parameter
	 * @return object parameter
	 */
	CSGObject* get(const std::string& name) const noexcept(false);

	/** Untyped getter for an object class parameter, identified by a name.
	 * Does not throw an error if class parameter object cannot be casted
	 * to appropriate internal type.
	 *
	 * @param name name of the parameter
	 * @return object parameter
	 */
	CSGObject* get(const std::string& name, std::nothrow_t) const noexcept;

#ifndef SWIG
	/** Typed setter for an object class parameter of a Shogun base class type,
	 * identified by a name.
	 *
	 * @param name name of the parameter
	 * @param value value of the parameter
	 */
	template <class T, class = typename std::enable_if_t<is_sg_base<T>::value>>
	void put(const std::string& name, Some<T> value)
	{
		put(name, value.get());
	}

	/** Typed appender for an object class parameter of a Shogun base class
	* type,
	* identified by a name.
	*
	* @param name name of the parameter
	* @param value value of the parameter
	*/
	template <class T, class = typename std::enable_if_t<is_sg_base<T>::value>>
	void add(const std::string& name, Some<T> value)
	{
		add(name, value.get());
	}
#endif // SWIG

	/** Typed setter for a non-object class parameter, identified by a name.
	 *
	 * @param name name of the parameter
	 * @param value value of the parameter along with type information
	 */
	template <typename T,
		      typename T2 = typename std::enable_if<
		          !std::is_base_of<
		              CSGObject, typename std::remove_pointer<T>::type>::value,
		          T>::type>
	void put(const std::string& name, T value)
	{
		put(Tag<T>(name), value);
	}

#ifndef SWIG
	/** Getter for a class parameter, identified by a Tag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name and type information of parameter
	 * @return value of the parameter identified by the input tag
	 */
	template <typename T, typename std::enable_if_t<!is_string<T>::value>* = nullptr>
	T get(const Tag<T>& _tag) const noexcept(false)
	{
		const Any value = get_parameter(_tag).get_value();
		try
		{
			return any_cast<T>(value);
		}
		catch (const TypeMismatchException& exc)
		{
			SG_ERROR(
				"Cannot get parameter %s::%s of type %s, incompatible "
				"requested type %s.\n",
				get_name(), _tag.name().c_str(), exc.actual().c_str(),
				exc.expected().c_str());
		}
		// we won't be there
		return any_cast<T>(value);
	}

	template <typename T, typename std::enable_if_t<is_string<T>::value>* = nullptr>
	T get(const Tag<T>& _tag) const noexcept(false)
	{
		if (m_string_to_enum_map.find(_tag.name()) == m_string_to_enum_map.end())
		{
			SG_ERROR(
					"There are no options for parameter %s::%s", get_name(),
					_tag.name().c_str());
		}
		return string_enum_reverse_lookup(_tag.name(), get<machine_int_t>(_tag.name()));
	}
#endif

	/** Getter for a class parameter, identified by a name.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param name name of the parameter
	 * @return value of the parameter corresponding to the input name and type
	 */
	template <typename T, typename U = void>
	T get(const std::string& name) const noexcept(false)
	{
		Tag<T> tag(name);
		return get(tag);
	}

	/** Returns string representation of the object that contains
	 * its name and parameters.
	 *
	 */
	virtual std::string to_string() const;

	/** Returns map of parameter names and AnyParameter pairs
	 * of the object.
	 *
	 */
#ifndef SWIG // SWIG should skip this part
	std::map<std::string, std::shared_ptr<const AnyParameter>> get_params() const;
#endif
	/** Specializes a provided object to the specified type.
	 * Throws exception if the object cannot be specialized.
	 *
	 * @param sgo object of CSGObject base type
	 * @return The requested type
	 */
	template<class T> static T* as(CSGObject* sgo)
	{
		REQUIRE(sgo, "No object provided!\n");
		return sgo->as<T>();
	}

	/** Specializes the object to the specified type.
	 * Throws exception if the object cannot be specialized.
	 *
	 * @return The requested type
	 */
	template<class T> T* as()
	{
		auto c = dynamic_cast<T*>(this);
		if (c)
			return c;

		SG_SERROR(
			"Object of type %s cannot be converted to type %s.\n",
			this->get_name(),
			demangled_type<T>().c_str());
		return nullptr;
	}
#ifndef SWIG
	/**
	  * Get parameters observable
	  * @return RxCpp observable
	  */
	SGObservable* get_parameters_observable()
	{
		return m_observable_params;
	};
#endif

	/** Subscribe a parameter observer to watch over params */
	void subscribe_to_parameters(ParameterObserverInterface* obs);

	/** Print to stdout a list of observable parameters */
	void list_observable_parameters();

	/** Get string to enum mapping */
	stringToEnumMapType get_string_to_enum_map() const
	{
		return m_string_to_enum_map;
	}

protected:
	template <typename T>
	CSGObject* get_sgobject_type_dispatcher(const std::string& name) const
	{
		if (has<T*>(name))
		{
			T* result = get<T*>(name);
			SG_REF(result)
			return (CSGObject*)result;
		}

		return nullptr;
	}

	/** Can (optionally) be overridden to pre-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
	 *  is called.
	 *
	 *  @exception ShogunException will be thrown if an error occurs.
	 */
	virtual void load_serializable_pre() throw (ShogunException);

	/** Can (optionally) be overridden to post-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST
	 *  is called.
	 *
	 *  @exception ShogunException will be thrown if an error occurs.
	 */
	virtual void load_serializable_post() throw (ShogunException);

	/** Can (optionally) be overridden to pre-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
	 *  is called.
	 *
	 *  @exception ShogunException will be thrown if an error occurs.
	 */
	virtual void save_serializable_pre() throw (ShogunException);

	/** Can (optionally) be overridden to post-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_POST
	 *  is called.
	 *
	 *  @exception ShogunException will be thrown if an error occurs.
	 */
	virtual void save_serializable_post() throw (ShogunException);

	/** Registers a class parameter which is identified by a tag.
	 * This enables the parameter to be modified by put() and retrieved by
	 * get().
	 * Parameters can be registered in the constructor of the class.
	 *
	 * @param _tag name and type information of parameter
	 * @param value value of the parameter
	 */
	template <typename T>
	void register_param(Tag<T>& _tag, const T& value)
	{
		create_parameter(_tag, AnyParameter(make_any(value)));
	}

	/** Registers a class parameter which is identified by a name.
	 * This enables the parameter to be modified by put() and retrieved by
	 * get().
	 * Parameters can be registered in the constructor of the class.
	 *
	 * @param name name of the parameter
	 * @param value value of the parameter along with type information
	 */
	template <typename T>
	void register_param(const std::string& name, const T& value)
	{
		BaseTag tag(name);
		create_parameter(tag, AnyParameter(make_any(value)));
	}

	/** Puts a pointer to some parameter into the parameter map.
	 *
	 * @param name name of the parameter
	 * @param value pointer to the parameter value
	 * @param properties properties of the parameter (e.g. if model selection is supported)
	 */
	template <typename T>
	void watch_param(
		const std::string& name, T* value,
		AnyParameterProperties properties = AnyParameterProperties())
	{
		BaseTag tag(name);
		create_parameter(tag, AnyParameter(make_any_ref(value), properties));
	}

	/** Puts a pointer to some parameter into the parameter map.
	 * The parameter is expected to be initialised at runtime
	 * using the provided lambda.
	 *
	 * @param name name of the parameter
	 * @param value pointer to the parameter value
	 * @param auto_init AutoInit object to initialise the value of the parameter
	 * @param properties properties of the parameter (e.g. if model selection is supported)
	 */
	template <typename T>
	void watch_param(
			const std::string& name, T* value,
			std::shared_ptr<params::AutoInit> auto_init,
			AnyParameterProperties properties = AnyParameterProperties())
	{
		BaseTag tag(name);
		create_parameter(tag, AnyParameter(make_any_ref(value), properties, std::move(auto_init)));
	}

	/** Puts a pointer to some parameter array into the parameter map.
	 *
	 * @param name name of the parameter array
	 * @param value pointer to the first element of the parameter array
	 * @param len number of elements in the array
	 * @param properties properties of the parameter (e.g. if model selection is
	 * supported)
	 */
	template <typename T, typename S>
	void watch_param(
		const std::string& name, T** value, S* len,
		AnyParameterProperties properties = AnyParameterProperties())
	{
		BaseTag tag(name);
		create_parameter(
			tag, AnyParameter(make_any_ref(value, len), properties));
	}

	/** Puts a pointer to some 2d parameter array (i.e. a matrix) into the
	 * parameter map.
	 *
	 * @param name name of the parameter array
	 * @param value pointer to the first element of the parameter array
	 * @param rows number of rows in the array
	 * @param cols number of columns in the array
	 * @param properties properties of the parameter (e.g. if model selection is
	 * supported)
	 */
	template <typename T, typename S>
	void watch_param(
		const std::string& name, T** value, S* rows, S* cols,
		AnyParameterProperties properties = AnyParameterProperties())
	{
		BaseTag tag(name);
		create_parameter(
			tag, AnyParameter(make_any_ref(value, rows, cols), properties));
	}

#ifndef SWIG
	/** Puts a pointer to a (lazily evaluated) function into the parameter map.
	 *
	 * @param name name of the parameter
	 * @param method pointer to the method
	 */
	template <typename T, typename S>
	void watch_method(const std::string& name, T (S::*method)() const)
	{
		BaseTag tag(name);
		AnyParameterProperties properties(
			"Dynamic parameter",
			ParameterProperties::HYPER |
			ParameterProperties::GRADIENT |
            ParameterProperties::MODEL);
		std::function<T()> bind_method =
			std::bind(method, dynamic_cast<const S*>(this));
		create_parameter(tag, AnyParameter(make_any(bind_method), properties));
	}
#endif

public:
	/** Updates the hash of current parameter combination */
	virtual void update_parameter_hash();

	/**
	 * @return whether parameter combination has changed since last update
	 */
	virtual bool parameter_hash_changed();

	/** Deep comparison of two objects.
	 *
	 * @param other object to compare with
	 * @return true if all parameters are equal
	 */
	virtual bool equals(const CSGObject* other) const;

	/** Creates a clone of the current object. This is done via recursively
	 * traversing all parameters, which corresponds to a deep copy.
	 * Calling equals on the cloned object always returns true although none
	 * of the memory of both objects overlaps.
	 *
	 * @return an identical copy of the given object, which is disjoint in memory.
	 * NULL if the clone fails. Note that the returned object is SG_REF'ed
	 */
	virtual CSGObject* clone();

	/**
	 * Looks up the option name of a parameter given the enum value.
	 *
	 * @param param the parameter name
	 * @param value the enum value to query
	 * @return the string representation of the enum (option name)
	 */
	std::string string_enum_reverse_lookup(
			const std::string& param, machine_int_t value) const;

protected:
	/** Returns an empty instance of own type.
	 *
	 * When inheriting from CSGObject from outside the main source tree (i.e.
	 * customized classes, or in a unit test), then this method has to be
	 * overloaded manually to return an empty instance.
	 * Shogun can only instantiate empty class instances from its source tree.
	 *
	 * @return empty instance of own type
	 */
	virtual CSGObject* create_empty() const;

	/** Initialises all parameters with ParameterProperties::AUTO flag */
	void init_auto_params();

private:
	void set_global_objects();
	void unset_global_objects();
	void init();

	/** Overloaded helper to increase reference counter */
	static void ref_value(CSGObject* value)
	{
		SG_REF(value);
	}

	/** Overloaded helper to increase reference counter
	 * Here a no-op for non CSGobject pointer parameters */
	template <typename T,
		      std::enable_if_t<
		          !std::is_base_of<
		              CSGObject, typename std::remove_pointer<T>::type>::value,
		          T>* = nullptr>
	static void ref_value(T value)
	{
	}

	/** Checks if object has a parameter identified by a BaseTag.
	 * This only checks for name and not type information.
	 * See its usage in has() and has<T>().
	 *
	 * @param _tag name information of parameter
	 * @return true if the parameter exists with the input tag
	 */
	bool has_parameter(const BaseTag& _tag) const;

	/** Creates a parameter identified by a BaseTag.
	 *
	 * @param _tag name information of parameter
	 * @param parameter parameter to be created
	 */
	void create_parameter(const BaseTag& _tag, const AnyParameter& parameter);

	/** Updates a parameter identified by a BaseTag.
	 *
	 * @param _tag name information of parameter
	 * @param value new value of parameter
	 */
	void update_parameter(const BaseTag& _tag, const Any& value);

	/** Getter for a class parameter, identified by a BaseTag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name information of parameter
	 * @return value of the parameter identified by the input tag
	 */
	AnyParameter get_parameter(const BaseTag& _tag) const;

	/** Gets an incremental hash of all parameters as well as the parameters of
	 * CSGObject children of the current object's parameters.
	 *
	 * @param hash the computed hash returned by reference
	 * @param carry value for Murmur3 incremental hash
	 * @param total_length total byte length of all hashed parameters so
	 * far. Byte length of parameters will be added to the total length
	 */
	void get_parameter_incremental_hash(uint32_t& hash, uint32_t& carry,
			uint32_t& total_length);

	class Self;
	Unique<Self> self;

	class ParameterObserverList;
	Unique<ParameterObserverList> param_obs_list;

protected:
	/**
	 * Observe a parameter value and emit them to observer.
	 * @param value Observed parameter's value
	 */
	void observe(const ObservedValue value);

	/**
	 * Register which params this object can emit.
	 * @param name the param name
	 * @param type the param type
	 * @param description a user oriented description
	 */
	void register_observable_param(
		const std::string& name,
		const std::string& description);

	/** mapping from strings to enum for SWIG interface */
	stringToEnumMapType m_string_to_enum_map;

private:
	void push_back(CDynamicObjectArray* array, CSGObject* value);

public:
	/** io */
	SGIO* io;

	/** parallel */
	Parallel* parallel;

	/** version */
	Version* version;

	/** parameters */
	Parameter* m_parameters;

	/** model selection parameters */
	Parameter* m_model_selection_parameters;

	/** parameters wrt which we can compute gradients */
	Parameter* m_gradient_parameters;

	/** Hash of parameter values*/
	uint32_t m_hash;

private:

	EPrimitiveType m_generic;
	bool m_load_pre_called;
	bool m_load_post_called;
	bool m_save_pre_called;
	bool m_save_post_called;

	RefCount* m_refcount;

	/** Subject used to create the params observer */
	SGSubject* m_subject_params;

	/** Parameter Observable */
	SGObservable* m_observable_params;

	/** Subscriber used to call onNext, onComplete etc.*/
	SGSubscriber* m_subscriber_params;
};
}
#endif // __SGOBJECT_H__
