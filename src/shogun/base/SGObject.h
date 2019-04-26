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
#include <shogun/lib/tag.h>
#include <shogun/util/mixins.h>
#include <shogun/base/mixins/HouseKeeper.h>

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
class Parameter;
class CSerializableFile;
class ObservedValue;
class ParameterObserver;
class CDynamicObjectArray;

#define IGNORE_IN_CLASSLIST

#ifndef SWIG
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace sgo_details
{
template <typename T1, typename T2>
bool dispatch_array_type(const CSGObject* obj, const std::string& name,
		T2&& lambda);
}
#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // SWIG

template <class T, class K> class CMap;

struct TParameter;
template <class T> class DynArray;
template <class T> class SGStringList;

using stringToEnumMapType = std::unordered_map<std::string, std::unordered_map<std::string, machine_int_t>>;

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


template <typename M>
IGNORE_IN_CLASSLIST class CSGObjectBase : public mixin<M, requires<HouseKeeper>>
{
	using Derived = typename M::derived_t;
public:
	/** Definition of observed subject */
	typedef rxcpp::subjects::subject<Some<ObservedValue>> SGSubject;
	/** Definition of observable */
	typedef rxcpp::observable<Some<ObservedValue>,
		                      rxcpp::dynamic_observable<Some<ObservedValue>>>
		SGObservable;
	/** Definition of subscriber */
	typedef rxcpp::subscriber<
		Some<ObservedValue>,
		rxcpp::observer<Some<ObservedValue>, void, void, void, void>>
		SGSubscriber;

	/** default constructor */
	CSGObjectBase();

	/** copy constructor */
	CSGObjectBase(const CSGObjectBase<M>& orig);

	/** destructor */
	virtual ~CSGObjectBase();


#ifdef TRACE_MEMORY_ALLOCS
	static void list_memory_allocs();
#endif

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
	void build_gradient_parameter_dictionary(CMap<TParameter*, Derived*>* dict);

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
	bool has(const std::string& name) const noexcept(true)
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
	template <typename T,
		      typename std::enable_if_t<!is_string<T>::value>* = nullptr>
	void put(const Tag<T>& _tag, const T& value) noexcept(false);

	/** Setter for a class parameter that has values of type string,
	 * identified by a Tag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name and type information of parameter
	 * @param value value of the parameter
	 */
	template <typename T,
		      typename std::enable_if_t<is_string<T>::value>* = nullptr>
	void put(const Tag<T>& _tag, const T& value) noexcept(false)
	{
	    std::string val_string(value);

		if (m_string_to_enum_map.find(_tag.name()) == m_string_to_enum_map.end())
		{
			SG_ERROR(
					"There are no options for parameter %s::%s", house_keeper.get_name(),
					_tag.name().c_str());
		}

		auto string_to_enum = m_string_to_enum_map[_tag.name()];

		if (string_to_enum.find(val_string) == string_to_enum.end())
		{
			SG_ERROR(
					"Illegal option '%s' for parameter %s::%s",
                    val_string.c_str(), house_keeper.get_name(), _tag.name().c_str());
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
			value, "Cannot add to %s::%s, no object provided.\n", house_keeper.get_name(),
			name.c_str());

		auto push_back_lambda = [&value](auto& array) {
			array.push_back(value);
		};

		auto derived = static_cast<const Derived *>(this);
		if (sgo_details::dispatch_array_type<T>(derived, name, push_back_lambda))
			return;

		SG_ERROR(
		    "Cannot add object %s to array parameter %s::%s of type %s.\n",
		    value->get_name(), house_keeper.get_name(), name.c_str(),
			demangled_type<T>().c_str());
	}

#ifndef SWIG
	/** Typed array getter for an object array class parameter of a Shogun base
	* class
	* type, identified by a name and an index.
	*
	* Returns nullptr if parameter of desired type does not exist.
	*
	* @param name name of the parameter array
	* @param index index of the element in the array
	* @return desired element
	*/
	template <class T,
		      class X = typename std::enable_if<is_sg_base<T>::value>::type>
	T* get(const std::string& name, index_t index, std::nothrow_t) const
	{
		Derived* result = nullptr;

		auto get_lambda = [&index, &result](auto& array) {
			result = array.at(index);
		};

		auto derived = static_cast<const Derived *>(this);
		if (sgo_details::dispatch_array_type<T>(derived, name, get_lambda))
		{
			ASSERT(result);
			// guard against mixed types in the array
			return result->template as<T>();
		}

		return nullptr;
	}

	template <class T,
		      class X = typename std::enable_if<is_sg_base<T>::value>::type>
	T* get(const std::string& name, index_t index) const
	{
		auto result = this->get<T>(name, index, std::nothrow);
		if (!result)
		{
			SG_ERROR(
				"Could not get array parameter %s::%s[%d] of type %s\n",
				house_keeper.get_name(), name.c_str(), index, demangled_type<T>().c_str());
		}
		return result;
	};
#endif

	/** Untyped getter for an object class parameter, identified by a name.
	 * Will attempt to get specified object of appropriate internal type.
	 * If this is not possible it will raise a ShogunException.
	 *
	 * @param name name of the parameter
	 * @return object parameter
	 */
	Derived* get(const std::string& name) const noexcept(false);

	/** Untyped getter for an object class parameter, identified by a name.
	 * Does not throw an error if class parameter object cannot be casted
	 * to appropriate internal type.
	 *
	 * @param name name of the parameter
	 * @return object parameter
	 */
	Derived* get(const std::string& name, std::nothrow_t) const noexcept;

	/** Untyped getter for an object array class parameter, identified by a name
	 * and an index.
	 * Will attempt to get specified object of appropriate internal type.
	 * If this is not possible it will raise a ShogunException.
	 *
	 * @param name name of the parameter
	 * @index index of the parameter
	 * @return object parameter
	 */
	Derived* get(const std::string& name, index_t index) const;

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
		              Derived, typename std::remove_pointer<T>::type>::value,
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
				house_keeper.get_name(), _tag.name().c_str(), exc.actual().c_str(),
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
			const Any value = get_parameter(_tag).get_value();
			try
			{
				return any_cast<T>(value);
			}
			catch (const TypeMismatchException& exc)
			{
				SG_ERROR(
					"Cannot get parameter %s::%s of type %s, incompatible "
					"requested type %s or there are no options for parameter "
					"%s::%s.\n",
					house_keeper.get_name(), _tag.name().c_str(), exc.actual().c_str(),
					exc.expected().c_str(), house_keeper.get_name(), _tag.name().c_str());
			}
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
	void subscribe(ParameterObserver* obs);

	/**
	 * Detach an observer from the current SGObject.
	 * @param subscription_index the index obtained by calling the subscribe
	 * procedure
	 */
	void unsubscribe(ParameterObserver* obs);

	/** Print to stdout a list of observable parameters */
	std::vector<std::string> observable_names();

	/** Get string to enum mapping */
	stringToEnumMapType get_string_to_enum_map() const
	{
		return m_string_to_enum_map;
	}

protected:
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
	virtual bool equals(const Derived* other) const;

	/** Creates a clone of the current object. This is done via recursively
	 * traversing all parameters, which corresponds to a deep copy.
	 * Calling equals on the cloned object always returns true although none
	 * of the memory of both objects overlaps.
	 *
	 * @return an identical copy of the given object, which is disjoint in memory.
	 * NULL if the clone fails. Note that the returned object is SG_REF'ed
	 */
	virtual Derived* clone() const;

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
	virtual Derived* create_empty() const;

	/** Initialises all parameters with ParameterProperties::AUTO flag */
	void init_auto_params();

private:
	void set_global_objects();
	void unset_global_objects();
	void init();

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
	 * Return total subscriptions
	 * @return total number of subscriptions
	 */
	index_t get_num_subscriptions() const
	{
		return static_cast<index_t>(m_subscriptions.size());
	}

	/**
	 * Observe a parameter value and emit them to observer.
	 * @param value Observed parameter's value
	 */
	void observe(const Some<ObservedValue> value) const;

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
	 * Register which params this object can emit.
	 * @param name the param name
	 * @param type the param type
	 * @param description a user oriented description
	 */
	void register_observable(
		const std::string& name, const std::string& description);

/**
 * Get the current step for the observed values.
 */
#ifndef SWIG
		SG_FORCED_INLINE int64_t get_step() const
		{
			int64_t step = -1;
			Tag<int64_t> tag("current_iteration");
			if (has(tag))
			{
				step = get(tag);
			}
			return step;
		}
#endif

		/** mapping from strings to enum for SWIG interface */
		stringToEnumMapType m_string_to_enum_map;

	public:
		/** parameters */
		Parameter* m_parameters;

		/** model selection parameters */
		Parameter* m_model_selection_parameters;

		/** parameters wrt which we can compute gradients */
		Parameter* m_gradient_parameters;

		/** Hash of parameter values*/
		uint32_t m_hash;

	private:
		bool m_load_pre_called;
		bool m_load_post_called;
		bool m_save_pre_called;
		bool m_save_post_called;

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
		SGIO*& io;
	};


	IGNORE_IN_CLASSLIST class CSGObject : public composition<CSGObject, CSGObjectBase, HouseKeeper>
	{
	public:
		virtual ~CSGObject() {};

	public:
		// to resolve naming conflict
		SGIO*& io = mixin_t<shogun::HouseKeeper>::io;
	};

#ifndef SWIG
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace sgo_details
{
template <typename T1, typename T2>
bool dispatch_array_type(const CSGObject* obj, const std::string& name, T2&& lambda)
{
	Tag<CDynamicObjectArray*> tag_array_sg(name);
	if (obj->has(tag_array_sg))
	{
		auto dispatched = obj->get(tag_array_sg);
		lambda(*dispatched); // is stored as a pointer
		return true;
	}

	Tag<std::vector<T1*>> tag_vector(name);
	if (obj->has(tag_vector))
	{
		auto dispatched = obj->get(tag_vector);
		lambda(dispatched);
		return true;
	}

	return false;
}

struct GetByName
{
};

struct GetByNameIndex
{
	GetByNameIndex(index_t index) : m_index(index) {}
	index_t m_index;
};

template <typename T>
CSGObject* get_if_possible(const CSGObject* obj, const std::string& name, GetByName)
{
	return obj->has<T*>(name) ? obj->get<T*>(name) : nullptr;
}

template <typename T>
CSGObject* get_if_possible(const CSGObject* obj, const std::string& name, GetByNameIndex how)
{
	CSGObject* result = nullptr;
	result = obj->get<T>(name, how.m_index, std::nothrow);
	return result;
}

template<typename T>
CSGObject* get_dispatch_all_base_types(const CSGObject* obj, const std::string& name,
		T&& how)
{
	if (auto* result = get_if_possible<CKernel>(obj, name, how))
		return result;
	if (auto* result = get_if_possible<CFeatures>(obj, name, how))
		return result;
	if (auto* result = get_if_possible<CMachine>(obj, name, how))
		return result;
	if (auto* result = get_if_possible<CLabels>(obj, name, how))
		return result;
	if (auto* result = get_if_possible<CEvaluationResult>(obj, name, how))
		return result;

	return nullptr;
}

template<class T>
CSGObject* get_by_tag(const CSGObject* obj, const std::string& name,
		T&& how)
{
	return get_dispatch_all_base_types(obj, name, how);
}
}

#endif //DOXYGEN_SHOULD_SKIP_THIS
#endif //SWIG

template <class T>
class ObservedValueTemplated;

/**
 * Observed value which is emitted by algorithms.
 */
class ObservedValue : public CSGObject
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
			AnyParameterProperties(description, ParameterProperties::READONLY));
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

template <typename M>
template <typename T, typename std::enable_if_t<!is_string<T>::value>*>
void CSGObjectBase<M>::put(const Tag<T>& _tag, const T& value) noexcept(false)
{
	if (has_parameter(_tag))
	{
		auto parameter_value = get_parameter(_tag).get_value();
		if (!parameter_value.cloneable())
		{
			SG_ERROR(
				"Cannot put parameter %s::%s.\n", house_keeper.get_name(),
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
				house_keeper.get_name(), _tag.name().c_str(), exc.actual().c_str(),
				exc.expected().c_str());
		}
		house_keeper.ref_value(value);
		update_parameter(_tag, make_any(value));

		observe<T>(this->get_step(), _tag.name());
	}
	else
	{
		SG_ERROR(
			"Parameter %s::%s does not exist.\n", house_keeper.get_name(),
			_tag.name().c_str());
	}
}

template <typename M>
template <class T>
void CSGObjectBase<M>::observe(
	const int64_t step, const std::string& name, const std::string& description,
	const T value) const
{
	auto obs = some<ObservedValueTemplated<T>>(step, name, description, value);
	this->observe(obs);
}

template <typename M>
template <class T>
void CSGObjectBase<M>::observe(const int64_t step, const std::string& name) const
{
	auto param = this->get_parameter(BaseTag(name));
	auto obs = some<ObservedValueTemplated<T>>(
		step, name, any_cast<T>(param.get_value()), param.get_properties());
	this->observe(obs);
}
}
#endif // __SGOBJECT_H__
