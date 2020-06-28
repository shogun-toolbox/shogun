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
#include <shogun/base/constraint.h>
#include <shogun/base/macros.h>
#include <shogun/base/unique.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/RxCppHeader.h>
#include <shogun/lib/any.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/lib/exception/ShogunException.h>
#include <shogun/lib/tag.h>
#include <shogun/util/clone.h>

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <optional>

/** \namespace shogun
 * @brief all of classes and functions are contained in the shogun namespace
 */
namespace shogun
{
class Parallel;
class ParameterObserverInterface;
class ObservedValue;
class ParameterObserver;

template <class T>
class ObservedValueTemplated;
	namespace io
	{
		class Deserializer;
		class Serializer;
	}

#ifndef SWIG
#ifndef DOXYGEN_SHOULD_SKIP_THIS
	namespace sgo_details
	{
		template <typename T1, typename T2>
		bool dispatch_array_type(
		    const std::shared_ptr<const SGObject>& obj, std::string_view name,
		    T2&& lambda);
	}  // namespace sgo_details
#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // SWIG

using stringToEnumMapType = std::unordered_map<std::string_view, std::unordered_map<std::string_view, machine_int_t>>;

/*******************************************************************************
 * Macros for registering parameter properties
 ******************************************************************************/

#ifndef SWIG
SG_FORCED_INLINE const char* convert_string_to_char(std::string_view name)
{
    return name.data();
}

SG_FORCED_INLINE const char* convert_string_to_char(const char* name)
{
    return name;
}
#endif

#define SG_ADD3(param, name, description)                                      \
	{                                                                          \
		auto pprop =                                                           \
		    AnyParameterProperties(description, this->m_default_mask);         \
		this->watch_param(name, param, pprop);                                 \
}

#define SG_ADD4(param, name, description, param_properties)                    \
	{                                                                          \
		static_assert(                                                         \
		    !static_cast<bool>((param_properties)&ParameterProperties::AUTO),  \
		    "Expected a lambda when passing param with "                       \
		    "ParameterProperty::AUTO");                                        \
		auto mask = param_properties;                                          \
		mask |= this->m_default_mask;                                          \
		AnyParameterProperties pprop =                                         \
		    AnyParameterProperties(description, mask);                         \
		this->watch_param(name, param, pprop);                                 \
	}

#define SG_ADD5(                                                               \
    param, name, description, param_properties, auto_or_constraint)            \
	{                                                                          \
		auto mask = param_properties;                                          \
		mask |= this->m_default_mask;                                          \
		AnyParameterProperties pprop =                                         \
		    AnyParameterProperties(description, mask);                         \
		this->watch_param(name, param, auto_or_constraint, pprop);             \
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
class SGObject: public std::enable_shared_from_this<SGObject>
{
	template <typename ReturnType, typename CastType>
	struct ParameterGetterInterface
	{
		ReturnType& m_value;
	};

	template <typename ValueType>
	struct ParameterPutInterface
	{
		const ValueType& m_value;
	};
	
public:
	/** Definition of observed subject */
	typedef rxcpp::subjects::subject<std::shared_ptr<ObservedValue>> SGSubject;
	/** Definition of observable */
	typedef rxcpp::observable<std::shared_ptr<ObservedValue>,
		                      rxcpp::dynamic_observable<std::shared_ptr<ObservedValue>>>
		SGObservable;
	/** Definition of subscriber */
	typedef rxcpp::subscriber<
		std::shared_ptr<ObservedValue>,
		rxcpp::observer<std::shared_ptr<ObservedValue>, void, void, void, void>>
		SGSubscriber;

	using Parameters = std::map<std::string, std::shared_ptr<const AnyParameter>>;

	/** default constructor */
	SGObject();

	/** copy constructor */
	SGObject(const SGObject& orig);

	/** destructor */
	virtual ~SGObject();

	/** A deep copy.
	 * All the instance variables will also be copied.
	 */
	virtual std::shared_ptr<SGObject> deep_copy() const;

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

	virtual size_t hash() const;

	/** Save this object to file.
	 *
	 * @param ser where to save the object;
	 * @return TRUE if done, otherwise FALSE
	 */
	virtual bool serialize(const std::shared_ptr<io::Serializer>& ser);

	/** Load this object from file.  If it will fail (returning FALSE)
	 *  then this object will contain inconsistent data and should not
	 *  be used!
	 *
	 *  @param deser where to load from
	 *
	 *  @return TRUE if done, otherwise FALSE
	 */
	virtual bool deserialize(const std::shared_ptr<io::Deserializer>& deser);

	/** get the parallel object
	 *
	 * @return parallel object
	 */
	Parallel* get_global_parallel();

	/**
	 * Return the description of a registered parameter given its name
	 * @param name parameter's name
	 * @return description of the parameter as a string
	 */
#ifdef SWIG
	std::string get_description(const std::string& name) const;
#else
	std::string get_description(std::string_view name) const;
#endif
	/** Builds a dictionary of all parameters in SGObject as well of those
	 *  of SGObjects that are parameters of this object. Dictionary maps
	 *  parameters to the objects that own them.
	 *
	 * @param dict dictionary of parameters to be built.
	 */
	void build_gradient_parameter_dictionary(
 		    std::map<Parameters::value_type, std::shared_ptr<SGObject>>& dict);

	/** Checks if object has a class parameter identified by a name.
	 *
	 * @param name name of the parameter
	 * @return true if the parameter exists with the input name
	 */
#ifdef SWIG
	bool has(const std::string& name) const;
#else
	bool has(std::string_view name) const;
#endif
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

#ifndef SWIG
	/** Checks if a type exists for a class parameter identified by a name.
	 *
	 * @param name name of the parameter
	 * @return true if the parameter exists with the input name and type
	 */
	template <typename T, typename std::enable_if_t<!is_sg_base<T>::value>* = nullptr>
	bool has(std::string_view name) const noexcept(true)
	{
		BaseTag tag(name);
		if (!has_parameter(tag))
			return false;
		const auto& param = get_parameter(tag);
		const auto& value = param.get_value();
		if (param.get_properties().has_property(ParameterProperties::AUTO))
			return value.has_type<AutoValue<T>>();
		else
			return value.has_type<T>();
	}

	template <typename T, typename std::enable_if_t<is_sg_base<T>::value>* = nullptr>
	bool has(std::string_view name) const noexcept(true)
	{
		BaseTag tag(name);
		if (!has_parameter(tag))
			return false;
		const Any value = get_parameter(tag).get_value();
		return value.has_type<std::shared_ptr<T>>();
	}

	/** Setter for a class parameter, identified by a Tag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name and type information of parameter
	 * @param value value of the parameter
	 */
	template <typename T,
		      typename std::enable_if_t<!is_string<T>::value>* = nullptr>
	void put(const Tag<T>& _tag, const T& value)
	{
		const auto& parameter_value = get_parameter(_tag).get_value();

		if (!parameter_value.cloneable())
		{
			error(
				"Cannot put parameter {}::{}.", get_name(),
				_tag.name().c_str());
		}

		update_parameter(_tag, value);
	}

	/** Setter for a class parameter that has values of type string,
	 * identified by a Tag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name and type information of parameter
	 * @param value value of the parameter
	 */
	template <typename T,
		      typename std::enable_if_t<is_string<T>::value>* = nullptr>
	void put(const Tag<T>& _tag, const T& value)
	{
	    std::string val_string(value);

		if (m_string_to_enum_map.find(_tag.name()) == m_string_to_enum_map.end())
		{
			error(
					"There are no options for parameter {}::{}", get_name(),
					_tag.name().c_str());
		}

		auto string_to_enum = m_string_to_enum_map.at(_tag.name());

		if (string_to_enum.find(val_string) == string_to_enum.end())
		{
			error(
					"Illegal option '{}' for parameter {}::{}",
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
		      class X = typename std::enable_if_t<is_sg_base<T>::value>,
		      class Z = void>
#ifdef SWIG
	void put(const std::string& name, std::shared_ptr<T> value)
#else
	void put(std::string_view name, std::shared_ptr<T> value)
#endif
	{
		put(Tag<std::shared_ptr<T>>(name), value);
	}

	/** Typed appender for an object class parameter of a Shogun base class
	* type, identified by a name.
	*
	* @param name name of the parameter
	* @param value value of the parameter
	*/
	template <class T,
		      class X = typename std::enable_if_t<is_sg_base<T>::value>>
#ifdef SWIG
	void add(const std::string& name, std::shared_ptr<T> value)
#else
	void add(std::string_view name, std::shared_ptr<T> value)
#endif
	{
		require(
			value, "Cannot add to {}::{}, no object provided.", get_name(),
			name.data());
		Tag<std::vector<std::shared_ptr<T>>> tag_vector(name);
		auto dispatched = get(tag_vector);
		dispatched.push_back(value);
		update_parameter(BaseTag(name), dispatched, false);
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
		      class X = typename std::enable_if_t<is_sg_base<T>::value>>
	std::shared_ptr<T> get(std::string_view name, index_t index, std::nothrow_t) const
	{
		std::shared_ptr<SGObject> result;

		auto get_lambda = [&index, &result](auto& array) {
			result = array.at(index);
		};
		if (sgo_details::dispatch_array_type<T>(shared_from_this(), name, get_lambda))
		{
			ASSERT(result);
			// guard against mixed types in the array
			return result->as<T>();
		}

		return nullptr;
	}

	template <class T,
		      class X = typename std::enable_if_t<is_sg_base<T>::value>>
	std::shared_ptr<T> get(std::string_view name, index_t index) const
	{
		auto result = this->get<T>(name, index, std::nothrow);
		if (!result)
		{
			error(
				"Could not get array parameter {}::{}[{}] of type {}",
				get_name(), name.data(), index, demangled_type<T>().c_str());
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
#ifdef SWIG
	std::shared_ptr<SGObject> get(const std::string& name) const;
#else
	std::shared_ptr<SGObject> get(std::string_view name) const;
#endif

#ifndef SWIG
	/** Untyped getter for an object class parameter, identified by a name.
	 * Does not throw an error if class parameter object cannot be casted
	 * to appropriate internal type.
	 *
	 * @param name name of the parameter
	 * @return object parameter
	 */
	std::shared_ptr<SGObject> get(std::string_view name, std::nothrow_t) const noexcept;
#endif

	/** Untyped getter for an object array class parameter, identified by a name
	 * and an index.
	 * Will attempt to get specified object of appropriate internal type.
	 * If this is not possible it will raise a ShogunException.
	 *
	 * @param name name of the parameter
	 * @param index of the parameter
	 * @return object parameter
	 */
#ifdef SWIG
	std::shared_ptr<SGObject> get(const std::string& name, index_t index) const;
#else
	std::shared_ptr<SGObject> get(std::string_view name, index_t index) const;
#endif

	/** Typed setter for a non-object class parameter, identified by a name.
	 *
	 * @param name name of the parameter
	 * @param value value of the parameter along with type information
	 */
	template <typename T,
		 typename T2 = typename std::enable_if_t<
		          !std::is_base_of_v<
		              SGObject, typename std::remove_pointer_t<T>>, T>>
#ifdef SWIG
	void put(const std::string& name, T value)
#else
	void put(std::string_view name, T value)
#endif
	{
		if constexpr (std::is_enum<T>::value)
			put(Tag<machine_int_t>(name), static_cast<machine_int_t>(value));
		else
			put(Tag<T>(name), value);
	}

#ifndef SWIG
	/** Getter for a class parameter, identified by a Tag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name and type information of parameter
	 * @return value of the parameter identified by the input tag
	 */
	template <typename T>
	auto get(const Tag<T>& _tag) const
	{
		using ReturnType = std::conditional_t<is_sg_base<T>::value, std::shared_ptr<T>, T>;
		ReturnType result;
		
		const auto& param = get_parameter(_tag);

		if constexpr (is_string<T>::value)
		{
			if (m_string_to_enum_map.count(_tag.name()))
				return std::string(string_enum_reverse_lookup(_tag.name(), get<machine_int_t>(_tag.name())));
		}
		
		const auto& value = param.get_value();
		try
		{
			if (param.get_properties().has_property(ParameterProperties::CONSTFUNCTION))
			{
				ParameterGetterInterface<ReturnType, std::function<ReturnType()>> visitor{result};
				value.visit_with(&visitor);
			}
			else if (param.get_properties().has_property(ParameterProperties::AUTO))
			{
				ParameterGetterInterface<ReturnType, AutoValue<ReturnType>> visitor{result};
				value.visit_with(&visitor);
			}
			else
			{
				ParameterGetterInterface<ReturnType, ReturnType> visitor{result};
				value.visit_with(&visitor);
			}
		}
		catch (const std::bad_optional_access&)
		{
			const auto& heuristic = param.get_init_function();
			error("The value of parameter {}::{} is automatically computed using \"{}\" during model training, "
				  "and is currently not set. Either set a value or read value after training.",
				  get_name(),_tag.name(), heuristic->display_name());
		}
		catch (const std::logic_error&)
		{
			error(
				"Cannot get parameter {}::{} of type {}, incompatible with "
				"requested type {}.",
				get_name(), _tag.name().c_str(), value.type().c_str(),
				demangled_type<T>().c_str());
		}
		return result;
	}
#endif

	/** Getter for a class parameter, identified by a name.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param name name of the parameter
	 * @return value of the parameter corresponding to the input name and type
	 */
	template <typename T, class X = typename std::enable_if_t<!is_sg_base<T>::value>>
#ifdef SWIG
	T get(const std::string& name) const
#else
	T get(std::string_view name) const
#endif
	{
		Tag<T> tag(name);
		return get(tag);
	}

	/**
	 *
	 * @param name name of the parameter
	 * @return value of the parameter corresponding to the input name and type
	 */
#ifdef SWIG
	void run(const std::string& name) const
#else
	void run(std::string_view name) const
#endif
	{
		const auto tag = Tag<bool>(name);
		const auto& param = get_function(tag);
		if (!any_cast<bool>(param.get_value()))
		{
			error("Failed to run function {}::{}", get_name(), name.data());
		}
	}

#ifndef SWIG
	template <typename T,  typename std::enable_if_t<is_sg_base<T>::value>* = nullptr>
	std::shared_ptr<T> get(std::string_view name) const
	{
		Tag<T> tag(name);
		return get(tag);
	}
#endif
	/** Returns string representation of the object that contains
	 * its name and parameters.
	 *
	 */
	virtual std::string to_string() const;

#ifndef SWIG // SWIG should skip this part
	/** Returns map of parameter names and AnyParameter pairs
	 * of the object.
	 *
	 */
	Parameters get_params() const;

	/** Calls a function on every parameter of type T with the name of the
	 * parameter and the parameter itself as arguments
	 *
	 * @param operation the function to be called
	 */
	template <typename T>
	void for_each_param_of_type(
		std::function<void(const std::string&, T*)> operation);
#endif
	/** Specializes a provided object to the specified type.
	 * Throws exception if the object cannot be specialized.
	 *
	 * @param sgo object of SGObject base type
	 * @return The requested type
	 */
	template<class T> static T* as(std::shared_ptr<SGObject> sgo)
	{
		require(sgo, "No object provided!");
		return sgo->as<T>();
	}

	/** Specializes the object to the specified type.
	 * Throws exception if the object cannot be specialized.
	 *
	 * @return The requested type
	 */
	template<class T>
	inline std::shared_ptr<T> as()
	{
		auto c = std::dynamic_pointer_cast<T>(shared_from_this());
		if (c)
			return c;

		error(
			"Object of type {} cannot be converted to type {}.",
			this->get_name(),
			demangled_type<T>().c_str());
		return nullptr;
	}

	template<class T>
	inline std::shared_ptr<const T> as() const
	{
		auto c = std::dynamic_pointer_cast<const T>(shared_from_this());
		if (c)
			return c;

		error(
			"Object of type {} cannot be converted to type {}.",
			this->get_name(),
			demangled_type<T>().c_str());
		return nullptr;
	}

#ifndef SWIG
	/**
	  * Get parameters observable
	  * @return RxCpp observable
	  */
	std::shared_ptr<SGObservable> get_parameters_observable()
	{
		return m_observable_params;
	};
#endif

	/** Subscribe a parameter observer to watch over params
	 * @param obs a parameter observer implementation
	 */
	void subscribe(const std::shared_ptr<ParameterObserver>& obs);

	/**
	 * Detach an observer from the current SGObject.
	 * @param obs a parameter observer implementation
	 */
	void unsubscribe(const std::shared_ptr<ParameterObserver>& obs);

	/** Print to stdout a list of observable parameters */
	std::vector<std::string> observable_names();

	/** Get string to enum mapping */
	stringToEnumMapType get_string_to_enum_map() const
	{
		return m_string_to_enum_map;
	}

public:
	/** Can (optionally) be overridden to pre-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
	 *  is called.
	 *
	 *  @exception ShogunException will be thrown if an error occurs.
	 */
	virtual void load_serializable_pre();

	/** Can (optionally) be overridden to post-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST
	 *  is called.
	 *
	 *  @exception ShogunException will be thrown if an error occurs.
	 */
	virtual void load_serializable_post();

	/** Can (optionally) be overridden to pre-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
	 *  is called.
	 *
	 *  @exception ShogunException will be thrown if an error occurs.
	 */
	virtual void save_serializable_pre();

	/** Can (optionally) be overridden to post-initialize some member
	 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
	 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_POST
	 *  is called.
	 *
	 *  @exception ShogunException will be thrown if an error occurs.
	 */
	virtual void save_serializable_post();

	inline bool get_load_serializable_pre() const
	{
		return m_load_pre_called;
	}

	inline bool get_load_serializable_post() const
	{
		return m_load_post_called;
	}

	inline bool get_save_serializable_pre() const
	{
		return m_save_pre_called;
	}

	inline bool get_save_serializable_post() const
	{
		return m_save_post_called;
	}

protected:
	template<typename T>
	void register_parameter_visitor() const
	{
		if constexpr (is_auto_value_v<T>)
		{
			using ReturnType = traits::get_variant_type_t<0, T>;
			Any::register_visitor<T, ParameterPutInterface<ReturnType>>(
				[](T* value, auto* visitor)
				{
					*value = visitor->m_value;	
				}
			);
		}
		else
		{
			Any::register_visitor<T, ParameterPutInterface<T>>(
				[](T* value, auto* visitor)
				{
					*value = visitor->m_value;	
				}
			);	
		}

		if constexpr (traits::is_functional<T>::value)
		{
			if constexpr (!traits::returns_void<T>::value)
			{
				using ReturnType = typename T::result_type;
				Any::register_visitor<T, ParameterGetterInterface<ReturnType, T>>(
					[](T* value, auto* visitor)
					{
						visitor->m_value = value->operator()();
					}
				);
			}
		}
		else if constexpr (is_auto_value_v<T>)
		{
			using ReturnType = traits::get_variant_type_t<0, T>;
			Any::register_visitor<T, ParameterGetterInterface<ReturnType, T>>(
				[](T* value, auto* visitor)
				{
					if (std::holds_alternative<AutoValueEmpty>(*value))
					{
						// std::bad_optional_access does not support error messages
						// but this is caught by SGObject::get and throws then
						// a ShogunException
						throw std::bad_optional_access{};
					}
					else
						visitor->m_value = std::get<ReturnType>(*value);
				}
			);
		}
		else
		{
			Any::register_visitor<T, ParameterGetterInterface<T, T>>(
				[](T* value, auto* visitor)
				{
					visitor->m_value = *value;	
				}
			);
		}
	}
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
	void register_param(std::string_view name, const T& value)
	{
		create_parameter(BaseTag(name), AnyParameter(make_any(value)));
	}

	/** Puts a pointer to some parameter into the parameter map.
	 *
	 * @param name name of the parameter
	 * @param value pointer to the parameter value
	 * @param properties properties of the parameter (e.g. if model selection is supported)
	 */
	template <typename T>
	void watch_param(
		std::string_view name, T* value,
		AnyParameterProperties properties = AnyParameterProperties())
	{
		create_parameter(BaseTag(name), AnyParameter(make_any_ref(value), properties));
		register_parameter_visitor<T>();
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
			std::string_view name, T* value,
			std::shared_ptr<params::AutoInit> auto_init,
			AnyParameterProperties properties)
	{
		require(
				properties.has_property(ParameterProperties::AUTO),
				"Expected param to have ParameterProperty::AUTO");
		create_parameter(
				BaseTag(name),
				AnyParameter(
						make_any_ref(value), properties, std::move(auto_init)));
		register_parameter_visitor<T>();
	}

#ifndef SWIG
	/** Puts a pointer to some parameter into the parameter map.
	 * The parameter is the constrained to the rules from the
	 * Composer.
	 *
	 * @param name name of the parameter
	 * @param value pointer to the parameter value
	 * @param constrain_function a function pointer that checks if
	 * a given value is within the provided constraints
	 * @param properties properties of the parameter (e.g. if model
	 * selection is supported)
	 */
	template <typename T1, typename ...Args>
	void watch_param(
			std::string_view name, T1* value,
			Constraint<Args...>&& constrain_function,
			AnyParameterProperties properties)
	{
		require(
				properties.has_property(ParameterProperties::CONSTRAIN),
				"Expected param to have ParameterProperty::CONSTRAIN");
		create_parameter(
				BaseTag(name), AnyParameter(
						make_any_ref(value), properties,
						[constrain_function](const auto& val) {
							auto casted_val = any_cast<T1>(val);
							return constrain_function.check(casted_val);
						}));
		register_parameter_visitor<T1>();
	}
#endif

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
		std::string_view name, T** value, S* len,
		AnyParameterProperties properties = AnyParameterProperties())
	{
		create_parameter(
			BaseTag(name), AnyParameter(make_any_ref(value, len), properties));
		register_parameter_visitor<T>();
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
		std::string_view name, T** value, S* rows, S* cols,
		AnyParameterProperties properties = AnyParameterProperties())
	{
		create_parameter(
			BaseTag(name), AnyParameter(make_any_ref(value, rows, cols), properties));
		register_parameter_visitor<T>();
	}

#ifndef SWIG
	/** Puts a pointer to a (lazily evaluated) const function into the parameter map.
	 *
	 * @param name name of the parameter
	 * @param method pointer to the method
	 */
	template <typename T, typename S>
	void watch_method(std::string_view name, T (S::*method)() const)
	{
		AnyParameterProperties properties(
			"Dynamic parameter",
			ParameterProperties::READONLY | ParameterProperties::CONSTFUNCTION);
		std::function<T()> bind_method = [this, method](){
			return (static_cast<const S*>(this) ->* method)();
		};
		create_parameter(BaseTag(name), AnyParameter(make_any(bind_method), properties));
		register_parameter_visitor<std::function<T()>>();
	}

	/** Puts a pointer to a (lazily evaluated) function into the parameter map.
	 * The bound function can modify the class members and can only be
	 * invoked using SGObject::run(name).
	 *
	 * @param name name of the parameter
	 * @param method pointer to the method
	 */
	template <typename T, typename S>
	void watch_method(std::string_view name, T (S::*method)())
	{
		AnyParameterProperties properties(
			"Non-const function",
			ParameterProperties::READONLY | ParameterProperties::FUNCTION);
		std::function<T()> bind_method = [this, method](){
			return (static_cast<S*>(this) ->* method)();
		};
		create_parameter(BaseTag(name), AnyParameter(make_any(bind_method), properties));
		register_parameter_visitor<std::function<T()>>();
	}

	/** Adds a callback function to a parameter identified by its name
	 *
	 * @param name name of the parameter
	 * @param method pointer to the function
	 */
	void add_callback_function(
		std::string_view name, std::function<void()> method);

	/** Get the shared_ptr from SGObject and cast it to the Derived class
	 */
	template <typename Derived>
	std::shared_ptr<Derived> shared_from_base()
	{
		return std::static_pointer_cast<Derived>(shared_from_this());
	}
#endif

public:
	/** Updates the hash of current parameter combination */
	virtual void update_parameter_hash() const;

	/**
	 * @return whether parameter combination has changed since last update
	 */
	virtual bool parameter_hash_changed() const;

	/** Deep comparison of two objects.
	 *
	 * @param other object to compare with
	 * @return true if all parameters are equal
	 */
	virtual bool equals(const SGObject* other) const;

	/** Deep comparison of two objects.
	 *
	 * @param other object to compare with
	 * @return true if all parameters are equal
	 */
	virtual bool equals(const std::shared_ptr<const SGObject>& other) const;

	/** Creates a clone of the current object. This is done via recursively
	 * traversing all parameters, which corresponds to a deep copy.
	 *
	 * @return Cloned object
	 */
	virtual std::shared_ptr<SGObject> clone(ParameterProperties pp = ParameterProperties::ALL) const;

	/**
	 * Looks up the option name of a parameter given the enum value.
	 *
	 * @param param the parameter name
	 * @param value the enum value to query
	 * @return the string representation of the enum (option name)
	 */
	std::string string_enum_reverse_lookup(
			std::string_view param, machine_int_t value) const;

	void visit_parameter(const BaseTag& _tag, AnyVisitor* v) const;
protected:
	/** Returns an empty instance of own type.
	 *
	 * When inheriting from SGObject from outside the main source tree (i.e.
	 * customized classes, or in a unit test), then this method has to be
	 * overloaded manually to return an empty instance.
	 * Shogun can only instantiate empty class instances from its source tree.
	 *
	 * @return empty instance of own type
	 */
	virtual std::shared_ptr<SGObject> create_empty() const;

	/** Initialises all parameters with ParameterProperties::AUTO flag */
	void init_auto_params();

	/**
	 * Set a default mask value that is added to any other mask
	 * passed to watch_param. The default mask is by default
	 * ParameterProperties::NONE.
	 * @param mask the default mask
	 */
	void set_default_mask(ParameterProperties mask) noexcept
	{
		m_default_mask = mask;
	}

	/** The default mask for all registered parameters */
	ParameterProperties m_default_mask = ParameterProperties::NONE;

private:
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
	void create_parameter(BaseTag&& _tag, AnyParameter&& parameter);

	/** Updates a parameter identified by a BaseTag.
	 *
	 * @param _tag name information of parameter
	 * @param value new value of parameter
	 */
	template <typename T>
	void update_parameter(const BaseTag& _tag, const T& value, bool do_checks = true)
	{
		auto& param = get_parameter(_tag);
		auto& pprop = param.get_properties();
		auto& parameter_value = param.get_value();
		if (pprop.has_property(ParameterProperties::READONLY))
		        require(!do_checks,
				"{}::{} is marked as read-only and cannot be modified!",
		          	get_name(), _tag.name().c_str());

		if (pprop.has_property(ParameterProperties::CONSTRAIN))
		{
			const auto& val = param.get_constrain_function()(make_any(value));
			if (val)
			{
				require(!do_checks,
					"{}::{} cannot be updated because it must be: {}!",
					get_name(), _tag.name().c_str(), *val);
			}
		}
		if constexpr (std::is_same_v<T, Any>)
		{
			param.set_value(value);
		}
		else
		{
			ParameterPutInterface<T> visitor{value};

			try 
			{
				parameter_value.visit_with(&visitor);			
			}
			catch (...)
			{
				error(
					"Cannot put parameter {}::{} of type {}, incompatible with "
					"provided type {}.",
					get_name(), _tag.name().c_str(), parameter_value.type().c_str(),
					demangled_type<T>().c_str());
			}
		}

		for (auto& method : param.get_callbacks())
			method();
	}

	/** Getter for a class parameter, identified by a BaseTag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name information of parameter
	 * @return value of the parameter identified by the input tag
	 */
	const AnyParameter& get_parameter(const BaseTag& _tag) const;

	/** Getter for a class parameter, identified by a BaseTag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name information of parameter
	 * @return value of the parameter identified by the input tag
	 */
	AnyParameter& get_parameter(const BaseTag& _tag);

	/** Getter for a class function, identified by a BaseTag.
	 * Throws an exception if the class does not have such a parameter.
	 *
	 * @param _tag name information of parameter
	 * @return value of the parameter identified by the input tag
	 */
	const AnyParameter& get_function(const BaseTag& _tag) const;

	class Self;
	std::unique_ptr<Self> self;

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
	 * Observe a parameter value, given a pointer.
	 * @param value Observed parameter's value
	 */
	void observe(std::shared_ptr<ObservedValue> value) const;

	/**
	 * Observe a parameter value given custom properties for the Any.
	 * If no observer is attached this command will do nothing.
	 * @tparam T type of the parameter
	 * @param step time step
	 * @param name name of the observed value
	 * @param value observed value
	 * @param properties AnyParameterProperties used to register the observed
	 * value.
	 */
	template <class T>
	void observe(
		const int64_t step, std::string_view name, const T& value,
		const AnyParameterProperties properties) const
	{
		// If there are no observers attached, do not create/emit anything.
		if (get_num_subscriptions() == 0)
			return;

		auto obs = std::make_shared<ObservedValueTemplated<T>>(
			step, name, static_cast<T>(clone_utils::clone(value)), properties);
		this->observe(obs);
	}

	/**
	 * Observe a parameter value given some information.
	 * If no observer is attached this command will do nothing.
	 * @tparam T value of the parameter
	 * @param step step
	 * @param name name of the observed value
	 * @param description description
	 * @param value observed value
	 */
	template <class T>
	void observe(
		const int64_t step, std::string_view name,
		std::string_view description, const T value) const
	{
		this->observe(
			step, name, value,
			AnyParameterProperties(description, ParameterProperties::READONLY));
	}

	/**
	 * Observe a registered tag.
	 * If no observer is attached this command will do nothing.
	 * @tparam T type of the tag
	 * @param step step
	 * @param name tag's name
	 */
	template <class T>
	void observe(const int64_t step, std::string_view name) const
	{
		const auto tag = Tag<T>(name);
		auto& param = this->get_parameter(tag);
		auto pprop = param.get_properties();
		auto cloned = get(tag);
		pprop.remove_property(ParameterProperties::CONSTFUNCTION);
		pprop.remove_property(ParameterProperties::FUNCTION);
		this->observe(
			step, name, static_cast<T>(clone_utils::clone(cloned)),
			pprop);
	}

	/**
	 * Register which params this object can emit.
	 * @param name the param name
	 * @param description a user oriented description
	 */
	void register_observable(
		std::string_view name, std::string_view description);

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
		/** Hash of parameter values*/
		mutable size_t m_hash;

	private:
		EPrimitiveType m_generic;
		bool m_load_pre_called;
		bool m_load_post_called;
		bool m_save_pre_called;
		bool m_save_post_called;

		/** Subject used to create the params observer */
		std::shared_ptr<SGSubject> m_subject_params;

		/** Parameter Observable */
		std::shared_ptr<SGObservable> m_observable_params;

		/** Subscriber used to call onNext, onComplete etc.*/
		std::shared_ptr<SGSubscriber> m_subscriber_params;

		/** List of subscription for this SGObject */
		std::map<int64_t, rxcpp::subscription> m_subscriptions;
		int64_t m_next_subscription_index;
	};

template <class T>
std::shared_ptr<T> make_clone(std::shared_ptr<T> orig, ParameterProperties pp = ParameterProperties::ALL)
{
	require(orig, "No object provided.");
	auto clone = orig->clone(pp);
	ASSERT(clone);
	return std::static_pointer_cast<T>(clone);
}

template <class T>
std::shared_ptr<const T> make_clone(std::shared_ptr<const T> orig, ParameterProperties pp = ParameterProperties::ALL)
{
	require(orig, "No object provided.");
	auto clone = orig->clone(pp);
	ASSERT(clone);
	return std::static_pointer_cast<const T>(clone);
}

#ifndef SWIG
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace sgo_details
{
		template <typename T1, typename T2>
		bool dispatch_array_type(
		    const std::shared_ptr<const SGObject>& obj, std::string_view name,
		    T2&& lambda)
		{
			Tag<std::vector<std::shared_ptr<T1>>> tag_vector(name);
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
		std::shared_ptr<SGObject> get_if_possible(const std::shared_ptr<const SGObject>& obj, std::string_view name, GetByName)
		{
			return obj->has<T>(name) ? obj->get<T>(name) : nullptr;
		}

		template <typename T>
		std::shared_ptr<SGObject> get_if_possible(const std::shared_ptr<const SGObject>& obj, std::string_view name, GetByNameIndex how)
		{
			std::shared_ptr<SGObject> result = nullptr;
			result = obj->get<T>(name, how.m_index, std::nothrow);
			return result;
		}

		template<typename T>
		std::shared_ptr<SGObject> get_dispatch_all_base_types(const std::shared_ptr<const SGObject>& obj, std::string_view name,
			T&& how)
		{
			if (auto result = get_if_possible<Kernel>(obj, name, how))
				return result;
			if (auto result = get_if_possible<Features>(obj, name, how))
				return result;
			if (auto result = get_if_possible<Machine>(obj, name, how))
				return result;
			if (auto result = get_if_possible<Labels>(obj, name, how))
				return result;
			if (auto result = get_if_possible<EvaluationResult>(obj, name, how))
				return result;
			if (auto result = get_if_possible<LikelihoodModel>(obj, name, how))
				return result;
			if (auto result = get_if_possible<MeanFunction>(obj, name, how))
				return result;

			return nullptr;
		}

		template<class T>
		std::shared_ptr<SGObject> get_by_tag(const std::shared_ptr<const SGObject>& obj, std::string_view name,
			T&& how)
		{
			return get_dispatch_all_base_types(obj, name, how);
		}
} // namespace sgo_details

#endif //DOXYGEN_SHOULD_SKIP_THIS
#endif //SWIG

}
#endif // __SGOBJECT_H__
