#ifndef __PARAMETERHANDLER_H__
#define __PARAMETERHANDLER_H__

#include <shogun/base/AnyParameter.h>
#include <shogun/base/base_types.h>
#include <shogun/base/macros.h>
#include <shogun/base/some.h>
#include <shogun/base/unique.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/any.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/lib/exception/ShogunException.h>
#include <shogun/lib/tag.h>

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shogun
{
	class CDynamicObjectArray;
	template <typename Dervied>
	class HouseKeeper;
	template <typename Derived>
	class ParameterWatcher;
	class CSGObject;

	using stringToEnumMapType = std::unordered_map<
	    std::string, std::unordered_map<std::string, machine_int_t>>;

#ifndef SWIG
#ifndef DOXYGEN_SHOULD_SKIP_THIS
	namespace sgo_details
	{
		template <typename T1, typename T2, typename ParamObject>
		bool dispatch_array_type(
		    const ParamObject* obj, const std::string& name, T2&& lambda);
	}
#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // SWIG

	template <typename Derived>
	class ParameterHandler
	{
		friend ParameterWatcher<Derived>;

	public:
		/** default constructor */
		ParameterHandler();

		/** copy constructor */
		ParameterHandler(const ParameterHandler<Derived>& orig);

		/** destructor */
		virtual ~ParameterHandler();

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
		 * @return an identical copy of the given object, which is disjoint in
		 * memory. NULL if the clone fails. Note that the returned object is
		 * SG_REF'ed
		 */
		virtual Derived* clone() const;

		/** Returns string representation of the object that contains
		 * its name and parameters.
		 */
		virtual std::string to_string() const;

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

		/** Get string to enum mapping */
		stringToEnumMapType get_string_to_enum_map() const
		{
			return m_string_to_enum_map;
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

		/** Typed appender for an object class parameter of a Shogun base class
		 * type, identified by a name.
		 *
		 * @param name name of the parameter
		 * @param value value of the parameter
		 */
		template <
		    class T,
		    class X = typename std::enable_if<is_sg_base<T>::value>::type>
		void add(const std::string& name, T* value)
		{
			REQUIRE(
			    value, "Cannot add to %s::%s, no object provided.\n",
			    house_keeper.get_name(), name.c_str());

			auto push_back_lambda = [&value](auto& array) {
				array.push_back(value);
			};

			auto derived = static_cast<const Derived*>(this);
			if (sgo_details::dispatch_array_type<T>(
			        derived, name, push_back_lambda))
				return;

			SG_ERROR(
			    "Cannot add object %s to array parameter %s::%s of type %s.\n",
			    value->template get_name(), house_keeper.get_name(),
			    name.c_str(), demangled_type<T>().c_str());
		}

#ifndef SWIG
		/** Typed array getter for an object array class parameter of a Shogun
		 * base class type, identified by a name and an index.
		 *
		 * Returns nullptr if parameter of desired type does not exist.
		 *
		 * @param name name of the parameter array
		 * @param index index of the element in the array
		 * @return desired element
		 */
		template <
		    class T,
		    class X = typename std::enable_if<is_sg_base<T>::value>::type>
		T* get(const std::string& name, index_t index, std::nothrow_t) const
		{
			CSGObject* result = nullptr;

			auto get_lambda = [&index, &result](auto& array) {
				result = array.at(index);
			};

			auto derived = static_cast<const Derived*>(this);
			if (sgo_details::dispatch_array_type<T>(derived, name, get_lambda))
			{
				ASSERT(result);
				// guard against mixed types in the array
				// FIXME
				return dynamic_cast<T*>(result);
			}

			return nullptr;
		}

		template <
		    class T,
		    class X = typename std::enable_if<is_sg_base<T>::value>::type>
		T* get(const std::string& name, index_t index) const
		{
			auto result = this->template get<T>(name, index, std::nothrow);
			if (!result)
			{
				SG_ERROR(
				    "Could not get array parameter %s::%s[%d] of type %s\n",
				    house_keeper.get_name(), name.c_str(), index,
				    demangled_type<T>().c_str());
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
		CSGObject* get(const std::string& name) const noexcept(false);

		/** Untyped getter for an object class parameter, identified by a name.
		 * Does not throw an error if class parameter object cannot be casted
		 * to appropriate internal type.
		 *
		 * @param name name of the parameter
		 * @return object parameter
		 */
		CSGObject* get(const std::string& name, std::nothrow_t) const noexcept;

		/** Untyped getter for an object array class parameter, identified by a
		 * name and an index. Will attempt to get specified object of
		 * appropriate internal type. If this is not possible it will raise a
		 * ShogunException.
		 *
		 * @param name name of the parameter
		 * @index index of the parameter
		 * @return object parameter
		 */
		CSGObject* get(const std::string& name, index_t index) const;

#ifndef SWIG
		/** Typed setter for an object class parameter of a Shogun base class
		 * type, identified by a name.
		 *
		 * @param name name of the parameter
		 * @param value value of the parameter
		 */
		template <
		    class T, class = typename std::enable_if_t<is_sg_base<T>::value>>
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
		template <
		    class T, class = typename std::enable_if_t<is_sg_base<T>::value>>
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
		template <
		    typename T,
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
		template <
		    typename T,
		    typename std::enable_if_t<!is_string<T>::value>* = nullptr>
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
				    house_keeper.get_name(), _tag.name().c_str(),
				    exc.actual().c_str(), exc.expected().c_str());
			}
			// we won't be there
			return any_cast<T>(value);
		}

		template <
		    typename T,
		    typename std::enable_if_t<is_string<T>::value>* = nullptr>
		T get(const Tag<T>& _tag) const noexcept(false)
		{
			if (m_string_to_enum_map.find(_tag.name()) ==
			    m_string_to_enum_map.end())
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
					    "requested type %s or there are no options for "
					    "parameter "
					    "%s::%s.\n",
					    house_keeper.get_name(), _tag.name().c_str(),
					    exc.actual().c_str(), exc.expected().c_str(),
					    house_keeper.get_name(), _tag.name().c_str());
				}
			}
			return string_enum_reverse_lookup(
			    _tag.name(), get<machine_int_t>(_tag.name()));
		}

		/** Returns map of parameter names and AnyParameter pairs
		 * of the object.
		 *
		 */
		std::map<std::string, std::shared_ptr<const AnyParameter>>
		get_params() const;

#endif
		/** Getter for a class parameter, identified by a name.
		 * Throws an exception if the class does not have such a parameter.
		 *
		 * @param name name of the parameter
		 * @return value of the parameter corresponding to the input name and
		 * type
		 */
		template <typename T, typename U = void>
		T get(const std::string& name) const noexcept(false)
		{
			Tag<T> tag(name);
			return get(tag);
		}

#ifndef SWIG
		/** Setter for a class parameter, identified by a Tag.
		 * Throws an exception if the class does not have such a parameter.
		 *
		 * @param _tag name and type information of parameter
		 * @param value value of the parameter
		 */
		template <
		    typename T,
		    typename std::enable_if_t<!is_string<T>::value>* = nullptr>
		void put(const Tag<T>& _tag, const T& value) noexcept(false)
		{
			if (has_parameter(_tag))
			{
				auto parameter_value = get_parameter(_tag).get_value();
				if (!parameter_value.cloneable())
				{
					SG_ERROR(
					    "Cannot put parameter %s::%s.\n",
					    house_keeper.get_name(), _tag.name().c_str());
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
					    house_keeper.get_name(), _tag.name().c_str(),
					    exc.actual().c_str(), exc.expected().c_str());
				}
				house_keeper.ref_value(value);
				update_parameter(_tag, make_any(value));

				// ParameterHandler needs observe
				// FIXME!
				static_cast<Derived*>(this)->template observe<T>(
				    this->get_step(), _tag.name());
			}
			else
			{
				SG_ERROR(
				    "Parameter %s::%s does not exist.\n",
				    house_keeper.get_name(), _tag.name().c_str());
			}
		}

		/** Setter for a class parameter that has values of type string,
		 * identified by a Tag.
		 * Throws an exception if the class does not have such a parameter.
		 *
		 * @param _tag name and type information of parameter
		 * @param value value of the parameter
		 */
		template <
		    typename T,
		    typename std::enable_if_t<is_string<T>::value>* = nullptr>
		void put(const Tag<T>& _tag, const T& value) noexcept(false)
		{
			std::string val_string(value);

			if (m_string_to_enum_map.find(_tag.name()) ==
			    m_string_to_enum_map.end())
			{
				SG_ERROR(
				    "There are no options for parameter %s::%s",
				    house_keeper.get_name(), _tag.name().c_str());
			}

			auto string_to_enum = m_string_to_enum_map[_tag.name()];

			if (string_to_enum.find(val_string) == string_to_enum.end())
			{
				SG_ERROR(
				    "Illegal option '%s' for parameter %s::%s",
				    val_string.c_str(), house_keeper.get_name(),
				    _tag.name().c_str());
			}

			machine_int_t enum_value = string_to_enum[val_string];

			put(Tag<machine_int_t>(_tag.name()), enum_value);
		}
#endif
		/** Typed setter for an object class parameter of a Shogun base class
		 * type, identified by a name.
		 *
		 * @param name name of the parameter
		 * @param value value of the parameter
		 */
		template <
		    class T,
		    class X = typename std::enable_if<is_sg_base<T>::value>::type,
		    class Z = void>
		void put(const std::string& name, T* value)
		{
			put(Tag<T*>(name), value);
		}

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

		/** Initialises all parameters with ParameterProperties::AUTO flag */
		void init_auto_params();

#ifndef SWIG
		/**
		 * Get the current step for the observed values.
		 */
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

	private:
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
		void
		create_parameter(const BaseTag& _tag, const AnyParameter& parameter);

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

		class Self;
		Unique<Self> self;

		// mixins
		HouseKeeper<Derived>& house_keeper;
		SGIO*& io;
	};

} // namespace shogun

#endif // __PARAMETERHANDLER_H__