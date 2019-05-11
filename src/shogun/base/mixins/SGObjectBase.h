/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Giovanni De Toni, Jacob Walker, Thoralf Klein, Chiyuan Zhang,
 *          Fernando Iglesias, Sanuj Sharma, Roman Votyakov, Yuyu Zhang,
 *          Viktor Gal, Bjoern Esser, Evangelos Anagnostopoulos, Pan Deng,
 *          Gil Hoben
 */

#ifndef __SGOBJECTBASE_H__
#define __SGOBJECTBASE_H__

#include <shogun/base/AnyParameter.h>
#include <shogun/base/Version.h>
#include <shogun/base/base_types.h>
#include <shogun/base/macros.h>
#include <shogun/base/mixins/HouseKeeper.h>
#include <shogun/base/mixins/ParameterHandler.h>
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

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

/** \namespace shogun
 * @brief all of classes and functions are contained in the shogun namespace
 */
namespace shogun
{
	class SGIO;
	class Parameter;
	class CSerializableFile;
	class ParameterObserver;
	class CDynamicObjectArray;

	template <class T, class K>
	class CMap;

	struct TParameter;
	template <class T>
	class DynArray;
	template <class T>
	class SGStringList;

	using stringToEnumMapType = std::unordered_map<
	    std::string, std::unordered_map<std::string, machine_int_t>>;

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
	 * objects in memory (erase unused object, avoid cleaning objects when they
	 * are still in use), it provides interfaces for:
	 *
	 * -# parallel - to determine the number of used CPUs for a method (cf.
	 * Parallel)
	 * -# io - to output messages and general i/o (cf. IO)
	 * -# version - to provide version information of the shogun version used
	 * (cf. Version)
	 *
	 * All objects can be cloned and compared (deep copy, recursively)
	 */

	template <typename Derived>
	class CSGObjectBase
	{
	public:
		/** default constructor */
		CSGObjectBase();

		/** copy constructor */
		CSGObjectBase(const CSGObjectBase<Derived>& orig);

		/** destructor */
		virtual ~CSGObjectBase();

#ifdef TRACE_MEMORY_ALLOCS
		static void list_memory_allocs();
#endif

		/** prints registered parameters out
		 *
		 *	@param prefix prefix for members
		 */
		virtual void print_serializable(const char* prefix = "");

		/** Save this object to file.
		 *
		 * @param file where to save the object; will be closed during
		 * returning if PREFIX is an empty string.
		 * @param prefix prefix for members
		 * @return TRUE if done, otherwise FALSE
		 */
		virtual bool
		save_serializable(CSerializableFile* file, const char* prefix = "");

		/** Load this object from file.  If it will fail (returning FALSE)
		 *  then this object will contain inconsistent data and should not
		 *  be used!
		 *
		 *  @param file where to load from
		 *  @param prefix prefix for members
		 *
		 *  @return TRUE if done, otherwise FALSE
		 */
		virtual bool
		load_serializable(CSerializableFile* file, const char* prefix = "");

		/** @return vector of names of all parameters which are registered for
		 * model selection */
		SGStringList<char> get_modelsel_names();

		/** prints all parameter registered for model selection and their type
		 */
		void print_modsel_params();

		/** Returns description of a given parameter string, if it exists.
		 * SG_ERROR otherwise
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
		void
		build_gradient_parameter_dictionary(CMap<TParameter*, Derived*>* dict);

		/** Print to stdout a list of observable parameters */
		std::vector<std::string> observable_names();

	protected:
		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException will be thrown if an error occurs.
		 */
		virtual void load_serializable_pre() throw(ShogunException);

		/** Can (optionally) be overridden to post-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST
		 *  is called.
		 *
		 *  @exception ShogunException will be thrown if an error occurs.
		 */
		virtual void load_serializable_post() throw(ShogunException);

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException will be thrown if an error occurs.
		 */
		virtual void save_serializable_pre() throw(ShogunException);

		/** Can (optionally) be overridden to post-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_POST
		 *  is called.
		 *
		 *  @exception ShogunException will be thrown if an error occurs.
		 */
		virtual void save_serializable_post() throw(ShogunException);

	public:
		/** Updates the hash of current parameter combination */
		virtual void update_parameter_hash();

		/**
		 * @return whether parameter combination has changed since last update
		 */
		virtual bool parameter_hash_changed();

	private:
		void init();

		/** Gets an incremental hash of all parameters as well as the parameters
		 * of CSGObject children of the current object's parameters.
		 *
		 * @param hash the computed hash returned by reference
		 * @param carry value for Murmur3 incremental hash
		 * @param total_length total byte length of all hashed parameters so
		 * far. Byte length of parameters will be added to the total length
		 */
		void get_parameter_incremental_hash(
		    uint32_t& hash, uint32_t& carry, uint32_t& total_length);

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

		// mixins
		HouseKeeper<Derived>& house_keeper;
		ParameterHandler<Derived>& param_handler;
		SGIO*& io;
	};
} // namespace shogun
#endif // __SGOBJECT_H__
