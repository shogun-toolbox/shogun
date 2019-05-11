#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include <shogun/base/mixins/HouseKeeper.h>
#include <shogun/base/mixins/ParameterHandler.h>
#include <shogun/base/mixins/ParameterWatcher.h>
#include <shogun/base/mixins/SGObjectBase.h>

namespace shogun
{
#ifndef IGNORE_IN_CLASSLIST
#define IGNORE_IN_CLASSLIST
#endif

	IGNORE_IN_CLASSLIST class CSGObject : public CSGObjectBase<CSGObject>,
	                                      public HouseKeeper<CSGObject>,
	                                      public ParameterHandler<CSGObject>,
	                                      public ParameterWatcher<CSGObject>
	{
	public:
		virtual ~CSGObject(){};

		/** Returns an empty instance of own type.
		 *
		 * When inheriting from CSGObject from outside the main source tree
		 * (i.e. customized classes, or in a unit test), then this method has to
		 * be overloaded manually to return an empty instance. Shogun can only
		 * instantiate empty class instances from its source tree.
		 *
		 * @return empty instance of own type
		 */
		virtual CSGObject* create_empty() const override;

		// to resolve naming conflict
		SGIO*& io = HouseKeeper<CSGObject>::io;
	};

#ifndef SWIG
#ifndef DOXYGEN_SHOULD_SKIP_THIS
	namespace sgo_details
	{
		template <typename T1, typename T2, typename ParamObject>
		bool dispatch_array_type(
		    const ParamObject* obj, const std::string& name, T2&& lambda)
		{
			Tag<CDynamicObjectArray*> tag_array_sg(name);
			if (obj->template has(tag_array_sg))
			{
				auto dispatched = obj->template get(tag_array_sg);
				lambda(*dispatched); // is stored as a pointer
				return true;
			}

			Tag<std::vector<T1*>> tag_vector(name);
			if (obj->template has(tag_vector))
			{
				auto dispatched = obj->template get(tag_vector);
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
			GetByNameIndex(index_t index) : m_index(index)
			{
			}
			index_t m_index;
		};

		template <typename T, typename ParamObject>
		CSGObject* get_if_possible(
		    const ParamObject* obj, const std::string& name, GetByName)
		{
			return obj->template has<T*>(name) ? obj->template get<T*>(name)
			                                   : nullptr;
		}

		template <typename T, typename ParamObject>
		CSGObject* get_if_possible(
		    const ParamObject* obj, const std::string& name, GetByNameIndex how)
		{
			CSGObject* result = nullptr;
			result = obj->template get<T>(name, how.m_index, std::nothrow);
			return result;
		}

		template <typename T, typename ParamObject>
		CSGObject* get_dispatch_all_base_types(
		    const ParamObject* obj, const std::string& name, T&& how)
		{
			if (auto* result = get_if_possible<CKernel>(obj, name, how))
				return result;
			if (auto* result = get_if_possible<CFeatures>(obj, name, how))
				return result;
			if (auto* result = get_if_possible<CMachine>(obj, name, how))
				return result;
			if (auto* result = get_if_possible<CLabels>(obj, name, how))
				return result;
			if (auto* result =
			        get_if_possible<CEvaluationResult>(obj, name, how))
				return result;

			return nullptr;
		}

		template <class T, typename ParamObject>
		CSGObject*
		get_by_tag(const ParamObject* obj, const std::string& name, T&& how)
		{
			return get_dispatch_all_base_types(obj, name, how);
		}
	}  // namespace sgo_details
#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // SWIG

} // namespace shogun

#endif