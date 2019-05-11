#ifndef __HOUSEKEEPER_H__
#define __HOUSEKEEPER_H__

#include <shogun/base/Parallel.h>
#include <shogun/base/Version.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/RefCount.h>
#include <shogun/lib/any.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/util/mixins.h>

namespace shogun
{
#define SG_REF(x)                                                              \
	{                                                                          \
		if (x)                                                                 \
			(x)->ref();                                                        \
	}
#define SG_UNREF(x)                                                            \
	{                                                                          \
		if (x)                                                                 \
		{                                                                      \
			if ((x)->unref() == 0)                                             \
				(x) = NULL;                                                    \
		}                                                                      \
	}
#define SG_UNREF_NO_NULL(x)                                                    \
	{                                                                          \
		if (x)                                                                 \
		{                                                                      \
			(x)->unref();                                                      \
		}                                                                      \
	}

	extern Parallel* sg_parallel;
	extern SGIO* sg_io;
	extern Version* sg_version;

	template <typename M>
	class HouseKeeper : public mixin<M>
	{
		using Derived = typename M::derived_t;

	public:
		/** copy constructor */
		HouseKeeper(const HouseKeeper<M>& orig) : HouseKeeper()
		{
			m_generic = orig.m_generic;
		}

		/** default constructor */
		HouseKeeper()
		{
			init();
			set_global_objects();
			m_refcount = new RefCount(0);
		}

		/** destructor */
		virtual ~HouseKeeper()
		{
			SG_SGCDEBUG("SGObject destroyed (%p)\n", this)
			unset_global_objects();
			delete m_refcount;
		}

		/** increase reference counter
		 *
		 * @return reference count
		 */
		int32_t ref()
		{
			int32_t count = m_refcount->ref();
			SG_SGCDEBUG(
			    "ref() refcount %ld obj %s (%p) increased\n", count,
			    this->get_name(), this)
			return m_refcount->ref_count();
		}

		/** display reference counter
		 *
		 * @return reference count
		 */
		int32_t ref_count()
		{
			int32_t count = m_refcount->ref_count();
			SG_SGCDEBUG(
			    "ref_count(): refcount %d, obj %s (%p)\n", count,
			    this->get_name(), this)
			return m_refcount->ref_count();
		}

		/** decrement reference counter and deallocate object if refcount is
		 * zero before or after decrementing it
		 *
		 * @return reference count
		 */
		int32_t unref()
		{
			int32_t count = m_refcount->unref();
			if (count <= 0)
			{
				SG_SGCDEBUG(
				    "unref() refcount %ld, obj %s (%p) destroying\n", count,
				    this->get_name(), this)
				delete this;
				return 0;
			}
			else
			{
				SG_SGCDEBUG(
				    "unref() refcount %ld obj %s (%p) decreased\n", count,
				    this->get_name(), this)
				return m_refcount->ref_count();
			}
		}

		/** Overloaded helper to increase reference counter */
		static void ref_value(Derived* value)
		{
			SG_REF(value);
		}

		/** Overloaded helper to increase reference counter
		 * Here a no-op for non CSGobject pointer parameters */
		template <
		    typename T,
		    std::enable_if_t<
		        !std::is_base_of<
		            Derived, typename std::remove_pointer<T>::type>::value,
		        T>* = nullptr>
		static void ref_value(T value)
		{
		}

		/** A shallow copy.
		 * All the SGObject instance variables will be simply assigned and
		 * SG_REF-ed.
		 */
		virtual Derived* shallow_copy() const
		{
			SG_NOTIMPLEMENTED
			return nullptr;
		}

		/** A deep copy.
		 * All the instance variables will also be copied.
		 */
		virtual Derived* deep_copy() const
		{
			SG_NOTIMPLEMENTED
			return nullptr;
		}

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		virtual const char* get_name() const = 0;

		/** set the io object
		 *
		 * @param io io object to use
		 */
		void set_global_io(SGIO* new_io)
		{
			SG_REF(new_io);
			SG_UNREF(sg_io);
			sg_io = new_io;
		}

		/** get the io object
		 *
		 * @return io object
		 */
		SGIO* get_global_io()
		{
			SG_REF(sg_io);
			return sg_io;
		}

		/** set the parallel object
		 *
		 * @param parallel parallel object to use
		 */
		void set_global_parallel(Parallel* new_parallel)
		{
			SG_REF(new_parallel);
			SG_UNREF(sg_parallel);
			sg_parallel = new_parallel;
		}

		/** get the parallel object
		 *
		 * @return parallel object
		 */
		Parallel* get_global_parallel()
		{
			SG_REF(sg_parallel);
			return sg_parallel;
		}

		/** set the version object
		 *
		 * @param version version object to use
		 */
		void set_global_version(Version* new_version)
		{
			SG_REF(new_version);
			SG_UNREF(sg_version);
			sg_version = new_version;
		}

		/** get the version object
		 *
		 * @return version object
		 */
		Version* get_global_version()
		{
			SG_REF(sg_version);
			return sg_version;
		}

		/** Specializes a provided object to the specified type.
		 * Throws exception if the object cannot be specialized.
		 *
		 * @param sgo object of CSGObject base type
		 * @return The requested type
		 */
		template <class T>
		static T* as(Derived* sgo)
		{
			REQUIRE(sgo, "No object provided!\n");
			return sgo->template as<T>();
		}

		/** Specializes the object to the specified type.
		 * Throws exception if the object cannot be specialized.
		 *
		 * @return The requested type
		 */
		template <class T>
		T* as()
		{
			auto c = dynamic_cast<T*>(this);
			if (c)
				return c;

			SG_SERROR(
			    "Object of type %s cannot be converted to type %s.\n",
			    this->get_name(), demangled_type<T>().c_str());
			return nullptr;
		}

		/** If the SGSerializable is a class template then TRUE will be
		 *  returned and GENERIC is set to the type of the generic.
		 *
		 *  @param generic set to the type of the generic if returning
		 *                 TRUE
		 *
		 *  @return TRUE if a class template.
		 */
		virtual bool is_generic(EPrimitiveType* generic) const
		{
			*generic = m_generic;
			return m_generic != PT_NOT_GENERIC;
		}

		/** set generic type to T
		 */
		template <class T>
		void set_generic()
		{
			m_generic = TSGDataType::type_to_ptype<T>();
		}

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
		void unset_generic()
		{
			m_generic = PT_NOT_GENERIC;
		}

	private:
		void set_global_objects()
		{
			if (!sg_io || !sg_parallel || !sg_version)
			{
				fprintf(
				    stderr,
				    "call init_shogun() before using the library, dying.\n");
				exit(1);
			}

			SG_REF(sg_io);
			SG_REF(sg_parallel);
			SG_REF(sg_version);

			io = sg_io;
			parallel = sg_parallel;
			version = sg_version;
		}

		void unset_global_objects()
		{
			SG_UNREF(version);
			SG_UNREF(parallel);
			SG_UNREF(io);
		}

		void init()
		{
			io = nullptr;
			parallel = nullptr;
			version = nullptr;
			m_refcount = nullptr;
			m_generic = PT_NOT_GENERIC;
		}

	public:
		SGIO* io;
		Parallel* parallel;
		Version* version;

	private:
		EPrimitiveType m_generic;
		RefCount* m_refcount;
	};

} // namespace shogun

#endif // __HOUSEKEEPER_H__