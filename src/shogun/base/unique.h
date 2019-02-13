#ifndef __SG_UNIQUE_H__
#define __SG_UNIQUE_H__

#include <shogun/base/macros.h>
#include <shogun/shogun_export.h>

namespace shogun
{

	/** Holds unique pointer that is deleted once this holder is deleted.
	 * Its main usage is to hold a pointer to implementation (pimpl idiom):
	 *
	 * class Self;
	 * Unique<Self> self;
	 *
	 */
	template <typename T>
	class Unique
	{
		public:
			/** Creates an instance of something unique.
			 *
			 * Calls default constructor of type T.
			 *
			 */
			Unique() : data()
			{
				data = new T();
			}
			~Unique()
			{
				delete reinterpret_cast<T*>(data);
			}

			/** Access underlying unique object as a raw pointer */
			SG_FORCED_INLINE T* operator->() const
			{
				return reinterpret_cast<T*>(data);
			}
		private:
			/** Untyped data storage */
			void* data;

			SG_DELETE_COPY_AND_ASSIGN(Unique);
	};

}
#endif /* __SG_UNIQUE_H__ */
