#ifndef __SG_UNIQUE_H__
#define __SG_UNIQUE_H__

#include <cstddef>
#include <shogun/base/macros.h>

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
			constexpr Unique() noexcept: data()
			{
				data = new T();
			}

			constexpr Unique(std::nullptr_t np) noexcept: data()
			{
				data = nullptr;
			}

			Unique(Unique&& orig) noexcept
			{
				data = orig.data;
				orig.data = nullptr;
			}

			~Unique()
			{
				delete reinterpret_cast<T*>(data);
			}

			Unique& operator=(Unique&& orig) noexcept
			{
				if (this->data != orig.data)
				{
					this->reset(orig.data);
					orig.data = nullptr;
				}
				return *this;
			}

			Unique& operator=(std::nullptr_t np) noexcept
			{
				this->data = nullptr;
				return *this;
			}

			void reset(T* ptr) noexcept
			{
				auto old_ptr = data;
				data = ptr;
				if (old_ptr)
					delete reinterpret_cast<T*>(old_ptr);
			}

			/** Access underlying unique object as a raw pointer */
			SG_FORCED_INLINE T* operator->() const noexcept
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
