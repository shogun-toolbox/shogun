#ifndef __SG_UNIQUE_H__
#define __SG_UNIQUE_H__

#ifdef HAVE_CXX11

#include <memory>

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
		class Unique : protected std::unique_ptr<T>
	{
		public:
			Unique() : std::unique_ptr<T>(new T)
		{
		}

			Unique(const Unique& other) = delete;
			Unique(Unique&& other) = delete;
			Unique& operator=(const Unique& other) = delete;

			~Unique()
			{
			}

			inline T* operator->() const
			{
				return std::unique_ptr<T>::get();
			}

	};

}

#endif /* HAVE_CXX11 */
#endif /* __SG_UNIQUE_H__ */
