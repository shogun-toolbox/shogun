#ifndef __SG_UNIQUE_H__
#define __SG_UNIQUE_H__

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
			Unique() : data()
			{
				data = new T();
			}
			~Unique()
			{
				delete reinterpret_cast<T*>(data);
			}

			Unique(const Unique& other);
			Unique& operator=(const Unique& other);

			inline T* operator->() const
			{
				return reinterpret_cast<T*>(data);
			}
		private:
			void* data;
	};

}
#endif /* __SG_UNIQUE_H__ */
