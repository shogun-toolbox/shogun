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

			/** Not implemented copy constructor */
			Unique(const Unique&);
			/** Not implemented assignment operator */
			Unique& operator=(const Unique& other);

			/** Access underlying unique object as a raw pointer */
			inline T* operator->() const
			{
				return reinterpret_cast<T*>(data);
			}
		private:
			/** Untyped data storage */
			void* data;
	};

}
#endif /* __SG_UNIQUE_H__ */
