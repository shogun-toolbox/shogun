/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Björn Esser
 */
//template<class T> class SGCachedVector : public SGVector<T>
//{
//	public:
//		/** default constructor */
//		SGCachedVector(CCache<T>* c, index_t i)
//			: SGVector<T>(), cache(c), idx(i)
//		{
//		}
//
//		/** constructor for setting params */
//		SGCachedVector(CCache<T>* c, index_t i,
//				T* v, index_t len, bool free_vec=false)
//			: SGVector<T>(v, len, free_vec), cache(c), idx(i)
//		{
//		}
//
//		/** constructor to create new vector in memory */
//		SGCachedVector(CCache<T>* c, index_t i, index_t len, bool free_vec=false) :
//			SGVector<T>(len, free_vec), cache(c), idx(i)
//		{
//		}
//
//		/** free vector */
//		virtual void free_vector()
//		{
//			//clean up cache fixme
//			SGVector<T>::free_vector();
//		}
//
//		/** destroy vector */
//		virtual void destroy_vector()
//		{
//			//clean up cache fixme
//			SGVector<T>::destroy_vector();
//			if (cache)
//				cache->unlock_entry(idx);
//		}
//
//	public:
//		/** idx */
//		index_t idx;
//
//		/** cache */
//		CCache<T>* cache;
//};
