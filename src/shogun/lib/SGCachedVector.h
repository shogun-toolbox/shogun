/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
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
