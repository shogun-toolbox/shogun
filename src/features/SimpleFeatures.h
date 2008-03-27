/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SIMPLEFEATURES__H__
#define _SIMPLEFEATURES__H__

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/Cache.h"
#include "lib/io.h"
#include "lib/Cache.h"
#include "preproc/SimplePreProc.h"
#include "features/Features.h"

#include <string.h>


template <class ST> class CSimpleFeatures;
template <class ST> class CSimplePreProc;

/** class SimpleFeatures */
template <class ST> class CSimpleFeatures: public CFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CSimpleFeatures(INT size=0) :
			CFeatures(size), num_vectors(0), num_features(0),
			feature_matrix(NULL), feature_cache(NULL) {}

		/** copy constructor */
		CSimpleFeatures(const CSimpleFeatures & orig) :
			CFeatures(orig), num_vectors(orig.num_vectors),
			num_features(orig.num_features),
			feature_matrix(orig.feature_matrix),
			feature_cache(orig.feature_cache)
		{
			if (orig.feature_matrix)
			{
				free_feature_matrix();
				feature_matrix=new ST(num_vectors*num_features);
				memcpy(feature_matrix, orig.feature_matrix, sizeof(double)*num_vectors*num_features);
			}
		}

		/** constructor
		 *
		 * @param fm feature matrix
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 */
		CSimpleFeatures(ST* fm, INT num_feat, INT num_vec) :
			CFeatures(0), num_vectors(0), num_features(0),
			feature_matrix(NULL), feature_cache(NULL)
		{
			set_feature_matrix(fm, num_feat, num_vec);
		}

		/** constructor
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param fname filename to load features from
		 */
		CSimpleFeatures(CHAR* fname) :
			CFeatures(fname), num_vectors(0), num_features(0),
			feature_matrix(NULL), feature_cache(NULL) {}

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const
		{
			return new CSimpleFeatures<ST>(*this);
		}

		virtual ~CSimpleFeatures()
		{
			SG_DEBUG("deleting simplefeatures (0x%p)\n", this);
			free_features();
		}

		/** free feature matrix
		 *
		 */
		void free_feature_matrix()
		{
			if (feature_matrix)
			{
				SG_DEBUG( "free_feature_matrix in (0x%p)\n", this);
				delete[] feature_matrix;
				feature_matrix = NULL;
				num_vectors=0;
				num_features=0;
			}
		}

		/** free feature matrix and cache
		 *
		 */
		void free_features()
		{
			free_feature_matrix();
			if (feature_cache)
			{
				delete feature_cache;
				feature_cache = NULL;
			}
		}

		/** get feature vector
		 * for sample num from the matrix as it is if matrix is
		 * initialized, else return preprocessed compute_feature_vector
		 *
		 * @param num index of feature vector
		 * @param len length is returned by reference
		 * @param free whether returned vector must be freed by
		 * caller via free_feature_vector
		 * @return feature vector
		 */
		ST* get_feature_vector(INT num, INT& len, bool& free)
		{
			len=num_features;

			if (feature_matrix)
			{
				free=false ;
				return &feature_matrix[num*num_features];
			} 
			else
			{
				SG_DEBUG( "compute feature!!!\n") ;

				ST* feat=NULL;
				free=false;

				if (feature_cache)
				{
					feat=feature_cache->lock_entry(num);

					if (feat)
						return feat;
					else
					{
						feat=feature_cache->set_entry(num);
					}
				}

				if (!feat)
					free=true;
				feat=compute_feature_vector(num, len, feat);


				if (get_num_preproc())
				{
					INT tmp_len=len;
					ST* tmp_feat_before = feat;
					ST* tmp_feat_after = NULL;

					for (INT i=0; i<get_num_preproc(); i++)
					{
						tmp_feat_after=((CSimplePreProc<ST>*) get_preproc(i))->apply_to_feature_vector(tmp_feat_before, tmp_len);

						if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
							delete[] tmp_feat_before;
						tmp_feat_before=tmp_feat_after;
					}

					memcpy(feat, tmp_feat_after, sizeof(ST)*tmp_len);
					delete[] tmp_feat_after;

					len=tmp_len ;
					SG_DEBUG( "len: %d len2: %d\n", len, num_features);
				}
				return feat ;
			}
		}

		/** free feature vector
		 *
		 * @param feat_vec feature vector to free
		 * @param num index in feature cache
		 * @param free if vector should be really deleted
		 */
		void free_feature_vector(ST* feat_vec, INT num, bool free)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (free)
				delete[] feat_vec ;
		}

		/** get the pointer to the feature matrix
		 * num_feat,num_vectors are returned by reference
		 *
		 * @param dst destination to store matrix in
		 * @param d1 dimension 1 of matrix
		 * @param d2 dimension 2 of matrix
		 */
		void get_fm(ST** dst, INT* d1, INT* d2)
		{
			ASSERT(feature_matrix);

			LONG num=num_features*num_vectors;
			*d1=num_features;
			*d2=num_vectors;
			*dst=(ST*) malloc(sizeof(ST)*num);
			memcpy(*dst, feature_matrix, num * sizeof(ST));
		}

		/** get the pointer to the feature matrix
		 * num_feat,num_vectors are returned by reference
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return feature matrix
		 */
		ST* get_feature_matrix(INT &num_feat, INT &num_vec)
		{
			num_feat=num_features;
			num_vec=num_vectors;
			return feature_matrix;
		}

		/** set feature matrix
		 * necessary to set feature_matrix, num_features,
		 * num_vectors, where num_features is the column offset,
		 * and columns are linear in memory
		 * see below for definition of feature_matrix
		 *
		 * @param fm feature matrix to se
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 */
		virtual void set_feature_matrix(ST* fm, INT num_feat, INT num_vec)
		{
			free_feature_matrix();
			feature_matrix=fm;
			num_features=num_feat;
			num_vectors=num_vec;
		}

		/** copy feature matrix
		 * store copy of feature_matrix, where num_features is the
		 * column offset, and columns are linear in memory
		 * see below for definition of feature_matrix
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 */
		virtual void copy_feature_matrix(ST* src, INT num_feat, INT num_vec)
		{
			free_feature_matrix();
			feature_matrix=new ST[((LONG) num_feat)*num_vec];
			ASSERT(feature_matrix);
			memcpy(feature_matrix, src, (sizeof(ST)*((LONG) num_feat)*num_vec));

			num_features=num_feat;
			num_vectors=num_vec;
		}

		/** apply preprocessor
		 *
		 * @param force_preprocessing if preprocssing shall be forced
		 * @return if applying was successful
		 */
		virtual bool apply_preproc(bool force_preprocessing=false)
		{
			SG_DEBUG( "force: %d\n", force_preprocessing);

			if ( feature_matrix && get_num_preproc())
			{

				for (INT i=0; i<get_num_preproc(); i++)
				{ 
					if ( (!is_preprocessed(i) || force_preprocessing) )
					{
						set_preprocessed(i);

						SG_INFO( "preprocessing using preproc %s\n", get_preproc(i)->get_name());
						if (((CSimplePreProc<ST>*) get_preproc(i))->apply_to_feature_matrix(this) == NULL)
							return false;
					}
				}
				return true;
			}
			else
			{
				if (!feature_matrix)
					SG_ERROR( "no feature matrix\n");

				if (!get_num_preproc())
					SG_ERROR( "no preprocessors available\n");

				return false;
			}
		}

		/** get memory footprint of one feature
		 *
		 * @return memory footprint of one feature
		 */
		virtual INT get_size() { return sizeof(ST); }


		/** get number of feature vectors
		 *
		 * @return number of feature vectors
		 */
		virtual inline INT  get_num_vectors() { return num_vectors; }

		/** get number of features
		 *
		 * @return number of features
		 */
		inline INT  get_num_features() { return num_features; }

		/** set number of features
		 *
		 * @param num number to set
		 */
		inline void set_num_features(INT num)
		{ 
			num_features= num;

			if (num_features && num_vectors)
			{
				delete feature_cache;
				feature_cache= new CCache<ST>(get_cache_size(), num_features, num_vectors);
			}
		}

		/** set number of vectors
		 *
		 * @param num number to set
		 */
		inline void set_num_vectors(INT num)
		{
			num_vectors= num;
			if (num_features && num_vectors)
			{
				delete feature_cache;
				feature_cache= new CCache<ST>(get_cache_size(), num_features, num_vectors);
			}
		}

		/** get feature class
		 *
		 * @return feature class SIMPLE
		 */
		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }

		/** get feature type
		 *
		 * @return templated feature type
		 */
		inline virtual EFeatureType get_feature_type();

		/** reshape
		 *
		 * @param p_num_features new number of features
		 * @param p_num_vectors new number of vectors
		 * @return if reshaping was successful
		 */
		virtual bool reshape(INT p_num_features, INT p_num_vectors)
		{
			if (p_num_features*p_num_vectors == this->num_features * this->num_vectors)
			{
				this->num_features=p_num_features;
				this->num_vectors=p_num_vectors;
				return true;
			}
			else
				return false;
		}

	protected:
		/** compute feature vector for sample num
		 * if target is set the vector is written to target
		 * len is returned by reference
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param num num
		 * @param len len
		 * @param target
		 * @return feature vector
		 */
		virtual ST* compute_feature_vector(INT num, INT& len, ST* target=NULL)
		{
			len=0;
			return NULL;
		}

		/// number of vectors in cache
		INT num_vectors;

		/// number of features in cache
		INT num_features;

		/** feature matrix */
		ST* feature_matrix;

		/** feature cache */
		CCache<ST>* feature_cache;
};

/** get feature type the DREAL feature can deal with
 *
 * @return feature type DREAL
 */
template<> inline EFeatureType CSimpleFeatures<DREAL>::get_feature_type()
{
	return F_DREAL;
}

/** get feature type the SHORTREAL feature can deal with
 *
 * @return feature type SHORTREAL
 */
template<> inline EFeatureType CSimpleFeatures<SHORTREAL>::get_feature_type()
{
	return F_SHORTREAL;
}

/** get feature type the SHORT feature can deal with
 *
 * @return feature type SHORT
 */
template<> inline EFeatureType CSimpleFeatures<SHORT>::get_feature_type()
{
	return F_SHORT;
}

/** get feature type the CHAR feature can deal with
 *
 * @return feature type CHAR
 */
template<> inline EFeatureType CSimpleFeatures<CHAR>::get_feature_type()
{
	return F_CHAR;
}

/** get feature type the BYTE feature can deal with
 *
 * @return feature type BYTE
 */
template<> inline EFeatureType CSimpleFeatures<BYTE>::get_feature_type()
{
	return F_BYTE;
}

/** get feature type the INT feature can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType CSimpleFeatures<INT>::get_feature_type()
{
	return F_INT;
}

/** get feature type the WORD feature can deal with
 *
 * @return feature type WORD
 */
template<> inline EFeatureType CSimpleFeatures<WORD>::get_feature_type()
{
	return F_WORD;
}

/** get feature type the ULONG feature can deal with
 *
 * @return feature type ULONG
 */
template<> inline EFeatureType CSimpleFeatures<ULONG>::get_feature_type()
{
	return F_ULONG;
}
#endif
