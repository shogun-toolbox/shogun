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
#include "features/DotFeatures.h"

#include <string.h>


template <class ST> class CSimpleFeatures;
template <class ST> class CSimplePreProc;

/** The class SimpleFeatures implements dense feature matrices, which are
 * stored en-block in memory in fortran order, i.e. column-by-column, where a
 * column denotes a feature vector.
 *
 * There are get_num_vectors() many feature vectors, of dimension
 * get_num_features(). To access a feature vector call
 * get_feature_vector() and when you are done treating it call
 * free_feature_vector(). While free_feature_vector() is a NOP in most cases
 * feature vectors might have been generated on the fly (due to a number
 * preprocessors being attached to them).
 *
 * From this template class a number of dense feature matrix classes are derived.
 * They all are only shortcuts for different data types and heavily rely on
 * this class:
 *
 * 8bit char matrix - CCharFeatures
 * 8bit Byte matrix - CByteFeatures
 * 16bit Integer matrix - CShortFeatures
 * 16bit Word matrix - CWordFeatures
 * 32bit Float matrix - CShortRealFeatures
 * 64bit Double matrix - CRealFeatures
 * 64bit Double matrix <b>in a file</b> - CRealFileFeatures
 * 64bit Tangent of posterior log-odds (TOP) features from HMM - CTOPFeatures
 * 64bit Fisher Kernel (FK) features from HMM - CTOPFeatures
 * 32bit Integer matrix - CIntFeatures
 */
template <class ST> class CSimpleFeatures: public CDotFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CSimpleFeatures(int32_t size=0)
		: CDotFeatures(size), num_vectors(0), num_features(0),
			feature_matrix(NULL), feature_cache(NULL) {}

		/** copy constructor */
		CSimpleFeatures(const CSimpleFeatures & orig)
		: CDotFeatures(orig), num_vectors(orig.num_vectors),
			num_features(orig.num_features),
			feature_matrix(orig.feature_matrix),
			feature_cache(orig.feature_cache)
		{
			if (orig.feature_matrix)
			{
				free_feature_matrix();
				feature_matrix=new ST(num_vectors*num_features);
				memcpy(feature_matrix, orig.feature_matrix, sizeof(float64_t)*num_vectors*num_features);
			}
		}

		/** constructor
		 *
		 * @param fm feature matrix
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 */
		CSimpleFeatures(ST* fm, int32_t num_feat, int32_t num_vec)
		: CDotFeatures(0), num_vectors(0), num_features(0),
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
		CSimpleFeatures(char* fname)
		: CDotFeatures(fname), num_vectors(0), num_features(0),
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
            delete[] feature_matrix;
            feature_matrix = NULL;
            num_vectors=0;
            num_features=0;
		}

		/** free feature matrix and cache
		 *
		 */
		void free_features()
		{
			free_feature_matrix();
            delete feature_cache;
            feature_cache = NULL;
		}

		/** get feature vector
		 * for sample num from the matrix as it is if matrix is
		 * initialized, else return preprocessed compute_feature_vector
		 *
		 * @param num index of feature vector
		 * @param len length is returned by reference
		 * @param dofree whether returned vector must be freed by
		 * caller via free_feature_vector
		 * @return feature vector
		 */
		ST* get_feature_vector(int32_t num, int32_t& len, bool& dofree)
		{
			len=num_features;

			if (feature_matrix)
			{
				dofree=false;
				return &feature_matrix[num*num_features];
			} 
			else
			{
				SG_DEBUG( "compute feature!!!\n") ;

				ST* feat=NULL;
				dofree=false;

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
					dofree=true;
				feat=compute_feature_vector(num, len, feat);


				if (get_num_preproc())
				{
					int32_t tmp_len=len;
					ST* tmp_feat_before = feat;
					ST* tmp_feat_after = NULL;

					for (int32_t i=0; i<get_num_preproc(); i++)
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
		 * @param dofree if vector should be really deleted
		 */
		void free_feature_vector(ST* feat_vec, int32_t num, bool dofree)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (dofree)
				delete[] feat_vec ;
		}

		/** get the pointer to the feature matrix
		 * num_feat,num_vectors are returned by reference
		 *
		 * @param dst destination to store matrix in
		 * @param d1 dimension 1 of matrix
		 * @param d2 dimension 2 of matrix
		 */
		void get_fm(ST** dst, int32_t* d1, int32_t* d2)
		{
			ASSERT(feature_matrix);

			int64_t num=num_features*num_vectors;
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
		ST* get_feature_matrix(int32_t &num_feat, int32_t &num_vec)
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
		virtual void set_feature_matrix(ST* fm, int32_t num_feat, int32_t num_vec)
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
		virtual void copy_feature_matrix(ST* src, int32_t num_feat, int32_t num_vec)
		{
			free_feature_matrix();
			feature_matrix=new ST[((int64_t) num_feat)*num_vec];
			memcpy(feature_matrix, src, (sizeof(ST)*((int64_t) num_feat)*num_vec));

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

				for (int32_t i=0; i<get_num_preproc(); i++)
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
		virtual int32_t get_size() { return sizeof(ST); }


		/** get number of feature vectors
		 *
		 * @return number of feature vectors
		 */
		virtual inline int32_t  get_num_vectors() { return num_vectors; }

		/** get number of features
		 *
		 * @return number of features
		 */
		inline int32_t  get_num_features() { return num_features; }

		/** set number of features
		 *
		 * @param num number to set
		 */
		inline void set_num_features(int32_t num)
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
		inline void set_num_vectors(int32_t num)
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
		virtual bool reshape(int32_t p_num_features, int32_t p_num_vectors)
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

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space()
		{
			return num_features;
		}

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, int32_t vec_idx2)
		{
			int32_t len1, len2;
			bool free1, free2;

			ST* vec1= get_feature_vector(vec_idx1, len1, free1);
			ST* vec2= get_feature_vector(vec_idx2, len2, free2);

			float64_t result=CMath::dot(vec1, vec2, len1);

			free_feature_vector(vec1, vec_idx1, free1);
			free_feature_vector(vec2, vec_idx2, free2);

			return result;
		}

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 */
		virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len);

		/** add vector 1 multiplied with alpha to dense vector2
		 *
		 * @param alpha scalar alpha
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false)
		{
			ASSERT(vec2_len == num_features);

			int32_t vlen;
			bool vfree;
			ST* vec1=get_feature_vector(vec_idx1, vlen, vfree);

			ASSERT(vlen == num_features);

			if (abs_val)
			{
				for (int32_t i=0; i<num_features; i++)
					vec2[i]+=alpha*CMath::abs(vec1[i]);
			}
			else
			{
				for (int32_t i=0; i<num_features; i++)
					vec2[i]+=alpha*vec1[i];
			}

			free_feature_vector(vec1, vec_idx1, vfree);
		}

		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
		 */
		virtual inline int32_t get_nnz_features_for_vector(int32_t num)
		{
			return num_features;
		}

		/** @return object name */
		inline virtual const char* get_name() { return "SimpleFeatures"; }

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
		virtual ST* compute_feature_vector(int32_t num, int32_t& len, ST* target=NULL)
		{
			len=0;
			return NULL;
		}

		/// number of vectors in cache
		int32_t num_vectors;

		/// number of features in cache
		int32_t num_features;

		/** feature matrix */
		ST* feature_matrix;

		/** feature cache */
		CCache<ST>* feature_cache;
};

/** get feature type the DREAL feature can deal with
 *
 * @return feature type DREAL
 */
template<> inline EFeatureType CSimpleFeatures<float64_t>::get_feature_type()
{
	return F_DREAL;
}

/** get feature type the SHORTREAL feature can deal with
 *
 * @return feature type SHORTREAL
 */
template<> inline EFeatureType CSimpleFeatures<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

/** get feature type the SHORT feature can deal with
 *
 * @return feature type SHORT
 */
template<> inline EFeatureType CSimpleFeatures<int16_t>::get_feature_type()
{
	return F_SHORT;
}

/** get feature type the CHAR feature can deal with
 *
 * @return feature type CHAR
 */
template<> inline EFeatureType CSimpleFeatures<char>::get_feature_type()
{
	return F_CHAR;
}

/** get feature type the BYTE feature can deal with
 *
 * @return feature type BYTE
 */
template<> inline EFeatureType CSimpleFeatures<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

/** get feature type the INT feature can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType CSimpleFeatures<int32_t>::get_feature_type()
{
	return F_INT;
}

/** get feature type the WORD feature can deal with
 *
 * @return feature type WORD
 */
template<> inline EFeatureType CSimpleFeatures<uint16_t>::get_feature_type()
{
	return F_WORD;
}

/** get feature type the ULONG feature can deal with
 *
 * @return feature type ULONG
 */
template<> inline EFeatureType CSimpleFeatures<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

template<> inline float64_t CSimpleFeatures<float64_t>:: dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	float64_t* vec1= get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result=CMath::dot(vec1, vec2, num_features);

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> inline float64_t CSimpleFeatures<float32_t>:: dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	float32_t* vec1= get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result=0;

	for (int32_t i=0 ; i<num_features; i++)
		result+=vec1[i]*vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> inline float64_t CSimpleFeatures<int16_t>:: dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	int16_t* vec1= get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result=0;

	for (int32_t i=0 ; i<num_features; i++)
		result+=vec1[i]*vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> inline float64_t CSimpleFeatures<char>:: dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	char* vec1= get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result=0;

	for (int32_t i=0 ; i<num_features; i++)
		result+=vec1[i]*vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> inline float64_t CSimpleFeatures<uint8_t>:: dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	uint8_t* vec1= get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result=0;

	for (int32_t i=0 ; i<num_features; i++)
		result+=vec1[i]*vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> inline float64_t CSimpleFeatures<int32_t>:: dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	int32_t* vec1= get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result=0;

	for (int32_t i=0 ; i<num_features; i++)
		result+=vec1[i]*vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> inline float64_t CSimpleFeatures<uint16_t>:: dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	uint16_t* vec1= get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result=0;

	for (int32_t i=0 ; i<num_features; i++)
		result+=vec1[i]*vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> inline float64_t CSimpleFeatures<uint64_t>:: dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	uint64_t* vec1= get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result=0;

	for (int32_t i=0 ; i<num_features; i++)
		result+=vec1[i]*vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}
#endif
