/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Subset support written (W) 2011 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _SPARSEFEATURES__H__
#define _SPARSEFEATURES__H__

#include <string.h>
#include <stdlib.h>

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Cache.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Cache.h>
#include <shogun/io/File.h>
#include <shogun/lib/DataType.h>

#include <shogun/features/Labels.h>
#include <shogun/features/Features.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/preprocessor/SparsePreprocessor.h>

namespace shogun
{

class CFile;
class CLabels;
class CFeatures;
class CDotFeatures;
template <class ST> class CSimpleFeatures;
template <class ST> class CSparsePreprocessor;

/** @brief Template class SparseFeatures implements sparse matrices.
 *
 * Features are an array of SGSparseVector, sorted w.r.t. vec_index (increasing) and
 * withing same vec_index w.r.t. feat_index (increasing);
 *
 * Sparse feature vectors can be accessed via get_sparse_feature_vector() and
 * should be freed (this operation is a NOP in most cases) via
 * free_sparse_feature_vector().
 *
 * As this is a template class it can directly be used for different data types
 * like sparse matrices of real valued, integer, byte etc type.
 *
 * (Partly) subset access is supported for this feature type.
 * Simple use the (inherited) set_subset(), remove_subset() functions.
 * If done, all calls that work with features are translated to the subset.
 * See comments to find out whether it is supported for that method
 */
template <class ST> class CSparseFeatures : public CDotFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CSparseFeatures(int32_t size=0)
		: CDotFeatures(size), num_vectors(0), num_features(0),
			sparse_feature_matrix(NULL), feature_cache(NULL)
		{ init(); }

		/** convenience constructor that creates sparse features from
		 * the ones passed as argument
		 *
		 * @param src dense feature matrix
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 * @param copy true to copy feature matrix
		 */
		CSparseFeatures(SGSparseVector<ST>* src, int32_t num_feat, int32_t num_vec, bool copy=false)
		: CDotFeatures(0), num_vectors(0), num_features(0),
			sparse_feature_matrix(NULL), feature_cache(NULL)
		{
			init();

			if (!copy)
				set_sparse_feature_matrix(SGSparseMatrix<ST>(src, num_feat, num_vec));
			else
			{
				sparse_feature_matrix = SG_MALLOC(SGSparseVector<ST>, num_vec);
				memcpy(sparse_feature_matrix, src, sizeof(SGSparseVector<ST>)*num_vec);
				for (int32_t i=0; i< num_vec; i++)
				{
					sparse_feature_matrix[i].features = SG_MALLOC(SGSparseVectorEntry<ST>, sparse_feature_matrix[i].num_feat_entries);
					memcpy(sparse_feature_matrix[i].features, src[i].features, sizeof(SGSparseVectorEntry<ST>)*sparse_feature_matrix[i].num_feat_entries);

				}
			}
		}

		/** convenience constructor that creates sparse features from
		 * sparse features
		 *
		 * @param sparse sparse matrix
		 */
		CSparseFeatures(SGSparseMatrix<ST> sparse)
		: CDotFeatures(0), num_vectors(0), num_features(0),
			sparse_feature_matrix(NULL), feature_cache(NULL)
		{
			init();

			set_sparse_feature_matrix(sparse);
		}

		/** convenience constructor that creates sparse features from
		 * dense features
		 *
		 * @param dense dense feature matrix
		 */
		CSparseFeatures(SGMatrix<ST> dense)
		: CDotFeatures(0), num_vectors(0), num_features(0),
			sparse_feature_matrix(NULL), feature_cache(NULL)
		{
			init();

			set_full_feature_matrix(dense);
		}

		/** copy constructor */
		CSparseFeatures(const CSparseFeatures & orig)
		: CDotFeatures(orig), num_vectors(orig.num_vectors),
			num_features(orig.num_features),
			sparse_feature_matrix(orig.sparse_feature_matrix),
			feature_cache(orig.feature_cache)
		{
			init();

			if (orig.sparse_feature_matrix)
			{
				free_sparse_feature_matrix();
				sparse_feature_matrix=SG_MALLOC(SGSparseVector<ST>, num_vectors);
				memcpy(sparse_feature_matrix, orig.sparse_feature_matrix, sizeof(SGSparseVector<ST>)*num_vectors);
				for (int32_t i=0; i< num_vectors; i++)
				{
					sparse_feature_matrix[i].features=SG_MALLOC(SGSparseVectorEntry<ST>, sparse_feature_matrix[i].num_feat_entries);
					memcpy(sparse_feature_matrix[i].features, orig.sparse_feature_matrix[i].features, sizeof(SGSparseVectorEntry<ST>)*sparse_feature_matrix[i].num_feat_entries);

				}
			}

			m_subset=orig.m_subset->duplicate();
		}

		/** constructor loading features from file
		 *
		 * @param loader File object to load data from
		 */
		CSparseFeatures(CFile* loader)
		: CDotFeatures(loader), num_vectors(0), num_features(0),
			sparse_feature_matrix(NULL), feature_cache(NULL)
		{
			init();

			load(loader);
		}

		/** default destructor */
		virtual ~CSparseFeatures()
		{
			free_sparse_features();
		}

		/** free sparse feature matrix
		 *
		 * any subset is removed
		 */
		void free_sparse_feature_matrix()
        {
            clean_tsparse(sparse_feature_matrix, num_vectors);
            sparse_feature_matrix = NULL;
            num_vectors=0;
            num_features=0;
            remove_subset();
        }

		/** free sparse feature matrix and cache
		 *
		 * any subset is removed
		 */
		void free_sparse_features()
		{
			free_sparse_feature_matrix();
            delete feature_cache;
            feature_cache = NULL;
		}

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const
		{
			return new CSparseFeatures<ST>(*this);
		}

		/** get a single feature
		 *
		 * possible with subset
		 *
		 * @param num number of feature vector to retrieve
		 * @param index index of feature in this vector
		 *
		 * @return sum of features that match dimension index and 0 if none is found
		 */
		ST get_feature(int32_t num, int32_t index)
		{
			ASSERT(index>=0 && index<num_features) ;
			ASSERT(num>=0 && num<get_num_vectors()) ;

			int32_t i;
			SGSparseVector<ST> sv=get_sparse_feature_vector(num);
			ST ret = 0 ;
			
			if (sv.features)
			{
				for (i=0; i<sv.num_feat_entries; i++)
					if (sv.features[i].feat_index==index)
						ret+=sv.features[i].entry ;
			}
			
			free_sparse_feature_vector(sv, num);
			
			return ret ;
		}
		

		/** converts a sparse feature vector into a dense one
		  * preprocessed compute_feature_vector
		  * caller cleans up
		  *
		  * @param num index of feature vector
		  * @param len length is returned by reference
		  * @return dense feature vector
		  */
		ST* get_full_feature_vector(int32_t num, int32_t& len)
		{
			int32_t i;
			len=0;
			SGSparseVector<ST> sv=get_sparse_feature_vector(num);
			ST* fv=NULL;

			if (sv.features)
			{
				len=num_features;
				fv=SG_MALLOC(ST, num_features);

				for (i=0; i<num_features; i++)
					fv[i]=0;

				for (i=0; i<sv.num_feat_entries; i++)
					fv[sv.features[i].feat_index]= sv.features[i].entry;
			}

			free_sparse_feature_vector(sv, num);

			return fv;
		}

		/** get the fully expanded dense feature vector num
		  *
		  * @return dense feature vector
		  * @param num index of feature vector
		  */
		SGVector<ST> get_full_feature_vector(int32_t num)
		{
			if (num>=num_vectors)
			{
				SG_ERROR("Index out of bounds (number of vectors %d, you "
						"requested %d)\n", num_vectors, num);
			}

			SGSparseVector<ST> sv=get_sparse_feature_vector(num);

			SGVector<ST> dense;

			if (sv.features)
			{
				dense.do_free=true;
				dense.vlen=num_features;
				dense.vector=SG_MALLOC(ST, num_features);
				memset(dense.vector, 0, sizeof(ST)*num_features);

				for (int32_t i=0; i<sv.num_feat_entries; i++)
				{
					dense.vector[sv.features[i].feat_index]= sv.features[i].entry;
				}
			}

			free_sparse_feature_vector(sv, num);

			return dense;
		}


		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
		 */
		virtual inline int32_t get_nnz_features_for_vector(int32_t num)
		{
			SGSparseVector<ST> sv = get_sparse_feature_vector(num);
			int32_t len=sv.num_feat_entries;
			free_sparse_feature_vector(sv, num);
			return len;
		}

		/** get sparse feature vector
		 * for sample num from the matrix as it is if matrix is initialized,
		 * else return preprocessed compute_feature_vector
		 *
		 * possible with subset
		 *
		 * @param num index of feature vector
		 * @return sparse feature vector
		 */
		SGSparseVector<ST> get_sparse_feature_vector(int32_t num)
		{
			ASSERT(num<get_num_vectors());

			index_t real_num=subset_idx_conversion(num);

			SGSparseVector<ST> result;

			if (sparse_feature_matrix)
			{
				result=sparse_feature_matrix[real_num];
				result.do_free=false;
				return result;
			} 
			else
			{
				result.do_free=false;

				if (feature_cache)
				{
					result.features=feature_cache->lock_entry(num);

					if (result.features)
						return result;
					else
					{
						result.features=feature_cache->set_entry(num);
					}
				}

				if (!result.features)
					result.do_free=true;

				result.features=compute_sparse_feature_vector(num,
					result.num_feat_entries, result.features);


				if (get_num_preprocessors())
				{
					int32_t tmp_len=result.num_feat_entries;
					SGSparseVectorEntry<ST>* tmp_feat_before=result.features;
					SGSparseVectorEntry<ST>* tmp_feat_after = NULL;

					for (int32_t i=0; i<get_num_preprocessors(); i++)
					{
						//tmp_feat_after=((CSparsePreprocessor<ST>*) get_preproc(i))->apply_to_feature_vector(tmp_feat_before, tmp_len);

						if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
							SG_FREE(tmp_feat_before);
						tmp_feat_before=tmp_feat_after;
					}

					memcpy(result.features, tmp_feat_after,
							sizeof(SGSparseVectorEntry<ST>)*tmp_len);

					SG_FREE(tmp_feat_after);
					result.num_feat_entries=tmp_len ;
					SG_DEBUG( "len: %d len2: %d\n", result.num_feat_entries, num_features);
				}
				result.vec_index=num;
				return result ;
			}
		}


		/** compute the dot product between two sparse feature vectors
		 * alpha * vec^T * vec
		 *
		 * @param alpha scalar to multiply with
		 * @param avec first sparse feature vector
		 * @param alen avec's length
		 * @param bvec second sparse feature vector
		 * @param blen bvec's length
		 * @return dot product between the two sparse feature vectors
		 */
		static ST sparse_dot(ST alpha, SGSparseVectorEntry<ST>* avec, int32_t alen, SGSparseVectorEntry<ST>* bvec, int32_t blen)
		{
			ST result=0;

			//result remains zero when one of the vectors is non existent
			if (avec && bvec)
			{
				if (alen<=blen)
				{
					int32_t j=0;
					for (int32_t i=0; i<alen; i++)
					{
						int32_t a_feat_idx=avec[i].feat_index;

						while ( (j<blen) && (bvec[j].feat_index < a_feat_idx) )
							j++;

						if ( (j<blen) && (bvec[j].feat_index == a_feat_idx) )
						{
							result+= avec[i].entry * bvec[j].entry;
							j++;
						}
					}
				}
				else
				{
					int32_t j=0;
					for (int32_t i=0; i<blen; i++)
					{
						int32_t b_feat_idx=bvec[i].feat_index;

						while ( (j<alen) && (avec[j].feat_index < b_feat_idx) )
							j++;

						if ( (j<alen) && (avec[j].feat_index == b_feat_idx) )
						{
							result+= bvec[i].entry * avec[j].entry;
							j++;
						}
					}
				}

				result*=alpha;
			}

			return result;
		}

		/** compute the dot product between dense weights and a sparse feature vector
		 * alpha * sparse^T * w + b
		 *
		 * possible with subset
		 *
		 * @param alpha scalar to multiply with
		 * @param num index of feature vector
		 * @param vec dense vector to compute dot product with
		 * @param dim length of the dense vector
		 * @param b bias
		 * @return dot product between dense weights and a sparse feature vector
		 */
		ST dense_dot(ST alpha, int32_t num, ST* vec, int32_t dim, ST b)
		{
			ASSERT(vec);
			ASSERT(dim==num_features);
			ST result=b;

			SGSparseVector<ST> sv=get_sparse_feature_vector(num);

			if (sv.features)
			{
				for (int32_t i=0; i<sv.num_feat_entries; i++)
				{
					result+=alpha*vec[sv.features[i].feat_index]
						*sv.features[i].entry;
				}
			}

			free_sparse_feature_vector(sv, num);
			return result;
		}

		/** add a sparse feature vector onto a dense one
		 * dense+=alpha*sparse
		 *
		 * possible with subset
		 *
		 @param alpha scalar to multiply with
		 @param num index of feature vector
		 @param vec dense vector
		 @param dim length of the dense vector
		 @param abs_val if true, do dense+=alpha*abs(sparse)
		 */
		void add_to_dense_vec(float64_t alpha, int32_t num, float64_t* vec, int32_t dim, bool abs_val=false)
		{
			ASSERT(vec);
			if (dim!=num_features)
			{
				SG_ERROR("dimension of vec (=%d) does not match number of features (=%d)\n",
						dim, num_features);
			}

			SGSparseVector<ST> sv=get_sparse_feature_vector(num);

			if (sv.features)
			{
				if (abs_val)
				{
					for (int32_t i=0; i<sv.num_feat_entries; i++)
					{
						vec[sv.features[i].feat_index]+=alpha
							*CMath::abs(sv.features[i].entry);
					}
				}
				else
				{
					for (int32_t i=0; i<sv.num_feat_entries; i++)
					{
						vec[sv.features[i].feat_index]+=alpha
								*sv.features[i].entry;
					}
				}
			}

			free_sparse_feature_vector(sv, num);
		}

		/** free sparse feature vector
		 *
		 * possible with subset
		 *
		 * @param feat_vec feature vector to free
		 * @param num index of this vector in the cache
		 */
		void free_sparse_feature_vector(SGSparseVector<ST> vec, int32_t num)
		{
			if (feature_cache)
				feature_cache->unlock_entry(subset_idx_conversion(num));

			vec.free_vector();
		} 

		/** get the pointer to the sparse feature matrix
		 * num_feat,num_vectors are returned by reference
		 *
		 * not possible with subset
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return feature matrix
		 */
		SGSparseVector<ST>* get_sparse_feature_matrix(int32_t &num_feat, int32_t &num_vec)
		{
			if (m_subset)
				SG_ERROR("get_sparse_feature_matrix() not allowed with subset\n");

			num_feat=num_features;
			num_vec=num_vectors;

			return sparse_feature_matrix;
		}

		/** get the sparse feature matrix
		 *
		 * not possible with subset
		 *
		 * @return sparse matrix
		 *
		 */
        SGSparseMatrix<ST> get_sparse_feature_matrix()
        {
        	if (m_subset)
				SG_ERROR("get_sparse_feature_matrix() not allowed with subset\n");

            SGSparseMatrix<ST> sm;
            sm.sparse_matrix=get_sparse_feature_matrix(sm.num_features, sm.num_vectors);
            return sm;
        }

		/** clean SGSparseVector
		 *
		 * @param sfm sparse feature matrix
		 * @param num_vec number of vectors in matrix
		 */
		static void clean_tsparse(SGSparseVector<ST>* sfm, int32_t num_vec)
		{
			if (sfm)
			{
				for (int32_t i=0; i<num_vec; i++)
					SG_FREE(sfm[i].features);

				SG_FREE(sfm);
			}
		}

		/** get a transposed copy of the features
		 *
		 * possible with subset
		 *
		 * @return transposed copy
		 */
		CSparseFeatures<ST>* get_transposed()
		{
			int32_t num_feat;
			int32_t num_vec;
			SGSparseVector<ST>* s=get_transposed(num_feat, num_vec);
			return new CSparseFeatures<ST>(s, num_feat, num_vec);
		}

		/** compute and return the transpose of the sparse feature matrix
		 * which will be prepocessed.
		 * num_feat, num_vectors are returned by reference
		 * caller has to clean up
		 *
		 * possible with subset
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return transposed sparse feature matrix
		 */
		SGSparseVector<ST>* get_transposed(int32_t &num_feat, int32_t &num_vec)
		{
			num_feat=get_num_vectors();
			num_vec=num_features;

			int32_t* hist=SG_MALLOC(int32_t, num_features);
			memset(hist, 0, sizeof(int32_t)*num_features);

			// count how lengths of future feature vectors
			for (int32_t v=0; v<num_feat; v++)
			{
				SGSparseVector<ST> sv=get_sparse_feature_vector(v);

				for (int32_t i=0; i<sv.num_feat_entries; i++)
					hist[sv.features[i].feat_index]++;

				free_sparse_feature_vector(sv, v);
			}

			// allocate room for future feature vectors
			SGSparseVector<ST>* sfm=SG_MALLOC(SGSparseVector<ST>, num_vec);
			for (int32_t v=0; v<num_vec; v++)
			{
				sfm[v].features= SG_MALLOC(SGSparseVectorEntry<ST>, hist[v]);
				sfm[v].num_feat_entries=hist[v];
				sfm[v].vec_index=v;
			}

			// fill future feature vectors with content
			memset(hist,0,sizeof(int32_t)*num_features);
			for (int32_t v=0; v<num_feat; v++)
			{
				SGSparseVector<ST> sv=get_sparse_feature_vector(v);

				for (int32_t i=0; i<sv.num_feat_entries; i++)
				{
					int32_t vidx=sv.features[i].feat_index;
					int32_t fidx=v;
					sfm[vidx].features[hist[vidx]].feat_index=fidx;
					sfm[vidx].features[hist[vidx]].entry=sv.features[i].entry;
					hist[vidx]++;
				}

				free_sparse_feature_vector(sv, v);
			}

			SG_FREE(hist);
			return sfm;
		}

		/** set sparse feature matrix
		 *
		 * not possible with subset
		 *
		 * @param sm sparse feature matrix
		 *
		 */
        void set_sparse_feature_matrix(SGSparseMatrix<ST> sm)
        {
        	if (m_subset)
				SG_ERROR("set_sparse_feature_matrix() not allowed with subset\n");


			free_sparse_feature_matrix();
			sm.own_matrix();

			sparse_feature_matrix=sm.sparse_matrix;
			num_features=sm.num_features;
			num_vectors=sm.num_vectors;
        }

		/** gets a copy of a full feature matrix
		 *
		 * possible with subset
		 *
		 * @return full dense feature matrix
		 */
		SGMatrix<ST> get_full_feature_matrix()
		{
			SGMatrix<ST> full;

			SG_INFO( "converting sparse features to full feature matrix of %ld x %ld entries\n", num_vectors, num_features);
			full.num_rows=num_features;
			full.num_cols=get_num_vectors();
			full.do_free=true;
			full.matrix=SG_MALLOC(ST, int64_t(num_features)*get_num_vectors());

			memset(full.matrix, 0, size_t(num_features)*size_t(get_num_vectors())*sizeof(ST));

			for (int32_t v=0; v<full.num_cols; v++)
			{
				SGSparseVector<ST> current=
					sparse_feature_matrix[subset_idx_conversion(v)];

				for (int32_t f=0; f<current.num_feat_entries; f++)
				{
					int64_t offs=(current.vec_index*num_features)
							+current.features[f].feat_index;

					full.matrix[offs]=current.features[f].entry;
				}
			}

			return full;
		}

		/** creates a sparse feature matrix from a full dense feature matrix
		 * necessary to set feature_matrix, num_features and num_vectors
		 * where num_features is the column offset, and columns are linear in memory
		 * see above for definition of sparse_feature_matrix
		 *
		 * any subset is removed before
		 *
		 * @param full full feature matrix
		 */
		virtual bool set_full_feature_matrix(SGMatrix<ST> full)
		{
			remove_subset();

			ST* src=full.matrix;
			int32_t num_feat=full.num_rows;
			int32_t num_vec=full.num_cols;

			free_sparse_feature_matrix();
			bool result=true;
			num_features=num_feat;
			num_vectors=num_vec;

			SG_INFO("converting dense feature matrix to sparse one\n");
			int32_t* num_feat_entries=SG_MALLOC(int, num_vectors);

			if (num_feat_entries)
			{
				int64_t num_total_entries=0;

				// count nr of non sparse features
				for (int32_t i=0; i< num_vec; i++)
				{
					num_feat_entries[i]=0;
					for (int32_t j=0; j< num_feat; j++)
					{
						if (src[i*((int64_t) num_feat) + j] != 0)
							num_feat_entries[i]++;
					}
				}

				if (num_vec>0)
				{
					sparse_feature_matrix=SG_MALLOC(SGSparseVector<ST>, num_vec);

					if (sparse_feature_matrix)
					{
						for (int32_t i=0; i< num_vec; i++)
						{
							sparse_feature_matrix[i].vec_index=i;
							sparse_feature_matrix[i].num_feat_entries=0;
							sparse_feature_matrix[i].features= NULL;

							if (num_feat_entries[i]>0)
							{
								sparse_feature_matrix[i].features= SG_MALLOC(SGSparseVectorEntry<ST>, num_feat_entries[i]);

								if (!sparse_feature_matrix[i].features)
								{
									SG_INFO( "allocation of features failed\n");
									return false;
								}

								sparse_feature_matrix[i].num_feat_entries=num_feat_entries[i];
								int32_t sparse_feat_idx=0;

								for (int32_t j=0; j< num_feat; j++)
								{
									int64_t pos= i*num_feat + j;

									if (src[pos] != 0)
									{
										sparse_feature_matrix[i].features[sparse_feat_idx].entry=src[pos];
										sparse_feature_matrix[i].features[sparse_feat_idx].feat_index=j;
										sparse_feat_idx++;
										num_total_entries++;
									}
								}
							}
						}
					}
					else
					{
						SG_ERROR( "allocation of sparse feature matrix failed\n");
						result=false;
					}

					SG_INFO( "sparse feature matrix has %ld entries (full matrix had %ld, sparsity %2.2f%%)\n",
							num_total_entries, int64_t(num_feat)*num_vec, (100.0*num_total_entries)/(int64_t(num_feat)*num_vec));
				}
				else
				{
					SG_ERROR( "huh ? zero size matrix given ?\n");
					result=false;
				}
			}
			SG_FREE(num_feat_entries);
			return result;
		}

		/** apply preprocessor
		 *
		 * possible with subset
		 *
		 * @param force_preprocessing if preprocssing shall be forced
		 * @return if applying was successful
		 */
		virtual bool apply_preprocessor(bool force_preprocessing=false)
		{
			SG_INFO( "force: %d\n", force_preprocessing);

			if ( sparse_feature_matrix && get_num_preprocessors() )
			{
				for (int32_t i=0; i<get_num_preprocessors(); i++)
				{
					if ( (!is_preprocessed(i) || force_preprocessing) )
					{
						set_preprocessed(i);
						SG_INFO( "preprocessing using preproc %s\n", get_preprocessor(i)->get_name());
						if (((CSparsePreprocessor<ST>*) get_preprocessor(i))->apply_to_sparse_feature_matrix(this) == NULL)
							return false;
					}
					return true;
				}
				return true;
			}
			else
			{
				SG_WARNING( "no sparse feature matrix available or features already preprocessed - skipping.\n");
				return false;
			}
		}

		/** get memory footprint of one feature
		 *
		 * @return memory footprint of one feature
		 */
		virtual int32_t get_size() { return sizeof(ST); }

		/** obtain sparse features from simple features
		 *
		 * subset on input is ignored, subset of this instance is removed
		 *
		 * @param sf simple features
		 * @return if obtaining was successful
		 */
		bool obtain_from_simple(CSimpleFeatures<ST>* sf)
		{
			SGMatrix<ST> fm=sf->get_feature_matrix();
			ASSERT(fm.matrix && fm.num_cols>0 && fm.num_rows>0);

			return set_full_feature_matrix(fm);
		}

		/** get number of feature vectors, possibly of subset
		 *
		 * @return number of feature vectors
		 */
		virtual inline int32_t  get_num_vectors() const
		{
			return m_subset ? m_subset->get_size() : num_vectors;
		}

		/** get number of features
		 *
		 * @return number of features
		 */
		inline int32_t  get_num_features() { return num_features; }

		/** set number of features
		 *
		 * Sometimes when loading sparse features not all possible dimensions
		 * are used. This may pose a problem to classifiers when being applied
		 * to higher dimensional test-data. This function allows to
		 * artificially explode the feature space
		 *
		 * @param num the number of features, must be larger
		 *        than the current number of features
		 * @return previous number of features
		 */
		inline int32_t set_num_features(int32_t num)
		{
			int32_t n=num_features;
			ASSERT(n<=num);
			num_features=num;
			return num_features;
		}

		/** get feature class
		 *
		 * @return feature class SPARSE
		 */
		inline virtual EFeatureClass get_feature_class() { return C_SPARSE; }

		/** get feature type
		 *
		 * @return templated feature type
		 */
		inline virtual EFeatureType get_feature_type();

		/** free feature vector
		 *
		 * possible with subset
		 *
		 * @param feat_vec feature vector to free
		 * @param num index of vector in cache
		 */
		void free_feature_vector(SGSparseVector<ST> vec, int32_t num)
		{
			if (feature_cache)
				feature_cache->unlock_entry(subset_idx_conversion(num));

			vec.free_vector();
		}

		/** get number of non-zero entries in sparse feature matrix
		 *
		 * @return number of non-zero entries in sparse feature matrix
		 */
		int64_t get_num_nonzero_entries()
		{
			int64_t num=0;
			index_t num_vec=get_num_vectors();
			for (int32_t i=0; i<num_vec; i++)
				num+=sparse_feature_matrix[subset_idx_conversion(i)].num_feat_entries;

			return num;
		}

		/** compute a^2 on all feature vectors
		 *
		 * possible with subset
		 *
		 * @param sq the square for each vector is stored in here
		 * @return the square for each vector
		 */
		float64_t* compute_squared(float64_t* sq)
		{
			ASSERT(sq);

			index_t num_vec=get_num_vectors();
			for (int32_t i=0; i<num_vec; i++)
			{
				sq[i]=0;
				SGSparseVector<ST> vec=get_sparse_feature_vector(i);

				for (int32_t j=0; j<vec.num_feat_entries; j++)
					sq[i]+=vec.features[j].entry*vec.features[j].entry;

				free_feature_vector(vec, i);
			}

			return sq;
		}

		/** compute (a-b)^2 (== a^2+b^2-2ab)
		 * usually called by kernels'/distances' compute functions
		 * works on two feature vectors, although it is a member of a single
		 * feature: can either be called by lhs or rhs.
		 *
		 * possible wiht subsets on lhs or rhs
		 *
		 * @param lhs left-hand side features
		 * @param sq_lhs squared values of left-hand side
		 * @param idx_a index of left-hand side's vector to compute
		 * @param rhs right-hand side features
		 * @param sq_rhs squared values of right-hand side
		 * @param idx_b index of right-hand side's vector to compute
		 */
		float64_t compute_squared_norm(CSparseFeatures<float64_t>* lhs, float64_t* sq_lhs, int32_t idx_a, CSparseFeatures<float64_t>* rhs, float64_t* sq_rhs, int32_t idx_b)
		{
			int32_t i,j;
			ASSERT(lhs);
			ASSERT(rhs);

			SGSparseVector<float64_t> avec=lhs->get_sparse_feature_vector(idx_a);
			SGSparseVector<float64_t> bvec=rhs->get_sparse_feature_vector(idx_b);
			ASSERT(avec.features);
			ASSERT(bvec.features);

			float64_t result=sq_lhs[idx_a]+sq_rhs[idx_b];

			if (avec.num_feat_entries<=bvec.num_feat_entries)
			{
				j=0;
				for (i=0; i<avec.num_feat_entries; i++)
				{
					int32_t a_feat_idx=avec.features[i].feat_index;

					while ((j<bvec.num_feat_entries)
							&&(bvec.features[j].feat_index<a_feat_idx))
						j++;

					if ((j<bvec.num_feat_entries)
							&&(bvec.features[j].feat_index==a_feat_idx))
					{
						result-=2*(avec.features[i].entry*bvec.features[j].entry);
						j++;
					}
				}
			}
			else
			{
				j=0;
				for (i=0; i<bvec.num_feat_entries; i++)
				{
					int32_t b_feat_idx=bvec.features[i].feat_index;

					while ((j<avec.num_feat_entries)
							&&(avec.features[j].feat_index<b_feat_idx))
						j++;

					if ((j<avec.num_feat_entries)
							&&(avec.features[j].feat_index==b_feat_idx))
					{
						result-=2*(bvec.features[i].entry*avec.features[j].entry);
						j++;
					}
				}
			}

			((CSparseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a);
			((CSparseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b);

			return CMath::abs(result);
		}

		/** load features from file
		 *
		 * any subset is removed before
		 *
		 * @param loader File object to load data from
		 */
		void load(CFile* loader);

		/** save features to file
		 *
		 * not possible with subset
		 *
		 * @param writer File object to write data to
		 */
		void save(CFile* writer);

		/** load features from file
		 *
		 * any subset is removed before
		 *
		 * @param fname filename to load from
		 * @param do_sort_features if true features will be sorted to ensure they
		 * 		 are in ascending order
		 * @return label object with corresponding labels
		 */
		CLabels* load_svmlight_file(char* fname, bool do_sort_features=true)
		{
			remove_subset();

			CLabels* lab=NULL;

			size_t blocksize=1024*1024;
			size_t required_blocksize=blocksize;
			uint8_t* dummy=SG_MALLOC(uint8_t, blocksize);
			FILE* f=fopen(fname, "ro");

			if (f)
			{
				free_sparse_feature_matrix();
				num_vectors=0;
				num_features=0;

				SG_INFO("counting line numbers in file %s\n", fname);
				size_t sz=blocksize;
				size_t block_offs=0;
				size_t old_block_offs=0;
				fseek(f, 0, SEEK_END);
				size_t fsize=ftell(f);
				rewind(f);

				while (sz == blocksize)
				{
					sz=fread(dummy, sizeof(uint8_t), blocksize, f);
					bool contains_cr=false;
					for (size_t i=0; i<sz; i++)
					{
						block_offs++;
						if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
						{
							num_vectors++;
							contains_cr=true;
							required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs+1);
							old_block_offs=block_offs;
						}
					}
					SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t");
				}

				SG_INFO("found %d feature vectors\n", num_vectors);
				SG_FREE(dummy);
				blocksize=required_blocksize;
				dummy = SG_MALLOC(uint8_t, blocksize+1); //allow setting of '\0' at EOL

				lab=new CLabels(num_vectors);
				sparse_feature_matrix=SG_MALLOC(SGSparseVector<ST>, num_vectors);

				rewind(f);
				sz=blocksize;
				int32_t lines=0;
				while (sz == blocksize)
				{
					sz=fread(dummy, sizeof(uint8_t), blocksize, f);

					size_t old_sz=0;
					for (size_t i=0; i<sz; i++)
					{
						if (i==sz-1 && dummy[i]!='\n' && sz==blocksize)
						{
							size_t len=i-old_sz+1;
							uint8_t* data=&dummy[old_sz];

							for (int32_t j=0; j<len; j++)
								dummy[j]=data[j];

							sz=fread(dummy+len, sizeof(uint8_t), blocksize-len, f);
							i=0;
							old_sz=0;
							sz+=len;
						}

						if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
						{

							size_t len=i-old_sz;
							uint8_t* data=&dummy[old_sz];

							int32_t dims=0;
							for (int32_t j=0; j<len; j++)
							{
								if (data[j]==':')
									dims++;
							}

							if (dims<=0)
							{
								SG_ERROR("Error in line %d - number of"
										" dimensions is %d line is %d characters"
										" long\n line_content:'%.*s'\n", lines,
										dims, len, len, (const char*) data);
							}

							SGSparseVectorEntry<ST>* feat=SG_MALLOC(SGSparseVectorEntry<ST>, dims);
							int32_t j=0;
							for (; j<len; j++)
							{
								if (data[j]==' ')
								{
									data[j]='\0';

									lab->set_label(lines, atof((const char*) data));
									break;
								}
							}

							int32_t d=0;
							j++;
							uint8_t* start=&data[j];
							for (; j<len; j++)
							{
								if (data[j]==':')
								{
									data[j]='\0';

									feat[d].feat_index=(int32_t) atoi((const char*) start)-1;
									num_features=CMath::max(num_features, feat[d].feat_index+1);

									j++;
									start=&data[j];
									for (; j<len; j++)
									{
										if (data[j]==' ' || data[j]=='\n')
										{
											data[j]='\0';
											feat[d].entry=(ST) atof((const char*) start);
											d++;
											break;
										}
									}

									if (j==len)
									{
										data[j]='\0';
										feat[dims-1].entry=(ST) atof((const char*) start);
									}

									j++;
									start=&data[j];
								}
							}

							sparse_feature_matrix[lines].vec_index=lines;
							sparse_feature_matrix[lines].num_feat_entries=dims;
							sparse_feature_matrix[lines].features=feat;

							old_sz=i+1;
							lines++;
							SG_PROGRESS(lines, 0, num_vectors, 1, "LOADING:\t");
						}
					}
				}
				SG_INFO("file successfully read\n");
				fclose(f);
			}

			SG_FREE(dummy);

			if (do_sort_features)
				sort_features();

			return lab;
		}

		/** ensure that features occur in ascending order, only call when no
		 * preprocessors are attached
		 *
		 * not possiblwe with subset
		 * */
		void sort_features()
		{
			if (m_subset)
				SG_ERROR("sort_features() not allowed with subset\n");

			ASSERT(get_num_preprocessors()==0);

			if (!sparse_feature_matrix)
				SG_ERROR("Requires sparse feature matrix to be available in-memory\n");

			for (int32_t i=0; i<num_vectors; i++)
			{
				int32_t len=sparse_feature_matrix[i].num_feat_entries;

				if (!len)
					continue;

				SGSparseVectorEntry<ST>* sf_orig=sparse_feature_matrix[i].features;
				int32_t* feat_idx=SG_MALLOC(int32_t, len);
				int32_t* orig_idx=SG_MALLOC(int32_t, len);

				for (int j=0; j<len; j++)
				{
					feat_idx[j]=sf_orig[j].feat_index;
					orig_idx[j]=j;
				}

				CMath::qsort_index(feat_idx, orig_idx, len);

				SGSparseVectorEntry<ST>* sf_new= SG_MALLOC(SGSparseVectorEntry<ST>, len);
				for (int j=0; j<len; j++)
					sf_new[j]=sf_orig[orig_idx[j]];

				sparse_feature_matrix[i].features=sf_new;

				// sanity check
				for (int j=0; j<len-1; j++)
					ASSERT(sf_new[j].feat_index<sf_new[j+1].feat_index);

				SG_FREE(orig_idx);
				SG_FREE(feat_idx);
				SG_FREE(sf_orig);
			}
		}

		/** write features to file using svm light format
		 *
		 * not possible with subset
		 *
		 * @param fname filename to write to
		 * @param label Label object (number of labels must correspond to number of features)
		 * @return true if successful
		 */
		bool write_svmlight_file(char* fname, CLabels* label)
		{
			if (m_subset)
				SG_ERROR("write_svmlight_file() not allowed with subset\n");

			ASSERT(label);
			int32_t num=label->get_num_labels();
			ASSERT(num>0);
			ASSERT(num==num_vectors);

			FILE* f=fopen(fname, "wb");

			if (f)
			{
				for (int32_t i=0; i<num; i++)
				{
					fprintf(f, "%d ", (int32_t) label->get_int_label(i));

					SGSparseVectorEntry<ST>* vec = sparse_feature_matrix[i].features;
					int32_t num_feat = sparse_feature_matrix[i].num_feat_entries;

					for (int32_t j=0; j<num_feat; j++)
					{
						if (j<num_feat-1)
							fprintf(f, "%d:%f ", (int32_t) vec[j].feat_index+1, (double) vec[j].entry);
						else
							fprintf(f, "%d:%f\n", (int32_t) vec[j].feat_index+1, (double) vec[j].entry);
					}
				}

				fclose(f);
				return true;
			}
			return false;
		}

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space() const
		{
			return num_features;
		}

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * possible with subset of this instance and of DotFeatures
		 *
		 * @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
		{
			ASSERT(df);
			ASSERT(df->get_feature_type() == get_feature_type());
			ASSERT(df->get_feature_class() == get_feature_class());
			CSparseFeatures<ST>* sf = (CSparseFeatures<ST>*) df;

			SGSparseVector<ST> avec=get_sparse_feature_vector(vec_idx1);
			SGSparseVector<ST> bvec=sf->get_sparse_feature_vector(vec_idx2);

			float64_t result=sparse_dot(1, avec.features, avec.num_feat_entries,
				bvec.features, bvec.num_feat_entries);

			free_sparse_feature_vector(avec, vec_idx1);
			sf->free_sparse_feature_vector(bvec, vec_idx2);

			return result;
		}

		/** compute dot product between vector1 and a dense vector
		 *
		 * possible with subset
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 */
		virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
		{
			ASSERT(vec2);
			if (vec2_len!=num_features)
			{
				SG_ERROR("dimension of vec2 (=%d) does not match number of features (=%d)\n",
						vec2_len, num_features);
			}
			float64_t result=0;

			SGSparseVector<ST> sv=get_sparse_feature_vector(vec_idx1);

			if (sv.features)
			{
				for (int32_t i=0; i<sv.num_feat_entries; i++)
					result+=vec2[sv.features[i].feat_index]*sv.features[i].entry;
			}

			free_sparse_feature_vector(sv, vec_idx1);

			return result;
		}

		/** iterator for sparse features */
		struct sparse_feature_iterator
		{
			/** feature vector */
			SGSparseVector<ST> sv;

			/** index */
			int32_t index;

			/** print details of iterator (for debugging purposes)*/
			void print_info()
			{
				SG_SPRINT("sv=%p, vidx=%d, num_feat_entries=%d, index=%d\n",
						sv.features, sv.vec_index, sv.num_feat_entries, index);
			}
		};

		/** iterate over the non-zero features
		 *
		 * call get_feature_iterator first, followed by get_next_feature and
		 * free_feature_iterator to cleanup
		 *
		 * possible with subset
		 *
		 * @param vector_index the index of the vector over whose components to
		 * 			iterate over
		 * @return feature iterator (to be passed to get_next_feature)
		 */
		virtual void* get_feature_iterator(int32_t vector_index)
		{
			if (vector_index>=get_num_vectors())
			{
				SG_ERROR("Index out of bounds (number of vectors %d, you "
						"requested %d)\n", get_num_vectors(), vector_index);
			}

			if (!sparse_feature_matrix)
				SG_ERROR("Requires a in-memory feature matrix\n");

			sparse_feature_iterator* it=SG_MALLOC(sparse_feature_iterator, 1);
			it->sv=get_sparse_feature_vector(vector_index);
			it->index=0;

			return it;
		}

		/** iterate over the non-zero features
		 *
		 * call this function with the iterator returned by get_first_feature
		 * and call free_feature_iterator to cleanup
		 *
		 * @param index is returned by reference (-1 when not available)
		 * @param value is returned by reference
		 * @param iterator as returned by get_first_feature
		 * @return true if a new non-zero feature got returned
		 */
		virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator)
		{
			sparse_feature_iterator* it=(sparse_feature_iterator*) iterator;
			if (!it || it->index>=it->sv.num_feat_entries)
				return false;

			int32_t i=it->index++;

			index=it->sv.features[i].feat_index;
			value=(float64_t) it->sv.features[i].entry;

			return true;
		}

		/** clean up iterator
		 * call this function with the iterator returned by get_first_feature
		 *
		 * @param iterator as returned by get_first_feature
		 */
		virtual void free_feature_iterator(void* iterator)
		{
			if (!iterator)
				return;

			sparse_feature_iterator* it=(sparse_feature_iterator*) iterator;
			free_sparse_feature_vector(it->sv, it->sv.vec_index);
			SG_FREE(it);
		}

		/** Creates a new CFeatures instance containing copies of the elements
		 * which are specified by the provided indices.
		 *
		 * @param indices indices of feature elements to copy
		 * @return new CFeatures instance with copies of feature data
		 */
		virtual CFeatures* copy_subset(SGVector<index_t> indices)
		{
			SGSparseMatrix<ST> matrix_copy=SGSparseMatrix<ST>(indices.vlen,
				get_dim_feature_space());

			for (index_t i=0; i<indices.vlen; ++i)
			{
				/* index to copy */
				index_t index=indices.vector[i];

				/* copy sparse vector TODO THINK ABOUT VECTOR INDEX (i or vec.index*/
				SGSparseVector<ST> current=get_sparse_feature_vector(index);
				matrix_copy.sparse_matrix[i]=SGSparseVector<ST>(
					current.num_feat_entries, current.vec_index);

				/* copy entries */
				memcpy(matrix_copy.sparse_matrix[i].features, current.features,
					sizeof(SGSparseVectorEntry<ST>)*current.num_feat_entries);

				free_sparse_feature_vector(current, index);
			}

			return new CSparseFeatures<ST>(matrix_copy);
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "SparseFeatures"; }

	protected:
		/** compute feature vector for sample num
		 * if target is set the vector is written to target
		 * len is returned by reference
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param num num
		 * @param len len
		 * @param target target
		 */
		virtual SGSparseVectorEntry<ST>* compute_sparse_feature_vector(int32_t num,
			int32_t& len, SGSparseVectorEntry<ST>* target=NULL)
		{
			SG_NOTIMPLEMENTED;

			len=0;
			return NULL;
		}

	private:
		void init(void)
		{
			set_generic<ST>();

			m_parameters->add_vector(&sparse_feature_matrix, &num_vectors,
					"sparse_feature_matrix",
					"Array of sparse vectors.");
			m_parameters->add(&num_features, "num_features",
					"Total number of features.");
		}


	protected:

		/// total number of vectors
		int32_t num_vectors;

		/// total number of features
		int32_t num_features;

		/// array of sparse vectors of size num_vectors
		SGSparseVector<ST>* sparse_feature_matrix;

		/** feature cache */
		CCache< SGSparseVectorEntry<ST> >* feature_cache;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#define GET_FEATURE_TYPE(sg_type, f_type)									\
template<> inline EFeatureType CSparseFeatures<sg_type>::get_feature_type()	\
{																			\
	return f_type;															\
}
GET_FEATURE_TYPE(bool, F_BOOL)
GET_FEATURE_TYPE(char, F_CHAR)
GET_FEATURE_TYPE(uint8_t, F_BYTE)
GET_FEATURE_TYPE(int8_t, F_BYTE)
GET_FEATURE_TYPE(int16_t, F_SHORT)
GET_FEATURE_TYPE(uint16_t, F_WORD)
GET_FEATURE_TYPE(int32_t, F_INT)
GET_FEATURE_TYPE(uint32_t, F_UINT)
GET_FEATURE_TYPE(int64_t, F_LONG)
GET_FEATURE_TYPE(uint64_t, F_ULONG)
GET_FEATURE_TYPE(float32_t, F_SHORTREAL)
GET_FEATURE_TYPE(float64_t, F_DREAL)
GET_FEATURE_TYPE(floatmax_t, F_LONGREAL)
#undef GET_FEATURE_TYPE

#define LOAD(fname, sg_type)											\
template<> inline void CSparseFeatures<sg_type>::load(CFile* loader)	\
{																		\
	remove_subset();													\
	SG_SET_LOCALE_C;													\
	ASSERT(loader);														\
	SGSparseVector<sg_type>* matrix=NULL;										\
	int32_t num_feat=0;													\
	int32_t num_vec=0;													\
	loader->fname(matrix, num_feat, num_vec);							\
	set_sparse_feature_matrix(SGSparseMatrix<sg_type>(matrix, num_feat, num_vec));				\
	SG_RESET_LOCALE;													\
}
LOAD(get_sparse_matrix, bool)
LOAD(get_sparse_matrix, char)
LOAD(get_sparse_matrix, uint8_t)
LOAD(get_int8_sparsematrix, int8_t)
LOAD(get_sparse_matrix, int16_t)
LOAD(get_sparse_matrix, uint16_t)
LOAD(get_sparse_matrix, int32_t)
LOAD(get_uint_sparsematrix, uint32_t)
LOAD(get_long_sparsematrix, int64_t)
LOAD(get_ulong_sparsematrix, uint64_t)
LOAD(get_sparse_matrix, float32_t)
LOAD(get_sparse_matrix, float64_t)
LOAD(get_longreal_sparsematrix, floatmax_t)
#undef LOAD

#define WRITE(fname, sg_type)											\
template<> inline void CSparseFeatures<sg_type>::save(CFile* writer)	\
{																		\
	if (m_subset)														\
		SG_ERROR("save() not allowed with subset\n");					\
	SG_SET_LOCALE_C;													\
	ASSERT(writer);														\
	writer->fname(sparse_feature_matrix, num_features, num_vectors);	\
	SG_RESET_LOCALE;													\
}
WRITE(set_sparse_matrix, bool)
WRITE(set_sparse_matrix, char)
WRITE(set_sparse_matrix, uint8_t)
WRITE(set_int8_sparsematrix, int8_t)
WRITE(set_sparse_matrix, int16_t)
WRITE(set_sparse_matrix, uint16_t)
WRITE(set_sparse_matrix, int32_t)
WRITE(set_uint_sparsematrix, uint32_t)
WRITE(set_long_sparsematrix, int64_t)
WRITE(set_ulong_sparsematrix, uint64_t)
WRITE(set_sparse_matrix, float32_t)
WRITE(set_sparse_matrix, float64_t)
WRITE(set_longreal_sparsematrix, floatmax_t)
#undef WRITE
#endif // DOXYGEN_SHOULD_SKIP_THIS
}
#endif /* _SPARSEFEATURES__H__ */
