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

#ifndef _SPARSEFEATURES__H__
#define _SPARSEFEATURES__H__

#include <string.h>
#include <stdlib.h>

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/Cache.h"
#include "lib/io.h"
#include "lib/Cache.h"

#include "features/Labels.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "features/RealFeatures.h"
#include "preproc/SparsePreProc.h"

//features are an array of TSparse, sorted w.r.t. vec_index (increasing) and
//withing same vec_index w.r.t. feat_index (increasing);

template <class ST> class CSparsePreProc;

/** template class TSparseEntry */
template <class ST> struct TSparseEntry
{
	/** feature index */
	INT feat_index;
	/** entry */
	ST entry;
};

/** template class TSparse */
template <class ST> struct TSparse
{
	public:
		/** vector index */
		INT vec_index;
		/** number of feature entries */
		INT num_feat_entries;
		/** features */
		TSparseEntry<ST>* features;
};

/** template class SparseFeatures */
template <class ST> class CSparseFeatures: public CFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CSparseFeatures(INT size=0) : CFeatures(size), num_vectors(0), num_features(0), sparse_feature_matrix(NULL), feature_cache(NULL) {}

		/** copy constructor */
		CSparseFeatures(const CSparseFeatures & orig) :
			CFeatures(orig), num_vectors(orig.num_vectors), num_features(orig.num_features), sparse_feature_matrix(orig.sparse_feature_matrix), feature_cache(orig.feature_cache)
	{
		if (orig.sparse_feature_matrix)
		{
			free_sparse_feature_matrix();
			sparse_feature_matrix=new TSparse<ST>[num_vectors];
			memcpy(sparse_feature_matrix, orig.sparse_feature_matrix, sizeof(TSparse<ST>)*num_vectors);
			for (INT i=0; i< num_vectors; i++)
			{
				sparse_feature_matrix[i].features=new TSparseEntry<ST>[sparse_feature_matrix[i].num_feat_entries];
				memcpy(sparse_feature_matrix[i].features, orig.sparse_feature_matrix[i].features, sizeof(TSparseEntry<ST>)*sparse_feature_matrix[i].num_feat_entries);

			}
		}
	}

		/** constructor
		 *
		 * @param fname filename to load features from
		 */
		CSparseFeatures(CHAR* fname) :
			CFeatures(fname), num_vectors(0), num_features(0), sparse_feature_matrix(NULL), feature_cache(NULL) {}

		virtual ~CSparseFeatures()
		{
			free_sparse_features();
		}

		/** free sparse feature matrix
		 *
		 */
		void free_sparse_feature_matrix()
        {
            clean_tsparse(sparse_feature_matrix, num_vectors);
            sparse_feature_matrix = NULL;
            num_vectors=0;
            num_features=0;
        }

		/** free sparse feature matrix and cache
		 *
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

		/** converts a sparse feature vector into a dense one
		  * preprocessed compute_feature_vector
		  * caller cleans up
		  *
		  * @param num index of feature vector
		  * @param len length is returned by reference
		  * @return dense feature vector
		  */
		ST* get_full_feature_vector(INT num, INT& len)
		{
			bool vfree;
			INT num_feat;
			INT i;
			len=0;
			TSparseEntry<ST>* sv=get_sparse_feature_vector(num, num_feat, vfree);
			ST* fv=NULL;

			if (sv)
			{
				len=num_features;
				fv=new ST[num_features];

				for (i=0; i<num_features; i++)
					fv[i]=0;

				for (i=0; i<num_feat; i++)
					fv[sv[i].feat_index]= sv[i].entry;
			}

			free_sparse_feature_vector(sv, num, vfree);

			return fv;
		}


		/** get number of sparse features in vector
		 *
		 * @param num which vector
		 * @return number of sparse features in vector
		 */
		inline INT get_num_sparse_vec_features(INT num)
		{
			bool vfree;
			INT len;
			TSparseEntry<ST>* sv = get_sparse_feature_vector(num, len, vfree);
			free_sparse_feature_vector(sv, num, vfree);
			return len;
		}

		/** get sparse feature vector
		 * for sample num from the matrix as it is if matrix is initialized,
		 * else return preprocessed compute_feature_vector
		 *
		 * @param num index of feature vector
		 * @param len number of sparse entries is returned by reference
		 * @param vfree whether returned vector must be freed by caller via
		 *              free_sparse_feature_vector
		 * @return sparse feature vector
		 */
		TSparseEntry<ST>* get_sparse_feature_vector(INT num, INT& len, bool& vfree)
		{
			ASSERT(num<num_vectors);

			if (sparse_feature_matrix)
			{
				len= sparse_feature_matrix[num].num_feat_entries;
				vfree=false ;
				return sparse_feature_matrix[num].features;
			} 
			else
			{
				TSparseEntry<ST>* feat=NULL;
				vfree=false;

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
					vfree=true;

				feat=compute_sparse_feature_vector(num, len, feat);


				if (get_num_preproc())
				{
					INT tmp_len=len;
					TSparseEntry<ST>* tmp_feat_before = feat;
					TSparseEntry<ST>* tmp_feat_after = NULL;

					for (INT i=0; i<get_num_preproc(); i++)
					{
						//tmp_feat_after=((CSparsePreProc<ST>*) get_preproc(i))->apply_to_feature_vector(tmp_feat_before, tmp_len);

						if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
							delete[] tmp_feat_before;
						tmp_feat_before=tmp_feat_after;
					}

					memcpy(feat, tmp_feat_after, sizeof(TSparseEntry<ST>)*tmp_len);
					delete[] tmp_feat_after;
					len=tmp_len ;
					SG_DEBUG( "len: %d len2: %d\n", len, num_features);
				}
				return feat ;
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
		ST sparse_dot(ST alpha, TSparseEntry<ST>* avec, INT alen, TSparseEntry<ST>* bvec, INT blen)
		{
			ST result=0;

			//result remains zero when one of the vectors is non existent
			if (avec && bvec)
			{
				if (alen<=blen)
				{
					INT j=0;
					for (INT i=0; i<alen; i++)
					{
						INT a_feat_idx=avec[i].feat_index;

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
					INT j=0;
					for (INT i=0; i<blen; i++)
					{
						INT b_feat_idx=bvec[i].feat_index;

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

		/** compute the dot product for a range of vectors
		 * alphas[i] * sparse[i]^T * w + b
		 *
		 * @param output result for the given vector range
		 * @param start start vector range from this idx
		 * @param stop stop vector range at this idx
		 * @param alphas scalars to multiply with
		 * @param vec dense vector to compute dot product with
		 * @param dim length of the dense vector
		 * @param b bias
		 */
		void dense_dot_range(ST* output, INT start, INT stop, ST* alphas, ST* vec, INT dim, ST b)
		{
			ASSERT(output);
			ASSERT(start>=0);
			ASSERT(stop<=num_vectors);

			for (INT i=start; i<stop; i++)
				output[i]=dense_dot(alphas[i], i, vec, dim, b);
		}

		/** compute the dot product between dense weights and a sparse feature vector
		 * alpha * sparse^T * w + b
		 *
		 * @param alpha scalar to multiply with
		 * @param num index of feature vector
		 * @param vec dense vector to compute dot product with
		 * @param dim length of the dense vector
		 * @param b bias
		 * @return dot product between dense weights and a sparse feature vector
		 */
		ST dense_dot(ST alpha, INT num, ST* vec, INT dim, ST b)
		{
			ASSERT(vec);
			ASSERT(dim==num_features);
			ST result=b;

			bool vfree;
			INT num_feat;
			TSparseEntry<ST>* sv=get_sparse_feature_vector(num, num_feat, vfree);

			if (sv)
			{
				for (INT i=0; i<num_feat; i++)
					result+=alpha*vec[sv[i].feat_index]*sv[i].entry;
			}

			free_sparse_feature_vector(sv, num, vfree);
			return result;
		}

		/** add a sparse feature vector onto a dense one
		 * dense+=alpha*sparse
		 *
		 @param alpha scalar to multiply with
		 @param num index of feature vector
		 @param vec dense vector
		 @param dim length of the dense vector
		 @param abs_val if true, do dense+=alpha*abs(sparse)
		 */
		void add_to_dense_vec(ST alpha, INT num, ST* vec, INT dim, bool abs_val=false)
		{
			ASSERT(vec);
			ASSERT(dim==num_features);

			bool vfree;
			INT num_feat;
			TSparseEntry<ST>* sv=get_sparse_feature_vector(num, num_feat, vfree);

			if (sv)
			{
				if (abs_val)
				{
					for (INT i=0; i<num_feat; i++)
						vec[sv[i].feat_index]+= alpha*CMath::abs(sv[i].entry);
				}
				else
				{
					for (INT i=0; i<num_feat; i++)
						vec[sv[i].feat_index]+= alpha*sv[i].entry;
				}
			}

			free_sparse_feature_vector(sv, num, vfree);
		}

		/** free sparse feature vector
		 *
		 * @param feat_vec feature vector to free
		 * @param num index of this vector in the cache
		 * @param free if vector should be really deleted
		 */
		void free_sparse_feature_vector(TSparseEntry<ST>* feat_vec, INT num, bool free)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (free)
				delete[] feat_vec ;
		} 

		/** get the pointer to the sparse feature matrix
		 * num_feat,num_vectors are returned by reference
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return feature matrix
		 */
		TSparse<ST>* get_sparse_feature_matrix(INT &num_feat, INT &num_vec)
		{
			num_feat=num_features;
			num_vec=num_vectors;

			return sparse_feature_matrix;
		}

		/** clean TSparse
		 *
		 * @param sfm sparse feature matrix
		 * @param num_vec number of vectors in matrix
		 */
		void clean_tsparse(TSparse<ST>* sfm, INT num_vec)
		{
			if (sfm)
			{
				for (INT i=0; i<num_vec; i++)
					delete[] sfm[i].features;

				delete[] sfm;
			}
		}

		/** compute and return the transpose of the sparse feature matrix
		 * which will be prepocessed.
		 * num_feat, num_vectors are returned by reference
		 * caller has to clean up
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return transposed sparse feature matrix
		 */
		TSparse<ST>* get_transposed(INT &num_feat, INT &num_vec)
		{
			num_feat=num_vectors;
			num_vec=num_features;

			INT* hist=new INT[num_features];
			memset(hist, 0, sizeof(INT)*num_features);

			// count how lengths of future feature vectors
			for (INT v=0; v<num_vectors; v++)
			{
				INT vlen;
				bool vfree;
				TSparseEntry<ST>* sv=get_sparse_feature_vector(v, vlen, vfree);

				for (INT i=0; i<vlen; i++)
					hist[sv[i].feat_index]++;

				free_sparse_feature_vector(sv, v, vfree);
			}

			// allocate room for future feature vectors
			TSparse<ST>* sfm=new TSparse<ST>[num_vec];
			for (INT v=0; v<num_vec; v++)
			{
				sfm[v].features= new TSparseEntry<ST>[hist[v]];
				sfm[v].num_feat_entries=hist[v];
				sfm[v].vec_index=v;
			}

			// fill future feature vectors with content
			memset(hist,0,sizeof(INT)*num_features);
			for (INT v=0; v<num_vectors; v++)
			{
				INT vlen;
				bool vfree;
				TSparseEntry<ST>* sv=get_sparse_feature_vector(v, vlen, vfree);

				for (INT i=0; i<vlen; i++)
				{
					INT vidx=sv[i].feat_index;
					INT fidx=v;
					sfm[vidx].features[hist[vidx]].feat_index=fidx;
					sfm[vidx].features[hist[vidx]].entry=sv[i].entry;
					hist[vidx]++;
				}

				free_sparse_feature_vector(sv, v, vfree);
			}

			delete[] hist;
			return sfm;
		}

		/** set feature matrix
		 * necessary to set feature_matrix, num_features, num_vectors, where
		 * num_features is the column offset, and columns are linear in memory
		 * see below for definition of feature_matrix
		 *
		 * @param sfm new sparse feature matrix
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 */
		virtual void set_sparse_feature_matrix(TSparse<ST>* sfm, INT num_feat, INT num_vec)
		{
			free_sparse_feature_matrix();

			sparse_feature_matrix=sfm;
			num_features=num_feat;
			num_vectors=num_vec;
		}

		/** gets a copy of a full  feature matrix
		 * num_feat,num_vectors are returned by reference
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return full feature matrix
		 */
		ST* get_full_feature_matrix(INT &num_feat, INT &num_vec)
		{
			SG_INFO( "converting sparse features to full feature matrix of %ld x %ld entries\n", num_vectors, num_features);
			num_feat=num_features;
			num_vec=num_vectors;

			ST* fm=new ST[num_feat*num_vec];

			if (fm)
			{
				for (LONG i=0; i<num_feat*num_vec; i++)
					fm[i]=0;

				for (INT v=0; v<num_vec; v++)
				{
					for (INT f=0; f<sparse_feature_matrix[v].num_feat_entries; f++)
					{
						LONG offs= (sparse_feature_matrix[v].vec_index * num_feat) + sparse_feature_matrix[v].features[f].feat_index;
						fm[offs]= sparse_feature_matrix[v].features[f].entry;
					}
				}
			}
			else
				SG_ERROR( "error allocating memory for dense feature matrix\n");

			return fm;
		}

		/** creates a sparse feature matrix from a full dense feature matrix
		 * necessary to set feature_matrix, num_features and num_vectors
		 * where num_features is the column offset, and columns are linear in memory
		 * see above for definition of sparse_feature_matrix
		 *
		 * @param ffm full feature matrix
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 */
		virtual bool set_full_feature_matrix(ST* ffm, INT num_feat, INT num_vec)
		{
			free_sparse_feature_matrix();
			bool result=true;
			num_features=num_feat;
			num_vectors=num_vec;

			SG_INFO("converting dense feature matrix to sparse one\n");
			INT* num_feat_entries=new int[num_vectors];

			if (num_feat_entries)
			{
				INT num_total_entries=0;

				// count nr of non sparse features
				for (INT i=0; i< num_vec; i++)
				{
					num_feat_entries[i]=0;
					for (INT j=0; j< num_feat; j++)
					{
						if (ffm[i*((LONG) num_feat) + j] != 0)
							num_feat_entries[i]++;
					}
				}

				if (num_vec>0)
				{
					sparse_feature_matrix=new TSparse<ST>[num_vec];

					if (sparse_feature_matrix)
					{
						for (INT i=0; i< num_vec; i++)
						{
							sparse_feature_matrix[i].vec_index=i;
							sparse_feature_matrix[i].num_feat_entries=0;
							sparse_feature_matrix[i].features= NULL;

							if (num_feat_entries[i]>0)
							{
								sparse_feature_matrix[i].features= new TSparseEntry<ST>[num_feat_entries[i]];

								if (!sparse_feature_matrix[i].features)
								{
									SG_INFO( "allocation of features failed\n");
									return false;
								}

								sparse_feature_matrix[i].num_feat_entries=num_feat_entries[i];
								INT sparse_feat_idx=0;

								for (INT j=0; j< num_feat; j++)
								{
									LONG pos= i*num_feat + j;

									if (ffm[pos] != 0)
									{
										sparse_feature_matrix[i].features[sparse_feat_idx].entry=ffm[pos];
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
							num_total_entries, num_feat*num_vec, (100.0*num_total_entries)/(num_feat*num_vec));
				}
				else
				{
					SG_ERROR( "huh ? zero size matrix given ?\n");
					result=false;
				}
			}
			delete[] num_feat_entries;
			return result;
		}

		/** apply preprocessor
		 *
		 * @param force_preprocessing if preprocssing shall be forced
		 * @return if applying was successful
		 */
		virtual bool apply_preproc(bool force_preprocessing=false)
		{
			SG_INFO( "force: %d\n", force_preprocessing);

			if ( sparse_feature_matrix && get_num_preproc() )
			{
				for (INT i=0; i<get_num_preproc(); i++)
				{
					if ( (!is_preprocessed(i) || force_preprocessing) )
					{
						set_preprocessed(i);
						SG_INFO( "preprocessing using preproc %s\n", get_preproc(i)->get_name());
						if (((CSparsePreProc<ST>*) get_preproc(i))->apply_to_sparse_feature_matrix(this) == NULL)
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
		virtual INT get_size() { return sizeof(ST); }

		/** obtain sparse features from simple features
		 *
		 * @param sf simple features
		 * @return if obtaining was successful
		 */
		bool obtain_from_simple(CSimpleFeatures<ST>* sf)
		{
			INT num_feat=0;
			INT num_vec=0;
			ST* fm=sf->get_feature_matrix(num_feat, num_vec);
			ASSERT(fm && num_feat>0 && num_vec>0);

			return set_full_feature_matrix(fm, num_feat, num_vec);
		}

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
		 * Sometimes when loading sparse features not all possible dimensions
		 * are used. This may pose a problem to classifiers when being applied
		 * to higher dimensional test-data. This function allows to
		 * artificially explode the feature space
		 *
		 * @param num the number of features, must be larger
		 *        than the current number of features
		 * @return previous number of features
		 */
		inline INT set_num_features(INT num)
		{
			INT n=num_features;
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
		 * @param feat_vec feature vector to free
		 * @param num index of vector in cache
		 * @param free if vector really should be deleted
		 */
		void free_feature_vector(TSparseEntry<ST>* feat_vec, INT num, bool free)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (free)
				delete[] feat_vec ;
		}

		/** get number of non-zero entries in sparse feature matrix
		 *
		 * @return number of non-zero entries in sparse feature matrix
		 */
		LONG get_num_nonzero_entries()
		{
			LONG num=0;
			for (INT i=0; i<num_vectors; i++)
				num+=sparse_feature_matrix[i].num_feat_entries;

			return num;
		}

		/** compute a^2 on all feature vectors
		 *
		 * @param sq the square for each vector is stored in here
		 * @return the square for each vector
		 */
		DREAL* compute_squared(DREAL* sq)
		{
			ASSERT(sq);

			INT len=0;
			bool do_free=false;

			for (INT i=0; i<this->get_num_vectors(); i++)
			{
				sq[i]=0;
				TSparseEntry<DREAL>* vec = ((CSparseFeatures<DREAL>*) this)->get_sparse_feature_vector(i, len, do_free);

				for (INT j=0; j<len; j++)
					sq[i] += vec[j].entry * vec[j].entry;

				((CSparseFeatures<DREAL>*) this)->free_feature_vector(vec, i, do_free);
			}

			return sq;
		}

		/** compute (a-b)^2 (== a^2+b^2+2ab)
		 * usually called by kernels'/distances' compute functions
		 * works on two feature vectors, although it is a member of a single
		 * feature: can either be called by lhs or rhs.
		 *
		 * @param lhs left-hand side features
		 * @param sq_lhs squared values of left-hand side
		 * @param idx_a index of left-hand side's vector to compute
		 * @param rhs right-hand side features
		 * @param sq_rhs squared values of right-hand side
		 * @param idx_b index of right-hand side's vector to compute
		 */
		DREAL compute_squared_norm(CSparseFeatures<DREAL>* lhs, DREAL* sq_lhs, INT idx_a, CSparseFeatures<DREAL>* rhs, DREAL* sq_rhs, INT idx_b)
		{
			INT i,j;
			INT alen, blen;
			bool afree, bfree;
			ASSERT(lhs);
			ASSERT(rhs);

			TSparseEntry<DREAL>* avec=lhs->get_sparse_feature_vector(idx_a, alen, afree);
			TSparseEntry<DREAL>* bvec=rhs->get_sparse_feature_vector(idx_b, blen, bfree);
			ASSERT(avec);
			ASSERT(bvec);

			DREAL result=sq_lhs[idx_a]+sq_rhs[idx_b];

			if (alen<=blen)
			{
				j=0;
				for (i=0; i<alen; i++)
				{
					INT a_feat_idx=avec[i].feat_index;

					while ((j<blen) && (bvec[j].feat_index < a_feat_idx))
						j++;

					if ((j<blen) && (bvec[j].feat_index == a_feat_idx))
					{
						result-=2*(avec[i].entry*bvec[j].entry);
						j++;
					}
				}
			}
			else
			{
				j=0;
				for (i=0; i<blen; i++)
				{
					INT b_feat_idx=bvec[i].feat_index;

					while ((j<alen) && (avec[j].feat_index<b_feat_idx))
						j++;

					if ((j<alen) && (avec[j].feat_index == b_feat_idx))
					{
						result-=2*(bvec[i].entry*avec[j].entry);
						j++;
					}
				}
			}

			((CSparseFeatures<DREAL>*) lhs)->free_feature_vector(avec, idx_a, afree);
			((CSparseFeatures<DREAL>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

			return CMath::abs(result);
		}

		/** load features from file
		 *
		 * @param fname filename to load from
		 * @return label object with corresponding labels
		 */
		CLabels* load_svmlight_file(CHAR* fname)
		{
			CLabels* lab=NULL;

			size_t blocksize=1024*1024;
			size_t required_blocksize=blocksize;
			BYTE* dummy=new BYTE[blocksize];
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
					sz=fread(dummy, sizeof(BYTE), blocksize, f);
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
				delete[] dummy;
				blocksize=required_blocksize;
				dummy = new BYTE[blocksize+1]; //allow setting of '\0' at EOL

				lab=new CLabels(num_vectors);
				sparse_feature_matrix=new TSparse<ST>[num_vectors];

				rewind(f);
				sz=blocksize;
				INT lines=0;
				while (sz == blocksize)
				{
					sz=fread(dummy, sizeof(BYTE), blocksize, f);

					size_t old_sz=0;
					for (size_t i=0; i<sz; i++)
					{
						if (i==sz-1 && dummy[i]!='\n' && sz==blocksize)
						{
							size_t len=i-old_sz+1;
							BYTE* data=&dummy[old_sz];

							for (INT j=0; j<len; j++)
								dummy[j]=data[j];

							sz=fread(dummy+len, sizeof(BYTE), blocksize-len, f);
							i=0;
							old_sz=0;
							sz+=len;
						}

						if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
						{

							size_t len=i-old_sz;
							BYTE* data=&dummy[old_sz];

							INT dims=0;
							for (INT j=0; j<len; j++)
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

							TSparseEntry<ST>* feat=new TSparseEntry<ST>[dims];
							INT j=0;
							for (; j<len; j++)
							{
								if (data[j]==' ')
								{
									data[j]='\0';

									lab->set_label(lines, atof((const char*) data));
									break;
								}
							}

							INT d=0;
							j++;
							BYTE* start=&data[j];
							for (; j<len; j++)
							{
								if (data[j]==':')
								{
									data[j]='\0';

									feat[d].feat_index=(INT) atoi((const char*) start)-1;
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

			delete[] dummy;

			return lab;
		}

		/** write features to file using svm light format
		 *
		 * @param fname filename to write to
		 * @param label Label object (number of labels must correspond to number of features)
		 * @return true if successful
		 */
		bool write_svmlight_file(CHAR* fname, CLabels* label)
		{
			ASSERT(label);
			INT num=label->get_num_labels();
			ASSERT(num>0);
			ASSERT(num==num_vectors);

			FILE* f=fopen(fname, "wb");

			if (f)
			{
				for (INT i=0; i<num; i++)
				{
					fprintf(f, "%d ", (INT) label->get_int_label(i));

					TSparseEntry<ST>* vec = sparse_feature_matrix[i].features;
					INT num_feat = sparse_feature_matrix[i].num_feat_entries;

					for (INT j=0; j<num_feat; j++)
					{
						if (j<num_feat-1)
							fprintf(f, "%d:%f ", (INT) vec[j].feat_index+1, (double) vec[j].entry);
						else
							fprintf(f, "%d:%f\n", (INT) vec[j].feat_index+1, (double) vec[j].entry);
					}
				}

				fclose(f);
				return true;
			}
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
		 * @param target target
		 */
		virtual TSparseEntry<ST>* compute_sparse_feature_vector(INT num, INT& len, TSparseEntry<ST>* target=NULL)
		{
			len=0;
			return NULL;
		}

	protected:

		/// total number of vectors
		INT num_vectors;

		/// total number of features
		INT num_features;

		/// array of sparse vectors of size num_vectors
		TSparse<ST>* sparse_feature_matrix;

		/** feature cache */
		CCache< TSparseEntry<ST> >* feature_cache;
};


/** get feature type the CHAR feature can deal with
 *
 * @return feature type CHAR
 */
template<> inline EFeatureType CSparseFeatures<CHAR>::get_feature_type()
{
	return F_CHAR;
}

/** get feature type the BYTE feature can deal with
 *
 * @return feature type BYTE
 */
template<> inline EFeatureType CSparseFeatures<BYTE>::get_feature_type()
{
	return F_BYTE;
}

/** get feature type the SHORT feature can deal with
 *
 * @return feature type SHORT
 */
template<> inline EFeatureType CSparseFeatures<SHORT>::get_feature_type()
{
	return F_SHORT;
}

/** get feature type the WORD feature can deal with
 *
 * @return feature type WORD
 */
template<> inline EFeatureType CSparseFeatures<WORD>::get_feature_type()
{
	return F_WORD;
}

/** get feature type the INT feature can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType CSparseFeatures<INT>::get_feature_type()
{
	return F_INT;
}

/** get feature type the UINT feature can deal with
 *
 * @return feature type UINT
 */
template<> inline EFeatureType CSparseFeatures<UINT>::get_feature_type()
{
	return F_UINT;
}

/** get feature type the LONG feature can deal with
 *
 * @return feature type LONG
 */
template<> inline EFeatureType CSparseFeatures<LONG>::get_feature_type()
{
	return F_LONG;
}

/** get feature type the ULONG feature can deal with
 *
 * @return feature type ULONG
 */
template<> inline EFeatureType CSparseFeatures<ULONG>::get_feature_type()
{
	return F_ULONG;
}

/** get feature type the DREAL feature can deal with
 *
 * @return feature type DREAL
 */
template<> inline EFeatureType CSparseFeatures<DREAL>::get_feature_type()
{
	return F_DREAL;
}

/** get feature type the SHORTREAL feature can deal with
 *
 * @return feature type SHORTREAL
 */
template<> inline EFeatureType CSparseFeatures<SHORTREAL>::get_feature_type()
{
	return F_SHORTREAL;
}

/** get feature type the LONGREAL feature can deal with
 *
 * @return feature type LONGREAL
 */
template<> inline EFeatureType CSparseFeatures<LONGREAL>::get_feature_type()
{
	return F_LONGREAL;
}
#endif /* _SPARSEFEATURES__H__ */
