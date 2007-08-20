/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEFEATURES__H__
#define _SPARSEFEATURES__H__

#include <string.h>

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/Cache.h"
#include "lib/io.h"
#include "lib/Cache.h"

#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "features/RealFeatures.h"
#include "preproc/SparsePreProc.h"

//features are an array of TSparse, sorted w.r.t. vec_index (increasing) and
//withing same vec_index w.r.t. feat_index (increasing);

template <class ST> class CSparsePreProc;

template <class ST> struct TSparseEntry
{
	INT feat_index;
	ST entry;
};

template <class ST> struct TSparse
{
public:
	INT vec_index;
	INT num_feat_entries;
	TSparseEntry<ST>* features;
};

template <class ST> class CSparseFeatures: public CFeatures
{
	public:
		CSparseFeatures(INT size=0) : CFeatures(size), num_vectors(0), num_features(0), sparse_feature_matrix(NULL), feature_cache(NULL)
		{
		}

		CSparseFeatures(const CSparseFeatures & orig) : 
			CFeatures(orig), num_vectors(orig.num_vectors), num_features(orig.num_features), sparse_feature_matrix(orig.sparse_feature_matrix), feature_cache(orig.feature_cache)
		{
			if (orig.sparse_feature_matrix)
			{
				sparse_feature_matrix=new TSparse<ST>[num_vectors];
				memcpy(sparse_feature_matrix, orig.sparse_feature_matrix, sizeof(TSparse<ST>)*num_vectors); 
				for (INT i=0; i< num_vectors; i++)
				{
					sparse_feature_matrix[i].features=new TSparseEntry<ST>[sparse_feature_matrix[i].num_feat_entries];
					memcpy(sparse_feature_matrix[i].features, orig.sparse_feature_matrix[i].features, sizeof(TSparseEntry<ST>)*sparse_feature_matrix[i].num_feat_entries); 

				}
			}
		}

		CSparseFeatures(CHAR* fname) : CFeatures(fname), num_vectors(0), num_features(0), sparse_feature_matrix(NULL), feature_cache(NULL)
		{
		}

		virtual ~CSparseFeatures()
		{
			clean_tsparse(sparse_feature_matrix, num_vectors);

			delete feature_cache;
		}

		virtual CFeatures* duplicate() const
		{
			return new CSparseFeatures<ST>(*this);
		}

		/** converts a sparse feature vector into a dense one
		  preprocessed compute_feature_vector  
		  caller cleans up
		  @param num index of feature vector
		  @param len length is returned by reference
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
				ASSERT(fv);

				for (i=0; i<num_features; i++)
					fv[i]=0;

				for (i=0; i<num_feat; i++)
					fv[sv[i].feat_index]= sv[i].entry;
			}

			free_sparse_feature_vector(sv, num, vfree);

			return fv;
		}

		/** get feature vector for sample num
		  from the matrix as it is if matrix is
		  initialized, else return
		  preprocessed compute_feature_vector  
		  @param num index of feature vector
		  @param len number of sparse entries is returned by reference
		  */
		TSparseEntry<ST>* get_sparse_feature_vector(INT num, INT& len, bool& vfree)
		{
			ASSERT(num<num_vectors);
			len= sparse_feature_matrix[num].num_feat_entries;

			if (sparse_feature_matrix)
			{
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
		 *	alpha * vec^T * vec
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

		/** compute the dot product between dense weights and a sparse feature vector
		 *	alpha * sparse^T * w + b
		 *
		 @param alpha scalar to multiply with
		 @param num index of feature vector
		 @param vec dense vector to compute dot product with
		 @param dim length of the dense vector
		 @param b bias
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
		 *	dense+=alpha*sparse
		 *
		 @param alpha scalar to multiply with
		 @param num index of feature vector
		 @param vec dense vector
		 @param dim length of the dense vector
		 */
		void add_to_dense_vec(ST alpha, INT num, ST* vec, INT dim)
		{
			ASSERT(vec);
			ASSERT(dim==num_features);

			bool vfree;
			INT num_feat;
			TSparseEntry<ST>* sv=get_sparse_feature_vector(num, num_feat, vfree);

			if (sv)
			{
				for (INT i=0; i<num_feat; i++)
					vec[sv[i].feat_index]+= alpha*sv[i].entry;
			}

			free_sparse_feature_vector(sv, num, vfree);
		}

		void free_sparse_feature_vector(TSparseEntry<ST>* feat_vec, INT num, bool free)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (free)
				delete[] feat_vec ;
		} 

		/// get the pointer to the sparse feature matrix
		/// num_feat,num_vectors are returned by reference
		TSparse<ST>* get_sparse_feature_matrix(INT &num_feat, INT &num_vec)
		{
			num_feat=num_features;
			num_vec=num_vectors;

			return sparse_feature_matrix;
		}

		void clean_tsparse(TSparse<ST>* sfm, INT num_vec)
		{
			if (sfm)
			{
				for (INT i=0; i<num_vec; i++)
					delete[] sfm[i].features;

				delete[] sfm;
			}
		}

		/// compute and return the transpose of the sparse feature matrix
		/// which will be prepocessed. 
		/// num_feat, num_vectors are returned by reference
		/// caller has to clean up
		TSparse<ST>* get_transposed(INT &num_feat, INT &num_vec)
		{
			num_feat=num_vectors;
			num_vec=num_features;

			INT* hist=new INT[num_features];
			ASSERT(hist);
			memset(hist,0,sizeof(INT)*num_features);

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
			TSparse<ST>* sfm = new TSparse<ST>[num_vec];
			ASSERT(sfm);

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
		  necessary to set feature_matrix, num_features, num_vectors, where
		  num_features is the column offset, and columns are linear in memory
		  see below for definition of feature_matrix
		  */
		virtual void set_sparse_feature_matrix(TSparse<ST>* sfm, INT num_feat, INT num_vec)
		{
			sparse_feature_matrix=sfm;
			num_features=num_feat;
			num_vectors=num_vec;
		}

		/// gets a copy of a full  feature matrix
		/// num_feat,num_vectors are returned by reference
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
		  necessary to set feature_matrix, num_features and num_vectors
		  where num_features is the column offset, and columns are linear in memory
		  see above for definition of sparse_feature_matrix
		  */
		virtual bool set_full_feature_matrix(ST* ffm, INT num_feat, INT num_vec)
		{
			bool result=true;
			num_features=num_feat;
			num_vectors=num_vec;

			SG_INFO( "converting dense feature matrix to sparse one\n");
			num_feat=num_features;
			num_vec=num_vectors;

			INT* num_feat_entries=new int[num_vectors];
			ASSERT(num_feat_entries);

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

		virtual INT get_size() { return sizeof(ST); }

		bool obtain_from_simple(CSimpleFeatures<ST>* sf)
		{
			INT num_feat=0;
			INT num_vec=0;
			ST* fm=sf->get_feature_matrix(num_feat, num_vec);
			ASSERT(fm && num_feat>0 && num_vec>0);

			return set_full_feature_matrix(fm, num_feat, num_vec);
		}

		virtual inline INT  get_num_vectors() { return num_vectors; }
		inline INT  get_num_features() { return num_features; }

		/// return that we are sparse features
		inline virtual EFeatureClass get_feature_class() { return C_SPARSE; }

		/// return feature type
		inline virtual EFeatureType get_feature_type();

		void free_feature_vector(TSparseEntry<ST>* feat_vec, INT num, bool free)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (free)
				delete[] feat_vec ;
		} 

		long get_num_nonzero_entries()
		{
			long num=0;
			for (int i=0; i<num_vectors; i++)
				num+=sparse_feature_matrix[i].num_feat_entries;

			return num;
		}

	protected:
		/// compute feature vector for sample num
		/// if target is set the vector is written to target
		/// len is returned by reference
		virtual TSparseEntry<ST>* compute_sparse_feature_vector(INT num, INT& len, TSparseEntry<ST>* target=NULL)
		{
			len=0;
			return NULL;
		}

		/// total number of vectors
		INT num_vectors;

		/// total number of features
		INT num_features;

		/// array of sparse vectors of size num_vectors
		TSparse<ST>* sparse_feature_matrix;

		CCache< TSparseEntry<ST> >* feature_cache;
};


template<> inline EFeatureType CSparseFeatures<CHAR>::get_feature_type()
{
	return F_CHAR;
}

template<> inline EFeatureType CSparseFeatures<BYTE>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CSparseFeatures<SHORT>::get_feature_type()
{
	return F_SHORT;
}

template<> inline EFeatureType CSparseFeatures<WORD>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CSparseFeatures<INT>::get_feature_type()
{
	return F_INT;
}

template<> inline EFeatureType CSparseFeatures<UINT>::get_feature_type()
{
	return F_UINT;
}

template<> inline EFeatureType CSparseFeatures<LONG>::get_feature_type()
{
	return F_LONG;
}

template<> inline EFeatureType CSparseFeatures<ULONG>::get_feature_type()
{
	return F_ULONG;
}

template<> inline EFeatureType CSparseFeatures<DREAL>::get_feature_type()
{
	return F_DREAL;
}

template<> inline EFeatureType CSparseFeatures<SHORTREAL>::get_feature_type()
{
	return F_SREAL;
}

template<> inline EFeatureType CSparseFeatures<LONGREAL>::get_feature_type()
{
	return F_LREAL;
}
#endif
