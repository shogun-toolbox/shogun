#ifndef _SPARSEFEATURES__H__
#define _SPARSEFEATURES__H__

#include "lib/common.h"
#include "lib/Mathmatics.h"
#include "lib/Cache.h"
#include "preproc/SparsePreProc.h"
#include "lib/io.h"
#include "lib/Cache.h"

#include <string.h>
#include <assert.h>

#include "features/Features.h"

//features are an array of TSparse, sorted w.r.t. vec_index (increasing) and
//withing same vec_index w.r.t. feat_index (increasing);

template <class ST> class CSparsePreproc;

template <class ST> struct TSparseEntry
{
	int feat_index;
	ST entry;
};

template <class ST> struct TSparse
{
public:
	int vec_index;
	int num_feat_entries;
	TSparseEntry<ST>* features;
};

template <class ST> class CSparseFeatures: public CFeatures
{
	public:
		CSparseFeatures(long size) : CFeatures(size), num_vectors(0), num_features(0), num_sparse_vectors(0), sparse_feature_matrix(NULL), feature_cache(NULL)
		{
		}

		CSparseFeatures(const CSparseFeatures & orig) : 
			CFeatures(orig), num_vectors(orig.num_vectors), num_features(orig.num_features), num_sparse_vectors(orig.num_sparse_vectors), sparse_feature_matrix(orig.sparse_feature_matrix), feature_cache(orig.feature_cache)
			{
				if (orig.sparse_feature_matrix)
				{
					sparse_feature_matrix=new TSparse<REAL>[num_vectors*num_features];
					memcpy(sparse_feature_matrix, orig.sparse_feature_matrix, sizeof(double)*num_vectors*num_features); 
				}
			}

		CSparseFeatures(char* fname) : CFeatures(fname), num_vectors(0), num_features(0), num_sparse_vectors(0), sparse_feature_matrix(NULL), feature_cache(NULL)
		{
		}

		virtual ~CSparseFeatures()
		{
			for (long i=0; i< num_sparse_vectors; i++)
				delete[] sparse_feature_matrix[i].features;

			delete[] sparse_feature_matrix;
			delete feature_cache;
		}

		/** converts a sparse feature vector into a dense one
		  preprocessed compute_feature_vector  
		  caller cleans up
		  @param num index of feature vector
		  @param len length is returned by reference
		  */
		ST* get_full_feature_vector(long num, long& len)
		{
			bool free;
			long num_feat;
			long i;
			len=0;
			TSparseEntry<ST>* sv=get_sparse_feature_vector(num, num_feat, free);

			if (sv)
			{
				len=num_features;
				ST* fv=new ST[num_features];

				for (i=0; i<num_features; i++)
					fv[i]=0;

				for (i=0; i<num_feat; i++)
					fv[sv[i].feat_index]= sv[feat].entry;
			}

			free_sparse_feature_vector(sv, num, free);

			return sv;
		}

		/** get feature vector for sample num
		  from the matrix as it is if matrix is
		  initialized, else return
		  preprocessed compute_feature_vector  
		  @param num index of feature vector
		  @param len number of sparse entries is returned by reference
		  */
		TSparseEntry<ST>* get_sparse_feature_vector(long num, long& len, bool& free)
		{
			assert(num<num_vectors);
			len= sparse_feature_matrix[num].num_feat_entries;

			if (sparse_feature_matrix)
			{
				free=false ;
				return sparse_feature_matrix[num].features;
			} 
			else
			{
				TSparseEntry<ST>* feat=NULL;
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

				feat=compute_sparse_feature_vector(num, len, feat);


				if (get_num_preproc())
				{
					int tmp_len=len;
					TSparseEntry<ST>* tmp_feat_before = feat;
					TSparseEntry<ST>* tmp_feat_after = NULL;

					for (int i=0; i<get_num_preproc(); i++)
					{
						//tmp_feat_after=((CSparsePreProc<ST>*) get_preproc(i))->apply_to_feature_vector(tmp_feat_before, tmp_len);

						if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
							delete[] tmp_feat_before;
						tmp_feat_before=tmp_feat_after;
					}

					memcpy(feat, tmp_feat_after, sizeof(TSparseEntry<ST>)*tmp_len);
					delete[] tmp_feat_after;
					len=tmp_len ;
					CIO::message(stderr, "len: %d len2: %d\n", len, num_features);
				}
				return feat ;
			}
		}

		void free_sparse_feature_vector(TSparseEntry<ST>* feat_vec, int num, bool free)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (free)
				delete[] feat_vec ;
		} 

		/// get the pointer to the sparse feature matrix
		/// num_feat,num_vectors are returned by reference
		/// the actual number of sparse vectors is returned in num_sparse_vec
		TSparse<ST>* get_sparse_feature_matrix(long &num_feat, long &num_vec, long &num_sparse_vec)
		{
			num_feat=num_features;
			num_vec=num_vectors;

			return feature_matrix;
		}

		/** set feature matrix
		  necessary to set feature_matrix, num_features, num_vectors, where
		  num_features is the column offset, and columns are linear in memory
		  see below for definition of feature_matrix
		  */
		virtual void set_sparse_feature_matrix(TSparse<ST>* sfm, long num_feat, long num_vec, long num_sparse_vec)
		{
			sparse_feature_matrix=sfm;
			num_features=num_feat;
			num_vectors=num_vec;
			num_sparse_vectors=num_sparse_vec;
		}

		/// gets a copy of a full  feature matrix
		/// num_feat,num_vectors are returned by reference
		ST* get_full_feature_matrix(long &num_feat, long &num_vec)
		{
			CIO::message("converting sparse features to full feature matrix of %ld x %ld entries\n", num_vectors, num_features);
			num_feat=num_features;
			num_vec=num_vectors;

			ST* fm=new ST[num_feat*num_vec];

			if (fm)
			{
				for (long i=0; i<num_feat*num_vec; i++)
					fm[i]=0;

				for (long v=0; v<num_sparse_vectors; v++)
				{
					for (long f=0; f<sparse_feature_matrix[v].num_feat_entries; f++)
					{
						long offs= (sparse_feature_matrix[v].vec_index * num_feat) + sparse_feature_matrix[v].features[f].feat_index;
						fm[offs]= sparse_feature_matrix[v].features[f].entry;
					}
				}
			}
			else
				CIO::message("error allocating memory for dense feature matrix\n");

			return fm;
		}

		/** creates a sparse feature matrix from a full dense feature matrix
		  necessary to set feature_matrix, num_features, num_vectors, num_sparse vec
		  where num_features is the column offset, and columns are linear in memory
		  see above for definition of sparse_feature_matrix
		  */
		virtual bool set_full_feature_matrix(ST* ffm, long num_feat, long num_vec)
		{
			bool result=true;
			num_features=num_feat;
			num_vectors=num_vec;

			CIO::message("converting dense feature matrix to sparse one\n");
			num_feat=num_features;
			num_vec=num_vectors;

			int* num_feat_entries=new int[num_vectors];
			assert(num_feat_entries);

			if (num_feat_entries)
			{
				long num_sparse_vec=0;
				long num_total_entries=0;

				// count nr of non sparse features
				for (long i=0; i< num_vec; i++)
				{
					num_feat_entries[i]=0;
					for (long j=0; j< num_feat; j++)
					{
						if (ffm[i*num_feat + j] != 0)
							num_feat_entries[i]++;
					}

					if (num_feat_entries[i])
						num_sparse_vec++;
				}

				num_sparse_vectors=num_sparse_vec;

				CIO::message("num_sparse_vec: %d\n", num_sparse_vec);

				if (num_sparse_vec>0)
				{
					long sparse_vec_idx=0;
					sparse_feature_matrix=new TSparse<ST>[num_sparse_vectors];

					if (sparse_feature_matrix)
					{
						for (long i=0; i< num_vec; i++)
						{
							if (num_feat_entries[i]>0)
							{
								sparse_feature_matrix[sparse_vec_idx].features= new TSparseEntry<ST>[num_feat_entries[i]];

								if (!sparse_feature_matrix[sparse_vec_idx].features)
								{
									CIO::message("allocation of features failed\n");
									return false;
								}

								sparse_feature_matrix[sparse_vec_idx].num_feat_entries=num_feat_entries[i];
								sparse_feature_matrix[sparse_vec_idx].vec_index=i;

								long sparse_feat_idx=0;

								for (long j=0; j< num_feat; j++)
								{
									long pos= i*num_feat + j;

									if (ffm[pos] != 0)
									{
										sparse_feature_matrix[sparse_vec_idx].features[sparse_feat_idx].entry=ffm[pos];
										sparse_feature_matrix[sparse_vec_idx].features[sparse_feat_idx].feat_index=j;
										sparse_feat_idx++;
										num_total_entries++;
									}

								}
								sparse_vec_idx++;
							}
						}
					}
					else
					{
						CIO::message("allocation of sparse feature matrix failed\n");
						result=false;
					}

					CIO::message("sparse feature matrix has %ld entries (full matrix had %ld)\n", num_total_entries, num_feat*num_vec);
				}
				else
				{
					CIO::message("huh ? zero matrix given ?\n");
					result=false;
				}
			}
			delete[] num_feat_entries;
			return result;
		}

		virtual bool preproc_feature_matrix(bool force_preprocessing=false)
		{
			return preproc_sparse_feature_matrix(force_preprocessing);
		}

		virtual bool preproc_sparse_feature_matrix(bool force_preprocessing=false)
		{
			CIO::message("force: %d\n", force_preprocessing);

			if ( sparse_feature_matrix && get_num_preproc() )
			{
				for (int i=0; i<get_num_preproc(); i++)
				{
					if ( (!is_preprocessed(i) || force_preprocessing) )
					{
						set_preprocessed(i);
						CIO::message("preprocessing using preproc %s\n", get_preproc(i)->get_name());
						if (((CSparsePreProc<ST>*) get_preproc(i))->apply_to_sparse_feature_matrix(this) == NULL)
							return false;
					}
					return true;
				}
				return true;
			}
			else
			{
				CIO::message("no sparse feature matrix available or features already preprocessed - skipping.\n");
				return false;
			}
		}

		virtual int get_size() { return sizeof(ST); }

		virtual inline long  get_num_vectors() { return num_vectors; }
		inline long  get_num_features() { return num_features; }

		inline long  get_num_sparse_vectors() { return num_sparse_vectors; }

		/// return that we are sparse features
		inline virtual EFeatureClass get_feature_class() { return C_SPARSE; }

		void free_feature_vector(TSparseEntry<ST>* feat_vec, int num, bool free)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (free)
				delete[] feat_vec ;
		} 

	protected:
		/// compute feature vector for sample num
		/// if target is set the vector is written to target
		/// len is returned by reference
		virtual TSparseEntry<ST>* compute_sparse_feature_vector(long num, long& len, TSparseEntry<ST>* target=NULL)
		{
			len=0;
			return NULL;
		}

		/// total number of vectors
		long num_vectors;

		/// total number of features
		long num_features;


		/// number of sparse vectors
		/// this number might be lower than num_vectors, as some vectors might be
		/// 0
		long num_sparse_vectors;

		/// array of sparse vectors of size num_sparse_vec_entries
		TSparse<ST>* sparse_feature_matrix;

		CCache< TSparseEntry<ST> >* feature_cache;
};
#endif
