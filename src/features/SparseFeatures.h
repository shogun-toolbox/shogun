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

template <class ST> class CSparseFeatures;

template <class ST> class CSparseFeatures: public CFeatures
{
	public:
		//features are an array of TSparse, sorted w.r.t. vec_index (increasing) and
		//withing same vec_index w.r.t. feat_index (increasing);
		struct TSparseEntry
		{
			int feat_index;
			ST entry;
		};

		struct TSparse
		{
			int vec_index;

			int num_feat_entries;
			TSparseEntry* features;
		};

		CSparseFeatures(long size) : CFeatures(size), num_vectors(0), num_features(0), num_sparse_vectors(0), sparse_feature_matrix(NULL), feature_cache(NULL)
		{
		}

		CSparseFeatures(const CSparseFeatures & orig) : 
			CFeatures(orig), num_vectors(orig.num_vectors), num_features(orig.num_features), num_sparse_vectors(orig.num_sparse_vectors), sparse_feature_matrix(orig.sparse_feature_matrix), feature_cache(orig.feature_cache)
			{
				if (orig.feature_matrix)
				{
					feature_matrix=new ST(num_vectors*num_features);
					memcpy(feature_matrix, orig.feature_matrix, sizeof(double)*num_vectors*num_features); 
				}
			}

		CSparseFeatures(char* fname) : CFeatures(fname), num_vectors(0), num_features(0), num_sparse_vectors(0), sparse_feature_matrix(NULL), feature_cache(NULL)
		{
		}

		virtual ~CSparseFeatures()
		{
			delete[] sparse_feature_matrix;
			delete feature_cache;
		}

		/** converts a sparse feature vector into a dense one
		  preprocessed compute_feature_vector  
		  @param num index of feature vector
		  @param len length is returned by reference
		  */
		ST* get_full_feature_vector(long num, long& len, bool& free)
		{
			len=num_features; 
			assert(num<num_vectors);

			if (feature_matrix)
			{
				//      CIO::message("returning %i th column of feature_matrix\n", (int)num) ;
				free=false ;
				return &feature_matrix[num*num_features];
			} 
			else
			{
				//CIO::message("computing %i th feature vector\n", (int)num);

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
					//CIO::message("preprocessing %i th feature vector\n", (int)num) ;

					int tmp_len=len;
					ST* tmp_feat_before = feat;
					ST* tmp_feat_after = NULL;

					for (int i=0; i<get_num_preproc(); i++)
					{
						tmp_feat_after=((CSimplePreProc<ST>*) get_preproc(i))->apply_to_feature_vector(tmp_feat_before, tmp_len);

						if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
							delete[] tmp_feat_before;
						tmp_feat_before=tmp_feat_after;
					}

					memcpy(feat, tmp_feat_after, sizeof(ST)*tmp_len);
					delete[] tmp_feat_after;
					//len=num_features=len2 ;
					len=tmp_len ;
					CIO::message(stderr, "len: %d len2: %d\n", len, num_features);
				}
				return feat ;
			}
		}

		/** get feature vector for sample num
		  from the matrix as it is if matrix is
		  initialized, else return
		  preprocessed compute_feature_vector  
		  @param num index of feature vector
		  @param len length is returned by reference
		  */
		TSparseEntry* get_sparse_feature_vector(long num, long& len, bool& free)
		{
			len=num_features; 
			assert(num<num_vectors);

			if (feature_matrix)
			{
				//      CIO::message("returning %i th column of feature_matrix\n", (int)num) ;
				free=false ;
				return &feature_matrix[num*num_features];
			} 
			else
			{
				//CIO::message("computing %i th feature vector\n", (int)num);

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
					//CIO::message("preprocessing %i th feature vector\n", (int)num) ;

					int tmp_len=len;
					ST* tmp_feat_before = feat;
					ST* tmp_feat_after = NULL;

					for (int i=0; i<get_num_preproc(); i++)
					{
						tmp_feat_after=((CSimplePreProc<ST>*) get_preproc(i))->apply_to_feature_vector(tmp_feat_before, tmp_len);

						if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
							delete[] tmp_feat_before;
						tmp_feat_before=tmp_feat_after;
					}

					memcpy(feat, tmp_feat_after, sizeof(ST)*tmp_len);
					delete[] tmp_feat_after;
					//len=num_features=len2 ;
					len=tmp_len ;
					CIO::message(stderr, "len: %d len2: %d\n", len, num_features);
				}
				return feat ;
			}
		}

		void free_sparse_feature_vector(ST* feat_vec, int num, bool free)
		{
			if (feature_cache)
				feature_cache->unlock_entry(num);

			if (free)
				delete[] feat_vec ;
		} 

		/// get the pointer to the sparse feature matrix
		/// num_feat,num_vectors are returned by reference
		/// the actual number of sparse vectors is returned in num_sparse_vec
		TSparse* get_sparse_feature_matrix(long &num_feat, long &num_vec, long &num_sparse_vec)
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
		virtual void set_sparse_feature_matrix(TSparse* sfm, long num_feat, long num_vec, long num_sparse_vec)
		{
			sparse_feature_matrix=sfm;
			num_features=num_feat;
			num_vectors=num_vec;
			num_sparse_vectors=num_sparse_vec;
		}

		/// gets a copy of a full  feature matrix
		/// num_feat,num_vectors are returned by reference
		/// the actual number of sparse vectors is returned in num_sparse_vec
		ST* get_full_feature_matrix(long &num_feat, long &num_vec, long &num_sparse_vec)
		{
			CIO::warning("converting sparse features to full feature matrix\n");
			num_feat=num_features;
			num_vec=num_vectors;

			ST* fm=new ST[num_feat*num_vec];

			if (fm)
			{
				for (long i=0; i< num_vec; i++)
				{
					for (long j=0; j< num_feat; j++)
					{
						fm[i*num_feat+ j]=blalblabla;
					}
				}
			}

			return fm;
		}

		/** creates a sparse feature matrix from a full dense feature matrix
		  necessary to set feature_matrix, num_features, num_vectors, num_sparse vec
		  where num_features is the column offset, and columns are linear in memory
		  see above for definition of sparse_feature_matrix
		  */
		virtual void set_full_feature_matrix(ST* ffm, long num_feat, long num_vec)
		{
			sparse_feature_matrix=sfm;
			num_features=num_feat;
			num_vectors=num_vec;

			CIO::warning("converting dense feature matrix to sparse one\n");
			num_feat=num_features;
			num_vec=num_vectors;

			int* num_feat_entries=new int[num_vectors];

			long vec_idx=0;
			long feat_idx=0;


			if (num_feat_entries)
			{
				for (long i=0; i< num_feat*num_vec; i++)
					num_feat_entries[i]=0;

				long num_sparse_vec=0;

				// count nr of non sparse features
				for (long i=0; i< num_vec; i++)
				{
					for (long j=0; j< num_feat; j++)
					{
						if (ffm[i*num_feat + j])
							num_feat_entries[i]++;
					}

					if (num_feat_entries[i])
						num_sparse_vec++;
				}

				num_sparse_vectors=num_sparse_vec;

				if (num_sparse_vec>0)
				{
					long sparse_vec_idx=0;
					sparse_feature_matrix=new TSparse[num_sparse_vectors];

					if (sparse_feature_matrix)
					{
						for (long i=0; i< num_vec; i++)
						{
							if (num_feat_entries[i]>0)
							{
								sparse_feature_matrix[sparse_vec_idx].features= new TSparseEntry[num_feat_entries[i]];
								if (sparse_feature_matrix[sparse_vec_idx])
								{
									long sparse_feat_idx=0;

									for (long j=0; j< num_feat; j++)
									{
										long pos= i*num_feat + j;

										if (ffm[pos])
											sparse_feature_matrix[sparse_vec_idx].features[sparse_feat_idx++]=ffm[pos];
									}

								}
								num_sparse_vec++;
							}
						}
					}
				}
			}
		}


		virtual bool preproc_sparse_feature_matrix(bool force_preprocessing=false)
		{
			CIO::message("preprocd: %d, force: %d\n", preprocessed, force_preprocessing);

			if ( sparse_feature_matrix && get_num_preproc() && (!preprocessed || force_preprocessing) )
			{
				preprocessed=true;	

				for (int i=0; i<get_num_preproc(); i++)
				{
					CIO::message("preprocessing using preproc %s\n", get_preproc(i)->get_name());
					if (((CSparsePreProc<ST>*) get_preproc(i))->apply_to_sparse_feature_matrix(this) == NULL)
						return false;
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

	protected:
		/// compute feature vector for sample num
		/// if target is set the vector is written to target
		/// len is returned by reference
		virtual ST* compute_feature_vector(long num, long& len, ST* target=NULL)
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
		TSparse* sparse_feature_matrix;

		CCache<TSParse>* feature_cache;
};
#endif

