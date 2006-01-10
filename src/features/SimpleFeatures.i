%module SimpleFeatures%{
 #include "features/SimpleFeatures.h" 
%}

%include "Features.i"
%include "swigfiles/common.i"

template <class ST> class CSimpleFeatures;
template <class ST> class CSimplePreProc;

/* %features("notabstract") CSimpleFeatures; */

template <class ST> class CSimpleFeatures: public CFeatures
{
 public:
	 CSimpleFeatures(LONG size) : CFeatures(size), num_vectors(0), num_features(0), feature_matrix(NULL), feature_cache(NULL)
	 {
	 }

	 CSimpleFeatures(const CSimpleFeatures & orig) : 
		 CFeatures(orig), num_vectors(orig.num_vectors), num_features(orig.num_features), feature_matrix(orig.feature_matrix), feature_cache(orig.feature_cache)
	 {
		 if (orig.feature_matrix)
		 {
			 feature_matrix=new ST(num_vectors*num_features);
			 memcpy(feature_matrix, orig.feature_matrix, sizeof(double)*num_vectors*num_features); 
		 }
	 }

	CSimpleFeatures(CHAR* fname) : CFeatures(fname), num_vectors(0), num_features(0), feature_matrix(NULL), feature_cache(NULL)
	{
	}

	 virtual ~CSimpleFeatures()
	 {
		 delete[] feature_matrix;
		 delete feature_cache;
	 }
  
  ST* get_feature_vector(INT num, INT& len, bool& free)
  {
	  len=num_features; 
//	  if (num>=num_vectors)
//	    {
//	      fprintf(stderr,"feature: %d  num=%d  num_vectors=%d  num_features=%d\n", (INT)this, num, num_vectors, num_features) ;
//	      free=false ;
//	      return NULL ;
//	    }
//	  assert(num<num_vectors);

	  if (feature_matrix)
	  {
		  free=false ;
		  return &feature_matrix[num*num_features];
	  } 
	  else
	  {
		  CIO::message(M_DEBUG, "compute feature!!!\n") ;
		  
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
			  CIO::message(M_DEBUG, "len: %d len2: %d\n", len, num_features);
		  }
		  return feat ;
	  }
  }

  void free_feature_vector(ST* feat_vec, INT num, bool free)
  {
	  if (feature_cache)
		  feature_cache->unlock_entry(num);

	  if (free)
		  delete[] feat_vec ;
  } 
  
  /// get the pointer to the feature matrix
  /// num_feat,num_vectors are returned by reference
  ST* get_feature_matrix(INT &num_feat, INT &num_vec)
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
  virtual void set_feature_matrix(ST* fm, INT num_feat, INT num_vec)
  {
	  feature_matrix=fm;
	  num_features=num_feat;
	  num_vectors=num_vec;
  }

  /// preprocess the feature_matrix
  virtual bool preproc_feature_matrix(bool force_preprocessing=false)
  {
	CIO::message(M_DEBUG, "force: %d\n", force_preprocessing);

	if ( feature_matrix && get_num_preproc())
	{

		for (INT i=0; i<get_num_preproc(); i++)
		{ 
			if ( (!is_preprocessed(i) || force_preprocessing) )
			{
				set_preprocessed(i);

				CIO::message(M_INFO, "preprocessing using preproc %s\n", get_preproc(i)->get_name());
				if (((CSimplePreProc<ST>*) get_preproc(i))->apply_to_feature_matrix(this) == NULL)
					return false;
			}
		}
		return true;
	}
	else
	{
		if (!feature_matrix)
			CIO::message(M_ERROR, "no feature matrix\n");

		if (!get_num_preproc())
			CIO::message(M_ERROR, "no preprocessors available\n");
		return false;
	}
  }

  virtual INT get_size() { return sizeof(ST); }
  virtual inline INT  get_num_vectors() { return num_vectors; }

  inline INT  get_num_features() { return num_features; }
  inline void set_num_features(INT num)
  { 
	  num_features= num; 

	  if (num_features && num_vectors)
	  {
		  delete feature_cache;
		  feature_cache= new CCache<ST>(get_cache_size(), num_features, num_vectors);
	  }
  }

  inline void set_num_vectors(INT num)
  {
	  num_vectors= num;
	  if (num_features && num_vectors)
	  {
		  delete feature_cache;
		  feature_cache= new CCache<ST>(get_cache_size(), num_features, num_vectors);
	  }
  }
  
  /// return that we are simple minded features (just fixed size matrices)
  inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }
  
  virtual bool reshape(INT num_features, INT num_vectors)
  {
	  if (num_features*num_vectors == this->num_features * this->num_vectors)
	  {
		  this->num_features=num_features;
		  this->num_vectors=num_vectors;
		  return true;
	  }
	  else
		  return false;
  }
	
protected:
  /// compute feature vector for sample num
  /// if target is set the vector is written to target
  /// len is returned by reference
  virtual ST* compute_feature_vector(INT num, INT& len, ST* target=NULL)
  {
	  len=0;
	  return NULL;
  }

  /// number of vectors in cache
  INT num_vectors;
 
  /// number of features in cache
  INT num_features;
  
  ST* feature_matrix;
  CCache<ST>* feature_cache;
};

%template(CSimpleCharFeatures) CSimpleFeatures<CHAR>;
%template(CSimpleIntFeatures)  CSimpleFeatures<INT>;

