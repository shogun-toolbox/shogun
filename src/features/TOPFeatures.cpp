CFeatures::CFeatures()
{
    preproc=NULL;
    num_vectors=0;
    num_features=0;
    feature_cache=NULL;
}

CFeatures::~CFeatures()
{
}

double* CHMM::compute_top_feature_vector(CHMM* pos, CHMM* neg, int dim, double* featurevector)
{

    if (!featurevector)
	{
		CIO::message("allocating %.2f M for top feature vector cache\n", 1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()));
		featurevector=new double[ 1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ];
	}

    if (!featurevector)
		return NULL;

    int i,j,p=0,x=dim;

    double posx=pos->model_probability(x);
    double negx=neg->model_probability(x);

    featurevector[p++]=(posx-negx);

#ifndef NORMALIZE_DERIVATIVE
    //first do positive model
    for (i=0; i<pos->get_N(); i++)
    {
	featurevector[p++]=exp(pos->model_derivative_p(i, x)-posx);
	featurevector[p++]=exp(pos->model_derivative_q(i, x)-posx);

	for (j=0; j<pos->get_N(); j++)
	    featurevector[p++]=exp(pos->model_derivative_a(i, j, x)-posx);

	for (j=0; j<pos->get_M(); j++)
	    featurevector[p++]=exp(pos->model_derivative_b(i, j, x)-posx);

    }
    
    for (i=0; i<neg->get_N(); i++)
    {
	featurevector[p++]= - exp(neg->model_derivative_p(i, x)-negx);
	featurevector[p++]= - exp(neg->model_derivative_q(i, x)-negx);

	for (j=0; j<neg->get_N(); j++)
	    featurevector[p++]= - exp(neg->model_derivative_a(i, j, x)-negx);

	for (j=0; j<neg->get_M(); j++)
	    featurevector[p++]= - exp(neg->model_derivative_b(i, j, x)-negx);
    }
#else
#ifdef NORMALIZE_TO_ONE
	double sum=0;
	sum+=featurevector[0]*featurevector[0];

	for (i=0; i<pos->get_N(); i++)
	{
		featurevector[p]=exp(pos->model_derivative_p(i, x)-posx);
		sum+=featurevector[p]*featurevector[p++];
		featurevector[p]=exp(pos->model_derivative_q(i, x)-posx);
		sum+=featurevector[p]*featurevector[p++];

		for (j=0; j<pos->get_N(); j++)
		{
			featurevector[p]=exp(pos->model_derivative_a(i, j, x)-posx);
			sum+=featurevector[p]*featurevector[p++];
		}

		for (j=0; j<pos->get_M(); j++)
		{

			sum+=featurevector[p]*featurevector[p++];
			featurevector[p]=exp(pos->model_derivative_b(i, j, x)-posx);
		}

	}

	for (i=0; i<neg->get_N(); i++)
	{
		featurevector[p]= - exp(neg->model_derivative_p(i, x)-negx);
		sum+=featurevector[p]*featurevector[p++];
		featurevector[p]= - exp(neg->model_derivative_q(i, x)-negx);
		sum+=featurevector[p]*featurevector[p++];

		for (j=0; j<neg->get_N(); j++)
		{
			featurevector[p++]= - exp(neg->model_derivative_a(i, j, x)-negx);
			sum+=featurevector[p]*featurevector[p++];
		}

		for (j=0; j<neg->get_M(); j++)
		{
			featurevector[p++]= - exp(neg->model_derivative_b(i, j, x)-negx);
			sum+=featurevector[p]*featurevector[p++];
		}
	}

	sum=sqrt(sum);
	for (p=0; p<1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()); p++)
		featurevector[p]/=sum;

#else //this is the normalization used in jaahau
    int o_p=1;
    double sum_p=0;
    double sum_q=0;
    //first do positive model
    for (i=0; i<pos->get_N(); i++)
    {
	featurevector[p]=exp(pos->model_derivative_p(i, x)-posx);
	sum_p=exp(pos->get_p(i))*featurevector[p++];
	featurevector[p]=exp(pos->model_derivative_q(i, x)-posx);
	sum_q=exp(pos->get_q(i))*featurevector[p++];

	double sum_a=0;
	for (j=0; j<pos->get_N(); j++)
	{
	    featurevector[p]=exp(pos->model_derivative_a(i, j, x)-posx);
	    sum_a=exp(pos->get_a(i,j))*featurevector[p++];
	}
	p-=pos->get_N();
	for (j=0; j<pos->get_N(); j++)
	    featurevector[p++]-=sum_a;

	double sum_b=0;
	for (j=0; j<pos->get_M(); j++)
	{
	    featurevector[p]=exp(pos->model_derivative_b(i, j, x)-posx);
	    sum_b=exp(pos->get_b(i,j))*featurevector[p++];
	}
	p-=pos->get_M();
	for (j=0; j<pos->get_M(); j++)
	    featurevector[p++]-=sum_b;
    }

    o_p=p;
    p=1;
    for (i=0; i<pos->get_N(); i++)
    {
	featurevector[p++]-=sum_p;
	featurevector[p++]-=sum_q;
    }
    p=o_p;

    for (i=0; i<neg->get_N(); i++)
    {
	featurevector[p]=-exp(neg->model_derivative_p(i, x)-negx);
	sum_p=exp(neg->get_p(i))*featurevector[p++];
	featurevector[p]=-exp(neg->model_derivative_q(i, x)-negx);
	sum_q=exp(neg->get_q(i))*featurevector[p++];

	double sum_a=0;
	for (j=0; j<neg->get_N(); j++)
	{
	    featurevector[p]=-exp(neg->model_derivative_a(i, j, x)-negx);
	    sum_a=exp(neg->get_a(i,j))*featurevector[p++];
	}
	p-=neg->get_N();
	for (j=0; j<neg->get_N(); j++)
	    featurevector[p++]-=sum_a;

	double sum_b=0;
	for (j=0; j<neg->get_M(); j++)
	{
	    featurevector[p]=-exp(neg->model_derivative_b(i, j, x)-negx);
	    sum_b=exp(neg->get_b(i,j))*featurevector[p++];
	}
	p-=neg->get_M();
	for (j=0; j<neg->get_M(); j++)
	    featurevector[p++]-=sum_b;
    }

    p=o_p;
    for (i=0; i<neg->get_N(); i++)
    {
	featurevector[p++]-=sum_p;
	featurevector[p++]-=sum_q;
    }
#endif
#endif
    return featurevector;
}

bool CHMM::save_top_features(CHMM* pos, CHMM* neg, FILE* dest)
{
	int totobs=pos->get_observations()->get_DIMENSION();
    int num_features=1+ pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());
	
	CIO::message("saving %i features (size %i) for %i sequences.\n total file size should be: %i byte.\nwriting as doubles of size %i\n",num_features,sizeof(double)*num_features,totobs,sizeof(double)*num_features*totobs, sizeof(double));

	double* feature_vector=new double[num_features];

	for (int x=0; x<totobs; x++)
	{
		if (!(x % (totobs/10+1)))
			CIO::message("%02d%%.", (int) (100.0*x/totobs));
		else if (!(x % (totobs/200+1)))
			CIO::message(".");

		compute_top_feature_vector(pos, neg, x, feature_vector);
		fwrite(feature_vector, sizeof(double),num_features, dest);
	}

	CIO::message(".done.\n");
	delete[] feature_vector;

	return true;
}

void CHMM::check_and_update_crc(CHMM* pos, CHMM* neg)
{
    bool obs_result=false;
    bool sv_result=false;
    CObservation* o=pos->get_observations();

    if ( (feature_cache_checksums[0]!= (unsigned int) pos) ||
	 (feature_cache_checksums[1]!= (unsigned int) neg) ||
	 (feature_cache_checksums[2]!= (unsigned int) pos->get_N()) ||
	 (feature_cache_checksums[3]!= (unsigned int) pos->get_M()) ||
	 (feature_cache_checksums[4]!= (unsigned int) neg->get_N()) ||
	 (feature_cache_checksums[5]!= (unsigned int) neg->get_M())
       )
    {
	obs_result=true;
	sv_result=true;
    }

    if ((unsigned int) o->get_DIMENSION() != feature_cache_checksums[6])
	obs_result=true;
    
    const int CRCSIZE=32;
    const int CRCSIZEHALF=CRCSIZE/2;

    for (int i=0; i<CRCSIZEHALF && i<o->get_DIMENSION(); i++)
    {
	int idx=i*o->get_DIMENSION()/CRCSIZEHALF;
	unsigned int crc=math.crc32( (unsigned char*) o->get_obs_vector(idx), o->get_obs_T(idx) ); 

#ifdef DEBUG
	printf("OB:idx: %d maxl: %d crc: %x\n", idx, o->get_DIMENSION(), crc);
#endif
	if (features_crc32[i]!=crc)
	  {
	    features_crc32[i]=crc;
	    obs_result=true;
	  }
    }
    
    if (obs_result)
      {
	delete[] feature_cache_obs;
	feature_cache_obs=NULL;
      }
    
    if ((unsigned int) o->get_support_vector_num() != feature_cache_checksums[7])
      sv_result=true;
    
    {
      for (int i=0; i<CRCSIZEHALF && i<o->get_support_vector_num(); i++)
	{
	  int idx=o->get_support_vector_idx(i*o->get_support_vector_num()/CRCSIZEHALF);
	  unsigned int crc=math.crc32( (unsigned char*) o->get_obs_vector(idx), o->get_obs_T(idx) ); 
	  
#ifdef DEBUG
	 CIO::message("SV:idx: %d maxl: %d crc: %x\n", idx, o->get_support_vector_idx(0)+o->get_support_vector_num(), crc);
#endif
	  
	  if (features_crc32[i+CRCSIZEHALF]!=crc)
	    {
	      features_crc32[i+CRCSIZEHALF]=crc;
	      sv_result=true;
	    }
	}	
    } 

    if (sv_result)
    {
	delete[] feature_cache_sv;
	feature_cache_sv=NULL;
    }
    
    feature_cache_checksums[0]=(unsigned int) pos;
    feature_cache_checksums[1]=(unsigned int) neg;
    feature_cache_checksums[2]=(unsigned int) pos->get_N();
    feature_cache_checksums[3]=(unsigned int) pos->get_M();
    feature_cache_checksums[4]=(unsigned int) neg->get_N();
    feature_cache_checksums[5]=(unsigned int) neg->get_M();
    feature_cache_checksums[6]=(unsigned int) pos->get_observations()->get_DIMENSION();
    feature_cache_checksums[7]=(unsigned int) pos->get_observations()->get_support_vector_num();
    feature_cache_in_question=false;
}


void CHMM::invalidate_top_feature_cache(E_TOP_FEATURE_CACHE_VALIDITY v)
{
    switch (v)
    {
	case VALID:
	    break;
	case OBS_INVALID:
	    delete[] feature_cache_obs;
	    feature_cache_obs=NULL;
	    break;
	case SV_INVALID:
	    delete[] feature_cache_sv;
	    feature_cache_sv=NULL;
	    break;
	case QUESTIONABLE:
	    feature_cache_in_question=true;
	    break;
	case INVALID:
	    delete[] feature_cache_obs;
	    delete[] feature_cache_sv;
	    feature_cache_obs=NULL;
	    feature_cache_sv=NULL;
	    feature_cache_in_question=false;
	    num_features=0;
    };
}

void CHMM::subtract_mean_from_top_feature_cache(int num_features, int totobs)
{
	if (feature_cache_obs)
	{
		for (int j=0; j<num_features; j++)
		{
			double mean=0;
			for (int i=0; i<totobs; i++)
				mean+=feature_cache_obs[i*num_features+j];
			for (int i=0; i<totobs; i++)
				feature_cache_obs[i*num_features+j]-=mean;
		}
	}
}

#ifndef PARALLEL

bool CHMM::compute_top_feature_cache(CHMM* pos, CHMM* neg)
{
    num_features=1+ pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());

    if (!feature_cache_sv || !feature_cache_obs || feature_cache_in_question )
    {
	check_and_update_crc(pos, neg);
	
	if (!feature_cache_sv)
	{
	   CIO::message("refreshing top_sv_feature_cache...........\n");

	    int totobs=pos->get_observations()->get_support_vector_num();
		CIO::message("allocating top feature cache of size %.2fM for sv\n", sizeof(double)*num_features*totobs/1024.0/1024.0);
	    feature_cache_sv=new double[num_features*totobs];

	   CIO::message("precalculating top feature vectors for support vectors\n");

		for (int x=0; x<totobs; x++)
		{
		    if (!(x % (totobs/10+1)))
			printf("%02d%%.", (int) (100.0*x/totobs));
		    else if (!(x % (totobs/200+1)))
			printf(".");

		    compute_top_feature_vector(pos, neg, pos->get_observations()->get_support_vector_idx(x), &feature_cache_sv[x*num_features]);
		}
	    
		printf(".done.\n");
		
	}
	else
	   CIO::message("WARNING: using previous top_sv_feature_cache NOT recalculating\n");

	if (!feature_cache_obs)
	{
	   CIO::message("refreshing top_obs_feature_cache...........\n");

	    int totobs=pos->get_observations()->get_DIMENSION();
		CIO::message("allocating top feature cache of size %.2fM for obs\n", sizeof(double)*num_features*totobs/1024.0/1024.0);
	    feature_cache_obs=new double[num_features*totobs];

	   CIO::message("precalculating top feature vectors for observations\n");
	    
	    for (int x=0; x<totobs; x++)
	    {
		if (!(x % (totobs/10+1)))
		   CIO::message("%02d%%.", (int) (100.0*x/totobs));
		else if (!(x % (totobs/200+1)))
		   CIO::message(".");

		
		compute_top_feature_vector(pos, neg, x, &feature_cache_obs[x*num_features]);
	    }

	   CIO::message(".done.\n");
	   
	}
	else
	   CIO::message("WARNING: using previous top_obs_feature_cache NOT recalculating\n");
   
	if ((feature_cache_obs!=NULL) && (feature_cache_sv!=NULL || pos->get_observations()->get_support_vector_num() <= 0))
		return true;
	else 
	    return false;
    }
    else
    {
	   CIO::message("WARNING: using previous top_feature_cache NOT recalculating\n");
	    return true;
    }

}

#else

struct S_THREAD_PARAM2
{
  REAL* dest;
  CHMM * pos, *neg ;
  int    dim ;
}  ;

typedef struct S_THREAD_PARAM2 T_THREAD_PARAM2 ;

void *compute_top_feature_vector_helper(void * p)
{
  T_THREAD_PARAM2* params=((T_THREAD_PARAM2*)p) ;
  CHMM::compute_top_feature_vector(params->pos, params->neg, params->dim, params->dest) ;
  return NULL ;
} ;

bool CHMM::compute_top_feature_cache(CHMM* pos, CHMM* neg)
{
  pthread_t *threads=new pthread_t[NUM_PARALLEL] ;
  T_THREAD_PARAM2 *params=new T_THREAD_PARAM2[NUM_PARALLEL] ;

  num_features=1+ pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());
    
  if (!feature_cache_sv || !feature_cache_obs || feature_cache_in_question )
    {
      check_and_update_crc(pos, neg);
      
      if (!feature_cache_sv)
	{
	 CIO::message("refreshing top_sv_feature_cache...........\n");
	  
	  int totobs=pos->get_observations()->get_support_vector_num();
	  feature_cache_sv=new double[num_features*totobs];
	  
	 CIO::message("precalculating top feature vectors for support vectors\n");
	  
	  for (int x=0; x<totobs; x++)
	    {
	      if (!(x % (totobs/10+1)))
		printf("%02d%%.", (int) (100.0*x/totobs));
	      else if (!(x % (totobs/200+1)))
		printf(".");
	      
	     

	      /* The following code calls the function
  		   compute_top_feature_vector(pos, neg, x, &feature_cache_sv[x*num_features]);
		 NUM_PARALLEL times (in parallel).
	      */
	      if (x%NUM_PARALLEL==0)
		{
		  int i ;
		  for (i=0; i<NUM_PARALLEL; i++)
		    if (x+i<pos->p_observations->get_DIMENSION())
		      {
			params[i].pos=pos ;
			params[i].neg=neg ;
			params[i].dim=x+i ;
			params[i].dest=&feature_cache_sv[x*num_features] ;
#ifdef SUNOS
			thr_create(NULL,0,compute_top_feature_vector_helper, (void*)&params[i], PTHREAD_SCOPE_SYSTEM, &threads[i]) ;
#else // SUNOS
			pthread_create(&threads[i], NULL, compute_top_feature_vector_helper, (void*)&params[i]) ;
#endif // SUNOS
		      } ;
		  for (i=0; i<NUM_PARALLEL; i++)
		    if (x+i<pos->p_observations->get_DIMENSION())
		      {
			void * ret ;
			pthread_join(threads[i], &ret) ;
		      } ;
		} ;
	    }
	  
	 CIO::message(".done.\n");
	 
	}
      else
	printf("WARNING: using previous top_sv_feature_cache NOT recalculating\n");
      
      if (!feature_cache_obs)
	{
	 CIO::message("refreshing top_obs_feature_cache...........\n");
	  
	  int totobs=pos->get_observations()->get_DIMENSION();
	  feature_cache_obs=new double[num_features*totobs];
	  
	 CIO::message("precalculating top feature vectors for observations\n");
	  
	  for (int x=0; x<totobs; x++)
	    {
	      if (!(x % (totobs/10+1)))
		printf("%02d%%.", (int) (100.0*x/totobs));
	      else if (!(x % (totobs/200+1)))
		printf(".");
	      
	     
	      
	      compute_top_feature_vector(pos, neg, x, &feature_cache_obs[x*num_features]);
	    }
	  
	 CIO::message(".done.\n");
	 
	}
      else
	printf("WARNING: using previous top_obs_feature_cache NOT recalculating\n");
      
      if ((feature_cache_obs!=NULL) && (feature_cache_sv!=NULL || pos->get_observations()->get_support_vector_num() <= 0))
	return true;
      else 
	return false;
    }
  else
    {
     CIO::message("WARNING: using previous top_feature_cache NOT recalculating\n");
      return true;
    }
  
}
#endif
