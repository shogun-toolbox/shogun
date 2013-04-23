/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 * Copyright (C) 2013 Kevin Hughes
 *
 * Thanks to Fernando José Iglesias García (shogun) 
 *           and Matthieu Perrot (scikit-learn)
 */

#include <shogun/lib/common.h>

#ifdef HAVE_LAPACK

#include <shogun/multiclass/MCLDA.h>
#include <shogun/machine/NativeMulticlassMachine.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>

using namespace shogun;

CMCLDA::CMCLDA(float64_t tolerance, bool store_cov)
: CNativeMulticlassMachine(), m_tolerance(tolerance), m_store_cov(store_cov), m_num_classes(0), m_dim(0)
{
	init();
}

CMCLDA::CMCLDA(CDenseFeatures<float64_t>* traindat, CLabels* trainlab, float64_t tolerance, bool store_cov)
: CNativeMulticlassMachine(), m_tolerance(tolerance), m_store_cov(store_cov), m_num_classes(0), m_dim(0)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
}

CMCLDA::~CMCLDA()
{
	SG_UNREF(m_features);

	cleanup();
}

void CMCLDA::init()
{
	SG_ADD(&m_tolerance, "m_tolerance", "Tolerance member.", MS_AVAILABLE);
	SG_ADD(&m_store_cov, "m_store_cov", "Store covariance member", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_features, "m_features", "Feature object.", MS_NOT_AVAILABLE);
	SG_ADD(&m_means, "m_means", "Mean vectors list", MS_NOT_AVAILABLE);
	SG_ADD(&m_cov, "m_cov", "covariance matrix", MS_NOT_AVAILABLE);
	SG_ADD(&m_xbar, "m_xbar", "total mean", MS_NOT_AVAILABLE);
    SG_ADD(&m_scalings, "m_scalings", "scalings", MS_NOT_AVAILABLE);
    SG_ADD(&m_rank, "m_rank", "rank", MS_NOT_AVAILABLE);
    SG_ADD(&m_coef, "m_coef", "weight vector", MS_NOT_AVAILABLE);
    SG_ADD(&m_intercept, "m_intercept", "intercept", MS_NOT_AVAILABLE);
	m_features  = NULL;
}

void CMCLDA::cleanup()
{
	m_num_classes = 0;
}

CMulticlassLabels* CMCLDA::apply_multiclass(CFeatures* data)
{
    if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")

		set_features((CDotFeatures*) data);
	}

	if ( !m_features )
		return NULL;
		
	int32_t num_vecs = m_features->get_num_vectors();
	ASSERT(num_vecs > 0)
	ASSERT( m_dim == m_features->get_dim_feature_space() )

    // collect features into a matrix
	CDenseFeatures< float64_t >* rf = (CDenseFeatures< float64_t >*) m_features;
	
	SGMatrix< float64_t > X = SGMatrix< float64_t >(num_vecs, m_dim);
	
	int i, j;
	int32_t vlen;
	bool vfree;
	float64_t* vec;
	for ( i = 0 ; i < num_vecs ; ++i )
	    for ( j = 0 ; j < m_dim ; ++j )
        {
            vec = rf->get_feature_vector(i, vlen, vfree);
            X(i,j) = vec[j] - m_xbar[j];
	    }

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying X ...\n");
	SGMatrix< float64_t >::display_matrix(X.matrix, num_vecs, m_dim);
#endif
	
	// center and scale data
    SGMatrix< float64_t > Xs = SGMatrix< float64_t >(num_vecs, m_rank);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	            num_vecs, m_rank, m_dim, 1.0,
	            X.matrix, num_vecs,
	            m_scalings.matrix, m_dim, 0.0,
	            Xs.matrix, num_vecs);
    	
#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying Xs ...\n");
	//SGVector< float64_t >::display_vector(Xs.vector, num_vecs);
	SGMatrix< float64_t >::display_matrix(Xs.matrix, num_vecs, m_rank);
#endif
	
	// decision function
	SGMatrix< float64_t > d = SGMatrix< float64_t >(num_vecs, m_num_classes);
	for ( i = 0 ; i < num_vecs ; ++i )
	    for ( j = 0 ; j < m_num_classes ; ++j )
	        d(i,j) = Xs[i] * m_coef[j] + m_intercept[j];      

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying d ...\n");
	SGMatrix< float64_t >::display_matrix(d.matrix, num_vecs, m_num_classes); 
#endif

	// argmax to apply labels       
	CMulticlassLabels* out = new CMulticlassLabels(num_vecs);
	for ( i = 0 ; i < num_vecs ; ++i )
        out->set_label(i, SGVector<float64_t>::arg_max(d.matrix+i, num_vecs, m_num_classes));

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying labels ...\n");
	SGVector<float64_t>::display_vector(out->get_labels().vector, out->get_num_labels());
#endif
	
	return out;
}

bool CMCLDA::train_machine(CFeatures* data)
{
    if ( !m_labels )
		SG_ERROR("No labels allocated in MCLDA training\n")

	if ( data )
	{
		if ( !data->has_property(FP_DOT) )
			SG_ERROR("Speficied features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}
	if ( !m_features )
		SG_ERROR("No features allocated in MCLDA training\n")
	SGVector< int32_t > train_labels = ((CMulticlassLabels*) m_labels)->get_int_labels();
	if ( !train_labels.vector )
		SG_ERROR("No train_labels allocated in MCLDA training\n")
	
	cleanup();
	
    m_num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();
	m_dim = m_features->get_dim_feature_space();
	int32_t num_vec  = m_features->get_num_vectors();
	if ( num_vec != train_labels.vlen )
		SG_ERROR("Dimension mismatch between features and labels in MCLDA training")
		
	int32_t* class_idxs = SG_MALLOC(int32_t, num_vec*m_num_classes);
	// number of examples of each class
	int32_t* class_nums = SG_MALLOC(int32_t, m_num_classes);
	memset(class_nums, 0, m_num_classes*sizeof(int32_t));
	int32_t class_idx;
	int32_t i, j, k, l;
	for ( i = 0 ; i < train_labels.vlen ; ++i )
	{
		class_idx = train_labels.vector[i];

		if ( class_idx < 0 || class_idx >= m_num_classes )
		{
			SG_ERROR("found label out of {0, 1, 2, ..., num_classes-1}...")
			return false;
		}
		else
		{
			class_idxs[ class_idx*num_vec + class_nums[class_idx]++ ] = i;
		}
	}
	
	for ( i = 0 ; i < m_num_classes ; ++i )
	{
		if ( class_nums[i] <= 0 )
		{
			SG_ERROR("What? One class with no elements\n")
			return false;
		}
	}
	
	CDenseFeatures< float64_t >* rf = (CDenseFeatures< float64_t >*) m_features;
	
	// if ( m_store_cov )
		index_t * cov_dims = SG_MALLOC(index_t, 3);
		cov_dims[0] = m_dim;
		cov_dims[1] = m_dim;
		cov_dims[2] = m_num_classes;
		SGNDArray< float64_t > covs(cov_dims, 3);
	
	m_means = SGMatrix< float64_t >(m_dim, m_num_classes, true);

	// matrix of all samples
	SGMatrix< float64_t > X = SGMatrix< float64_t >(num_vec, m_dim, true);
	int32_t iX = 0;
	
	m_means.zero();
	m_cov.zero();
	
	int32_t vlen;
	bool vfree;
	float64_t* vec;
	for ( k = 0 ; k < m_num_classes ; ++k )
	{
	    // gather all the samples for class k into buffer
	    // and calculate the mean of class k
	    SGMatrix< float64_t > buffer(class_nums[k], m_dim);    
	    for ( i = 0 ; i < class_nums[k] ; ++i )
		{
            vec = rf->get_feature_vector(class_idxs[k*num_vec + i], vlen, vfree);
			ASSERT(vec)
            
            for ( j = 0 ; j < vlen ; ++j )
			{
				m_means[k*m_dim + j] += vec[j];
				buffer[i + j*class_nums[k]] = vec[j];
			}
		
		}
		for ( j = 0 ; j < m_dim ; ++j )
			m_means[k*m_dim + j] /= class_nums[k];
	    
	    // subtract the mean of class k from each sample of class k
	    // and store the centered data in Xc
	    for ( i = 0 ; i < class_nums[k] ; ++i )
	    {
			for ( j = 0 ; j < m_dim ; ++j )
			{
				buffer[i + j*class_nums[k]] -= m_means[k*m_dim + j];
				X(iX,j) = buffer[i + j*class_nums[k]];
	        }
	        iX+=1;
        } 

	    if ( m_store_cov )
	    {
            // calc cov = buffer.T * buffer
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                        m_dim, m_dim, m_dim, 1.0,
				        buffer.matrix, m_dim, 
				        buffer.matrix, m_dim, 0.0, 
				        covs.get_matrix(k), m_dim);
	    }

	}   

#ifdef DEBUG_MCLDA
	SG_PRINT("\n>>> Displaying means ...\n");
	SGMatrix< float64_t >::display_matrix(m_means.matrix, m_dim, m_num_classes);
#endif
	
	if ( m_store_cov )
    {
        m_cov = SGMatrix< float64_t >(m_dim, m_dim, true);
        m_cov.zero();
        
        // normalize the covar mat
        for ( i = 0 ; i < m_dim*m_dim ; ++i )
        {
            for ( k = 0 ; k < m_num_classes ; ++k ) 
                m_cov[i] += covs.get_matrix(k)[i];
                
            m_cov[i] / m_num_classes;
        }
    }

#ifdef DEBUG_MCLDA		
	if ( m_store_cov )
    {
	    SG_PRINT("\n>>> Displaying cov ...\n");
	    SGMatrix< float64_t >::display_matrix(m_cov.matrix, m_dim, m_dim);
	}
#endif
	
	///////////////////////////////////////////////////////////
	// 1) within (univariate) scaling by with classes std-dev
	
	// std-dev of X
	m_xbar = SGVector< float64_t >(m_dim);
	m_xbar.zero();
	for ( i = 0 ; i < num_vec ; ++i )
	    for ( j = 0 ; j < m_dim ; ++j )
		    m_xbar[j] += X(i,j) / num_vec;

	SGVector< float64_t > std = SGVector< float64_t >(m_dim);
	std.zero();
	for ( i = 0 ; i < num_vec ; ++i )
	    for ( j = 0 ; j < m_dim ; ++j )
		    std[j] += (X(i,j) - m_xbar[j])*(X(i,j) - m_xbar[j]);

	for ( j = 0 ; j < m_dim ; ++j )
	{
	    std[j] = sqrt(std[j] / num_vec);
	    if(std[j] == 0)
	        std[j] = 1;
    }
	
	float64_t fac = 1.0 / (num_vec - m_num_classes);
    
    ///////////////////////////////
	// 2) Within variance scaling
	for ( i = 0 ; i < num_vec ; ++i )
	    for ( j = 0 ; j < m_dim ; ++j )
	        X(i,j) = sqrt(fac) * X(i,j) / std[j];	
    
    // SVD of centered (within)scaled data
    /* calling external lib, buffer = U * S * V^T, U is not interesting here */
    SGVector< float64_t > S  = SGVector< float64_t >(m_dim);
    SGMatrix< float64_t > V  = SGMatrix< float64_t >(m_dim, m_dim);
    
	char jobu = 'N', jobvt = 'A';
	int m = num_vec, n = m_dim;
	int lda = m, ldu = m, ldvt = n;
	int info = -1;
	
	wrap_dgesvd(jobu, jobvt, m, n, X.matrix, lda, S.vector, NULL, ldu,
			V.matrix, ldvt, &info);
	ASSERT(info == 0)
	
	//SGMatrix< float64_t >::display_matrix(V.matrix, m_dim, m_dim);
	//SGVector< float64_t >::display_vector(S.vector, m_dim);
	
	int rank = 0;
	while ( S[rank] > m_tolerance && rank < m_dim)
	    rank++;
	//printf("%d",rank);
	
	if ( rank < m_dim )
        SG_ERROR("Warning: Variables are collinear\n")
	
	SGMatrix< float64_t > scalings  = SGMatrix< float64_t >(m_dim, rank);
    for ( i = 0 ; i < m_dim ; ++i )
	    for ( j = 0 ; j < rank ; ++j )
		    scalings(i,j) = V(j,i) / std[j] / S[j];

#ifdef DEBUG_MCLDA	
	SG_PRINT("\n>>> Displaying scalings ...\n");
	SGMatrix< float64_t >::display_matrix(scalings.matrix, m_dim, rank);
#endif
	
	///////////////////////////////
	// 3) Between variance scaling
	
	// Xc = m_means dot scalings
	SGMatrix< float64_t > Xc  = SGMatrix< float64_t >(m_num_classes, rank);
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
	            m_num_classes, rank, m_dim, 1.0,
				m_means.matrix, m_dim, 
				scalings.matrix, rank, 0.0, 
				Xc.matrix, m_num_classes);
    for ( i = 0 ; i < m_num_classes ; ++i )	
        for ( j = 0 ; j < rank ; ++j )
		    Xc(i,j) *= sqrt(class_nums[i] * fac);
    //SGMatrix< float64_t >::display_matrix(Xc.matrix, m_num_classes, rank);

    // Centers are living in a space with n_classes-1 dim (maximum)
    // Use svd to find projection in the space spanned by the
    // (n_classes) centers
    S  = SGVector< float64_t >(rank);
    V  = SGMatrix< float64_t >(rank, rank);
    
	jobu = 'N', jobvt = 'A';
	m = m_num_classes, n = rank;
	lda = m, ldu = m, ldvt = n;
	info = -1;
	
	wrap_dgesvd(jobu, jobvt, m, n, Xc.matrix, lda, S.vector, NULL, ldu,
			V.matrix, ldvt, &info);
	ASSERT(info == 0)
    
    //SGMatrix< float64_t >::display_matrix(V.matrix, rank, rank);
	//SGVector< float64_t >::display_vector(S.vector, rank);
	
	m_rank = 0;
	while ( S[m_rank] > m_tolerance*S[0] && m_rank < rank )
	    m_rank++;
	//printf("%d",m_rank);
	
	// compose the scalings
    // m_scalings = scalings dot V^T[:,:new_rank]
    m_scalings  = SGMatrix< float64_t >(rank, m_rank);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
	            m_dim, m_rank, rank, 1.0,
				scalings.matrix, rank, 
				V.matrix, rank, 0.0, 
				m_scalings.matrix, rank);
								
#ifdef DEBUG_MCLDA	
	SG_PRINT("\n>>> Displaying m_scalings ...\n");
	SGMatrix< float64_t >::display_matrix(m_scalings.matrix, rank, m_rank);
#endif

	// weight vectors / centroids
	// m_coef = (m_means - xbar) dot m_scalings
	SGMatrix< float64_t > meansc = SGMatrix< float64_t >(m_dim, m_num_classes);
	for ( i = 0 ; i < m_dim ; ++i )
	    for ( j = 0 ; j < m_num_classes ; ++j )
	        meansc(i,j) = m_means(i,j) - m_xbar[j];    
	
	//SGMatrix< float64_t >::display_matrix(meansc.matrix, m_dim, m_num_classes);
	m_coef = SGMatrix< float64_t >(m_num_classes, m_rank);
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                m_num_classes, m_rank, rank, 1.0,
                meansc.matrix, m_dim,
                m_scalings.matrix, rank, 0.0,
                m_coef.matrix, m_num_classes);
			
#ifdef DEBUG_MCLDA	
	SG_PRINT("\n>>> Displaying m_coefs ...\n");
	SGMatrix< float64_t >::display_matrix(m_coef.matrix, m_num_classes, m_rank);
#endif
    // intercept
    m_intercept  = SGVector< float64_t >(m_num_classes);
    m_intercept.zero();
    for ( j = 0 ; j < m_num_classes ; ++j )
        m_intercept[j] = -0.5*m_coef[j]*m_coef[j] + log(class_nums[j]/float(num_vec));

#ifdef DEBUG_MCLDA	
	SG_PRINT("\n>>> Displaying m_intercept ...\n");
	SGVector< float64_t >::display_vector(m_intercept.vector, m_num_classes);
#endif    
    
    SG_FREE(class_idxs);
	SG_FREE(class_nums);
	
	return true;
}

#endif /* HAVE_LAPACK */
