/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 *
 * Indentation: bcpp -f 1 -s -ylcnc -bcl -i 4
 */

#include "lib/config.h"

#ifdef HAVE_MINDY

#include <mindy.h>

#include "lib/common.h"
#include "features/MindyGramFeatures.h"
#include "lib/io.h"
#include "kernel/MindyGramKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"

/*
 * Similarity parameters
 */
param_spec_t p_map[] = {
    { "expo",  SP_EXPO,  2.0,  "Exponent (polynomial, minkowski)" },
    { "shift", SP_SHIFT, 0.0,  "Shift value (polynomial)" },
    { "dist",  SP_DIST,  ST_MINKOWSKI, "Distance name (rbf)" },
    { "width", SP_WIDTH, 1.0,  "Kernel width (rbf)" },
    { NULL },
};

/**
 * Mindy kernel constructor
 * @param cache Cache size to use (?)
 * @param meas Similarity measure to use
 * @param w Kernel width
 */
CMindyGramKernel::CMindyGramKernel(int32_t ch, char *meas, float64_t w)
: CKernel(ch)
{
	/* Init attributes */
	measure=meas;
	norm=NO_NORMALIZATION;
	width=w;
	cache=0;
	
	/* Check for similarity coefficients */
	simcof=sico_get_type(measure);

	/* Create similarity measure */
	SG_INFO("Initializing Mindy kernel.\n");
	if (simcof==SC_NONE)
		kernel=sm_create(sm_get_type(measure));
	else
		kernel=sm_create(ST_MINKERN);
   
	SG_INFO("Mindy similarity measure: %s (using %s).\n",
		measure, sm_get_descr(kernel->type));

	/* Initialize optimization */
	if (kernel->type == ST_LINEAR)
	{
		SG_INFO("Optimization supported.\n");
		properties |= KP_LINADD;
	}

	normal=NULL;
	clear_normal();

	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CMindyGramKernel::CMindyGramKernel(
	CFeatures* l, CFeatures* r, char *m, float64_t w)
: CKernel(10), measure(m), width(w)
{
	/* Check for similarity coefficients */
	simcof=sico_get_type(measure);

	/* Create similarity measure */
	SG_INFO("Initializing Mindy kernel.\n");
	if (simcof==SC_NONE)
		kernel=sm_create(sm_get_type(measure));
	else
		kernel=sm_create(ST_MINKERN);
   
	SG_INFO("Mindy similarity measure: %s (using %s).\n",
		 measure, sm_get_descr(kernel->type));

	/* Initialize optimization */
	if (kernel->type == ST_LINEAR)
	{
		SG_INFO("Optimization supported.\n");
		properties |= KP_LINADD;
	}

	normal=NULL;
	clear_normal();
	init_normalizer(new CSqrtDiagKernelNormalizer());

	init(l, r);
}

/*
 * Set MD5 cache
 */
void CMindyGramKernel::set_md5cache(int32_t c)
{
    cache = c;
    if (cache <= 0) 
        return;
        
    SG_INFO("Creating MD5 cache of %d kb", cache);
    md5_cache_create(cache);
} 

/*
 * Set parameters 
 */
void CMindyGramKernel::set_param(char *param) 
{
    /* Parse and set parameters */
    parse_params(param);

    /* Display paramater list */
    for (int32_t i = 0; p_map[i].name; i++) {
        if (p_map[i].idx != SP_DIST)
            SG_INFO( "Param %8s=%8.6f\t %s\n", 
			p_map[i].name, p_map[i].val, p_map[i].descr);
        else
            SG_INFO( "Param %8s=%s\t %s\n", p_map[i].name, 
                        sm_get_name((sm_type_t) p_map[i].val), 
                        p_map[i].descr);
    }
} 

/**
 * Mindy kernel destructor
 */
CMindyGramKernel::~CMindyGramKernel()
{
    cleanup();
    
    if (cache > 0)
        md5_cache_destroy();
    
    sm_destroy(kernel);
}

/**
 * Parse provided parameters
 */
void CMindyGramKernel::parse_params(char *pa)
{
    int32_t i;
    char *t, *p;

    if (strlen(pa) == 0)
        return;

    /* Loop over delimited parameter definitions */
    while ((t = strsep(&pa, ",;"))) {
        for (i = 0; p_map[i].name; i++) {
            /* Check for parameter name */
            size_t l = strlen(p_map[i].name);
            if (!strncasecmp(t, p_map[i].name, l)) {
                p = t + l + 1;
                if (p_map[i].idx == SP_DIST)
                    p_map[i].val = sm_get_type(p);
                else
                    p_map[i].val = atof(p);
                break;
            }
        }
     	if (!p_map[i].name)
            SG_WARNING( "Unknown parameter '%s'. Skipping", t);
     }   

     /* Set parameters */	
     for (i = 0; p_map[i].name; i++)
	sm_set_param(kernel, p_map[i].idx, p_map[i].val);	 
} 

/**
 * Clean up method
 */
void CMindyGramKernel::cleanup()
{
    delete_optimization();
    clear_normal();

	CKernel::cleanup();
}

/**
 * Remove left-hand side of data. In opposite to remove_rhs() this
 * method also invalidated any optimization and precalculated
 * normalization diagonals.
 */
void CMindyGramKernel::remove_lhs()
{
    delete_optimization();

#ifdef SVMLIGHT
    if (lhs)
        cache_reset();
#endif

    lhs = NULL ;
    rhs = NULL ;
}

/**
 * Remove right-hand side of data and replace with left-hand side
 */
void CMindyGramKernel::remove_rhs()
{
#ifdef SVMLIGHT
    if (rhs)
        cache_reset();
#endif

    if (sdiag_lhs != sdiag_rhs)
        delete[] sdiag_rhs;

    sdiag_rhs = sdiag_lhs;
    rhs = lhs;
}

/**
 * Initialize the kernel with features vectors
 * @param l Set of feature vectors
 * @param r Set of feature vectors
 * @return true on success, false otherwise
 */
bool CMindyGramKernel::init(CFeatures* l, CFeatures* r)
{
    SG_DEBUG( "Initializing MindyGramKernel %p %p\n", l, r);
    /* Call constructor of super class */
    bool result = CKernel::init(l,r);

    /* Assert correct types of features */
    ASSERT(l->get_feature_class()== C_MINDYGRAM);
    ASSERT(r->get_feature_class()==C_MINDYGRAM);
    ASSERT(l->get_feature_type()==F_ULONG);
    ASSERT(r->get_feature_type()==F_ULONG);

    return init_normalizer();
}

/**
 * Compute kernel value for the given pair of feature vectors
 * @param i Index to lhs vector
 * @param j Index to rhs vector
 * @param kernel value
 */
float64_t CMindyGramKernel::compute(int32_t i, int32_t j)
{
    /* Cast things to mindy gram features */
    CMindyGramFeatures *lm = (CMindyGramFeatures *) lhs;
    CMindyGramFeatures *rm = (CMindyGramFeatures *) rhs;

    /* Call (internal) mindy comparison function */
    float64_t result = gram_cmp(kernel, lm->get_feature_vector(i),
        rm->get_feature_vector(j));
    
    /* Compute similartiy coefficients and convert to distance */
    if (simcof != SC_NONE)
        result = 1 - sico(simcof, result, sdiag_lhs[i], sdiag_rhs[j]);

    if (sm_get_class(kernel->type) == SC_DIST || simcof != SC_NONE) {
        if (width > 1e-10) {
              /* Distance to kernel using RBF */
              result = exp(-result / width);
        } else {
            if (i != j) {
                /* Distance to kernel, the Hilbertian way */
                result = 0.5 * (sdiag_lhs[i] + sdiag_rhs[j] - result);    
            } else {
                /* Distance based norm  */
                gram_t *zero = gram_empty();
                result = gram_cmp(kernel, lm->get_feature_vector(i), zero);
                gram_destroy(zero);
            }  
        }   
    }    
}

/**
 * Add a featrue vector to a global normal vector
 * @param i Index of feature vector
 * @param w Weight for addition (usually alpha_i)
 */
void CMindyGramKernel::add_to_normal(int32_t i, float64_t w)
{
    /* Add indexed vector to normal */
    CMindyGramFeatures *lm = (CMindyGramFeatures *) lhs;
    
    /* Initialize empty normal vector if necessary */
    if (!normal) 
        normal = gram_empty();

    gram_add(normal, lm->get_feature_vector(i),
                     normalizer->normalize_lhs(w, i));

    set_is_initialized(true);
}

/**
 * Clear global normal vector
 */
void CMindyGramKernel::clear_normal()
{
    if (normal)
        gram_destroy(normal);
    normal = NULL;
    set_is_initialized(false);
}

/**
 * Initialize optimization with a set of feature vectors
 * @param n Number of vectors
 * @param is Array of indices in left-hand side of data
 * @param ws Array of weights
 */
bool CMindyGramKernel::init_optimization(int32_t n, int32_t *is, float64_t * ws)
{
    /* Delete old optimization */
    delete_optimization();

    /* Return empty optimization if no vectors are given */
    if (n <= 0) {
        set_is_initialized(true);
        SG_DEBUG( "empty set of SVs\n");
        return true;
    }

    SG_DEBUG( "initializing MindyGramKernel optimization\n");
    for (int32_t i = 0; i < n; i++) {
        if ( (i % (n / 10 + 1)) == 0)
            SG_PROGRESS(i, 0, n);

        /* Call add to normal */
        add_to_normal(is[i], ws[i]);
    }
    SG_PRINT( "Done.         \n");

    set_is_initialized(true);
    return true;
}

/**
 * Delete optimization
 */
bool CMindyGramKernel::delete_optimization()
{
    SG_DEBUG( "deleting MindyGramKernel optimization\n");
    clear_normal();
    return true;
}

/**
 * Compute optimized kernel function (dot-product with normal vector)
 * @param i Index of feature vector on the right-hand side
 * @return kernel value for normal and feature vector
 */
float64_t CMindyGramKernel::compute_optimized(int32_t i)
{
    if (!get_is_initialized()) {
        SG_ERROR( "MindyGramKernel optimization not initialized\n");
        return -CMath::INFTY;
    }

    CMindyGramFeatures *rm = (CMindyGramFeatures *) rhs;
    float64_t result = gram_cmp(kernel, rm->get_feature_vector(i), normal);

	return normalizer->normalize_rhs(result, i);
}
#endif
