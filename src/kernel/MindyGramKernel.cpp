/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
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
 * @param measure Similarity measure to use
 * @param param String of mindy parameters
 * @param n Normalization type
 */
CMindyGramKernel::CMindyGramKernel(LONG cache, CHAR *measure, CHAR *param, ENormalizationType n) : CKernel(cache)
{
    /* Init attributes */
    sdiag_lhs = NULL;
    sdiag_rhs = NULL;
    initialized = false;
    norm = n;

    /* Create similarity measure */
    SG_INFO( "Initializing Mindy kernel\n");
    kernel = sm_create(sm_get_type(measure));
   
    SG_INFO( "Mindy similarity measure: %s\n", 
	         sm_get_descr(kernel->type));

    /* Parse and set parameters */
    parse_params(param);

    /* Display paramater list */
    for (INT i = 0; p_map[i].name; i++) {
        if (p_map[i].idx != SP_DIST)
            SG_INFO( "Param %8s=%8.6f\t %s\n", 
			p_map[i].name, p_map[i].val, p_map[i].descr);
        else
            SG_INFO( "Param %8s=%s\t %s\n", p_map[i].name, 
                        sm_get_name((sm_type_t) p_map[i].val), 
                        p_map[i].descr);
    }

    /* Initialize optimization */
    if (kernel->type == ST_LINEAR) {
	SG_INFO( "Optimization supported\n");
        properties |= KP_LINADD;
    }
    
    normal = NULL;
    clear_normal();
}


/**
 * Mindy kernel destructor
 */
CMindyGramKernel::~CMindyGramKernel()
{
    cleanup();
    sm_destroy(kernel);
}

/**
 * Parse provided parameters
 */
void CMindyGramKernel::parse_params(CHAR *pa)
{
    INT i;
    CHAR *t, *p;

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
    remove_lhs();
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

    if (sdiag_lhs != sdiag_rhs)
        delete[] sdiag_rhs;
    delete[] sdiag_lhs;

    lhs = NULL ;
    rhs = NULL ;
    initialized = false;
    sdiag_lhs = NULL;
    sdiag_rhs = NULL;
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
 * @param do_init Flag to force initialization
 * @return true on success, false otherwise
 */
bool CMindyGramKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{

    SG_DEBUG( "Initializing MindyGramKernel %p %p\n", l, r);
    /* Call constructor of super class */
    bool result = CKernel::init(l,r,do_init);

    initialized = false;
    INT i;

    /* Assert correct types of features */
    ASSERT(l->get_feature_class() == C_MINDYGRAM);
    ASSERT(r->get_feature_class() == C_MINDYGRAM);
    ASSERT(l->get_feature_type() == F_ULONG);
    ASSERT(r->get_feature_type() == F_ULONG);

    /* Clean diagonals */
    if (sdiag_lhs != sdiag_rhs)
        delete[] sdiag_rhs;
    sdiag_rhs=NULL;
    delete[] sdiag_lhs;
    sdiag_lhs=NULL;

    /* Initialize left normalization diagonal */
    sdiag_lhs= new DREAL[lhs->get_num_vectors()];
    for (i = 0; i < lhs->get_num_vectors(); i++)
        sdiag_lhs[i]=1;

    /* Initialize (or copy) right normalization diagonal */
    if (l == r) {
        sdiag_rhs=sdiag_lhs;
    }
    else {
        sdiag_rhs= new DREAL[rhs->get_num_vectors()];
        for (i = 0; i<rhs->get_num_vectors(); i++)
            sdiag_rhs[i]=1;
    }

    ASSERT(sdiag_lhs);
    ASSERT(sdiag_rhs);

    this->lhs=(CMindyGramFeatures *) l;
    this->rhs=(CMindyGramFeatures *) l;

    /* Compute left normalization diagonal */
    for (i = 0; i<lhs->get_num_vectors(); i++) {
        sdiag_lhs[i] = sqrt(compute(i,i));

        /* trap divide by zero exception */
        if (sdiag_lhs[i] == 0)
            sdiag_lhs[i] = 1e-16;
    }

    /*  Skip if rhs computation if necessary */
    if (sdiag_lhs == sdiag_rhs)
        goto skip_rhs;

    this->lhs=(CMindyGramFeatures *) r;
    this->rhs=(CMindyGramFeatures *) r;

    /* Compute right normalization diagonal */
    for (i=0; i<rhs->get_num_vectors(); i++) {
        sdiag_rhs[i] = sqrt(compute(i,i));

        /* trap divide by zero exception */
        if (sdiag_rhs[i]==0)
            sdiag_rhs[i]=1e-16;
    }

    skip_rhs:
    /* Reset feature pointers */
    this->lhs=(CStringFeatures<WORD>*) l;
    this->rhs=(CStringFeatures<WORD>*) r;

    initialized = true;
    return result;
}

/**
 * Compute kernel value for the given pair of feature vectors
 * @param i Index to lhs vector
 * @param j Index to rhs vector
 * @param kernel value
 */
DREAL CMindyGramKernel::compute(INT i, INT j)
{
    /* Cast things to mindy gram features */
    CMindyGramFeatures *lm = (CMindyGramFeatures *) lhs;
    CMindyGramFeatures *rm = (CMindyGramFeatures *) rhs;

    /* Call (internal) mindy comparison function */
    DREAL result = gram_cmp(kernel, lm->get_feature_vector(i),
        rm->get_feature_vector(j));

    /* Sloppy and incorrect distance to kernel conversion ;) */    
    if (sm_get_class(kernel->type) == SC_DIST)
        result = 1 / (result + 1);

    if (!initialized)
        return result;

    /* Normalize result */
    switch (norm) {
        case NO_NORMALIZATION:
            return result;
        case SQRT_NORMALIZATION:
            return result/sqrt(sdiag_lhs[i]*sdiag_rhs[i]);
        case FULL_NORMALIZATION:
            return result/(sdiag_lhs[i]*sdiag_rhs[j]);
        default:
            SG_ERROR( "Unknown Normalization in use!\n");
            return -CMath::INFTY;
    }
}

/**
 * Add a featrue vector to a global normal vector
 * @param i Index of feature vector
 * @param w Weight for addition (usually alpha_i)
 */
void CMindyGramKernel::add_to_normal(INT i, DREAL w)
{
    /* Initialize normal vector if necessary */
    if (!normal)
        normal = gram_create();

    /* Add indexed vector to normal */
    CMindyGramFeatures *lm = (CMindyGramFeatures *) lhs;
    gram_t *new_normal = gram_add(normal, lm->get_feature_vector(i),
        normalize_weight(w, i, norm));

    /* Destroy old normal and exchange pointers */
    gram_destroy(normal);
    normal = new_normal;

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
bool CMindyGramKernel::init_optimization(INT n, INT *is, DREAL * ws)
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
    for (int i = 0; i < n; i++) {
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
DREAL CMindyGramKernel::compute_optimized(INT i)
{
    if (!get_is_initialized()) {
        SG_ERROR( "MindyGramKernel optimization not initialized\n");
        return -CMath::INFTY;
    }

    CMindyGramFeatures *rm = (CMindyGramFeatures *) rhs;
    DREAL result = gram_cmp(kernel, rm->get_feature_vector(i), normal);

    switch (norm) {
        case NO_NORMALIZATION:
            return result;
        case SQRT_NORMALIZATION:
            return result/sqrt(sdiag_rhs[i]);
        case FULL_NORMALIZATION:
            return result/sdiag_rhs[i];
        default:
            SG_ERROR( "Unknown Normalization in use!\n");
            return -CMath::INFTY;
    }
}

bool CMindyGramKernel::load_init(FILE* src)
{
    return false;
}

bool CMindyGramKernel::save_init(FILE* dest)
{
    return false;
}
#endif
