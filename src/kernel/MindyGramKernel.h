/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef HAVE_MINDY

#ifndef _MINDYGRAMKERNEL_H___
#define _MINDYGRAMKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"

/* Parameter specifications */
typedef struct {
    char *name;                 /* Name of parameter */
    int idx;                    /* Index in param array (see sm.h) */
    real_t val;                 /* Default value */
    char *descr;                /* Description */
} param_spec_t;

class CMindyGramKernel: public CKernel
{

    public:
        /* Constructors */
        CMindyGramKernel(LONG ch, CHAR *measure, CHAR *param, ENormalizationType n, LONG c);
        ~CMindyGramKernel();

        /* Init and cleanup functions */
        void parse_params(CHAR *);
        virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
        virtual void cleanup();
        virtual void remove_lhs();
        virtual void remove_rhs();

        /* Identification functions */
        inline virtual EKernelType get_kernel_type() { return K_MINDYGRAM; }
        inline virtual EFeatureType get_feature_type() { return F_ULONG; }
        inline virtual EFeatureClass get_feature_class() { return C_MINDYGRAM; }
        inline virtual const CHAR* get_name() { return "MindyGram"; }

        /* Optimization functions */
        virtual bool init_optimization(INT count, INT *IDX, DREAL * weights);
        virtual bool delete_optimization();
        virtual DREAL compute_optimized(INT idx);
        virtual void add_to_normal(INT idx, DREAL weight);
        virtual void clear_normal();

        /* Load and ysave functions */
        bool load_init(FILE* src);
        bool save_init(FILE* dest);

        /* Normalize function for optimization */
        inline DREAL normalize_weight(DREAL v, INT i, ENormalizationType n) {
            switch (n) {
                case NO_NORMALIZATION:
                    return v;
                case SQRT_NORMALIZATION:
                    return v / sqrt(sdiag_lhs[i]);
                case FULL_NORMALIZATION:
                    return v / sdiag_lhs[i];
                default:
                    ASSERT(0);
            }
            return -CMath::INFTY;
        }

    protected:

        /* Kernel function */
        DREAL compute(INT idx_a, INT idx_b);

    private:

        /* Arrays of kernel matrix diagonals */
        DREAL* sdiag_lhs;
        DREAL* sdiag_rhs;

        /* Initialization flag */
        bool initialized;
        /* Normalization mode */
        ENormalizationType norm;
        /* Kernel function (=> Mindy similarity measure) */
        sm_t *kernel;
        /* Normal vector for optimization */
        gram_t *normal;
        /* MD5 cache size */
        size_t cache;

};
#endif
#endif                                            /* HAVE_MINDY */
