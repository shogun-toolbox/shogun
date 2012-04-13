/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *
 * Copyright (C) 2012 Michal Uricar, uricamic@cmp.felk.cvut.cz
 */

#ifndef _GSKERNELRIDGEREGRESSION_H__
#define _GSKERNELRIDGEREGRESSION_H__

#include <shogun/lib/config.h>
#include <shogun/regression/Regression.h>

#ifdef HAVE_LAPACK

#include <shogun/machine/KernelMachine.h>

namespace shogun
{
class CGSKernelRidgeRegression :public CKernelMachine
{
    public:
        /** default constructor */
        CGSKernelRidgeRegression();

        /** constructor
          *
          * @param tau regularization constant tau
          * @param k kernel
          * @param lab labels
          */
        CGSKernelRidgeRegression(float64_t tau, CKernel* k, CLabels* lab);
        virtual ~CGSKernelRidgeRegression() {}

        /** set regularization constant
         *
         * @param tau new tau
         */
        inline void set_tau(float64_t tau) { m_tau = tau; }

        /** set regularization constant
         *
         * @param tau new tau
         */
        inline void set_epsilon(float64_t epsilon) { m_epsilon = epsilon; }

        /** load regression from file
         *
         * @param srcfile file to load from
         * @return if loading was successful
         */
        virtual bool load(FILE* srcfile);

        /** save regression to file
         *
         * @param dstfile file to save to
         * @return if saving was successful
         */
        virtual bool save(FILE* dstfile);

        /** get classifier type
         *
         * @return classifier type KernelRidgeRegression
         */
        inline virtual EClassifierType get_classifier_type()
        {
            return CT_GSKERNELRIDGEREGRESSION;
        }

        /** @return object name */
        inline virtual const char* get_name() const { return "GSKernelRidgeRegression"; }

    protected:
        /** train regression
         *
         * @param data training data (parameter can be avoided if distance or
         * kernel-based regressors are used and distance/kernels are
         * initialized with train data)
         *
         * @return whether training was successful
         */
        virtual bool train_machine(CFeatures* data=NULL);

    private:
        void init();

    private:
        /** regularization parameter tau */
        float64_t m_tau;

        /** epsilon constant */
        float64_t m_epsilon;
};
}

#endif // HAVE_LAPACK
#endif // _GSKERNELRIDGEREGRESSION_H__
