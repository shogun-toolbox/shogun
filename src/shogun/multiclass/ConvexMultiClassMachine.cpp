
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Mohamed AlRefaie
 * Copyright (C) 2012 Mohamed AlRefaie
 */

#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/multiclass/ConvexMultiClassMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/v_array.h>
#include <shogun/base/SGObject.h>
#include "ConvexMultiClassMachine.h"
#include "mathematics/lapack.h"
#include <shogun/machine/LinearMulticlassMachine.h>

using namespace shogun;

CConvexMultiClassMachine::CConvexMultiClassMachine() :
        CLinearMulticlassMachine()
    {
       init_defaults();
    }

void CConvexMultiClassMachine::init_defaults()
{
    setInit_epsilon(1e-2);
    setIterations(10000);
}

CConvexMultiClassMachine::CConvexMultiClassMachine(CDotFeatures* features, CLabels* labs,
        TRAINING_METHOD method, SGVector<float64_t>* gammas,SGMatrix<float64_t>* d_ini,
    float64_t init_epsilon, CKernel* kernel, int32_t iterations, int32_t* task_indexes):
        CLinearMulticlassMachine(ONE_VS_ONE_STRATEGY, features, NULL, labs), training_method(method)
    {
        set_features((CDotFeatures*)features);
        set_labels((CLabels*)labs);
        setInit_epsilon(init_epsilon);

        setGammas((SGVector<float64_t>*)gammas);
        setD_ini((SGMatrix<float64_t>*)d_ini);
        setKernel((CKernel*)kernel);
        setIterations(iterations);
        setTask_indexes(task_indexes);
    }
//void CConvexMultiClassMachine::register_parameters()
//{
//    m_parameters->add(&m_epsilon_init, "m_epsilon_init", "initial value of epsilon");
//    m_parameters->add(&m_method, "m_method", "type of learning method, either feat, diagonal or independent");
//    m_parameters->add(&m_gammas, "m_gammas", "regularization parameter");
//    m_parameters->add(&m_d_ini, "m_d_ini", "initial matrix d");
//    m_parameters->add(&m_kernel, "m_kernel", "kernel used for training");
//    m_parameters->add(&m_iterations, "m_iterations", "No. of iterations");
//    m_parameters->add(&m_task_indexes, "m_task_indexes", "task_indexes array");
//}   




//void CConvexMultiClassMachine::eval_d(SGMatrix<float64_t>* matrix)
//{
////    //http://software.intel.com/sites/products/documentation/hpc/mkl/lapack/mkl_lapack_examples/dgesvd_ex.c.htm
////    //http://www.netlib.org/lapack/double/dgesvd.f
////    //s matrix
////    //M = number of rows
////    //N = number of columns
////    //LDA = M
////    //LDU =M
////    //LDVT = N
//    float64_t s_matrix=SG_MALLOC(float64_t, matrix->num_cols);
//    float64_t u_matrix=SG_MALLOC(float64_t, (matrix->num_rows*matrix->num_rows));
//    float64_t vt_matrix=SG_MALLOC(float64_t, (matrix->num_cols*matrix->num_cols));
//    int32_t info;
//    wrap_dgesvd('A', 'A', matrix.num_rows, matrix.num_cols, matrix.matrix, matrix.num_rows, s_matrix, u_matrix, matrix.num_rows, vt_matrix, &info);
//       
//}

void CConvexMultiClassMachine::f_method(float64_t* vec, float64_t* new_vec)
{
    int32_t size=(sizeof vec/sizeof(float64_t));
    //float64_t* new_vec=SG_MALLOC(float64_t, size);
    for (int32_t i=0;i<size;i++)
    {
        //assuming eps=2^-52
        //TODO check the accuracy of this value against float64_t
        if (vec[i]>(2^(-52)))
        {
            new_vec[i]=1/vec[i];
        }
    }
}

void CConvexMultiClassMachine::d_method(float64_t* vec, float64_t* new_vec)
{
    int32_t size=(sizeof vec/sizeof(float64_t));
    float64_t sum_value=CMath::sum(vec, size);
    for (int32_t i=0;i<size;i++)
    {
        new_vec[i]=vec[i]/sum_value;
    }
}





#endif /* HAVE_LAPACK */
