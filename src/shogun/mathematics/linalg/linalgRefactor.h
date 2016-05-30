/*
 * -*- coding: utf-8 -*-
 * vim: set fileencoding=utf-8
 *
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Pan Deng
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <iostream>

using namespace shogun;

template <class T>
struct BaseVector
{
    BaseVector(){}
    BaseVector(const SGVector<T> vector){}

    virtual bool onGPU()
    {
        return false;
    }
};

template <class T>
struct CPU_Vector : public BaseVector<T>
{
    //<SGVector<T>>* CPUptr;
    SGVector<T> vec;

    CPU_Vector(const SGVector<T> vector)
    {
        //CPUptr = &vector;
        vec = vector;
    }

    bool onGPU()
    {
        return false;
    }
};

template <typename T>
struct GPU_Vector : public BaseVector<T>
{
#ifdef HAVE_VIENNACL
    // unique_pointer<VCLMemoryArray> GPUptr;
    // other gpu related stuff
#endif
    bool onGPU()
    {
        return true;
    }
};

class CPUBackend
{
public:
    template <typename T>
    T dot(CPU_Vector<T> a, CPU_Vector<T> b)
    {
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
        Eigen::Map<VectorXt> vec_a = a.vec;
        Eigen::Map<VectorXt> vec_b = b.vec;
        return vec_a.dot(vec_b);
    }

   // similarly, other methods
};

//template <typename T>
class GPUBackend
{
public:
 #ifdef HAVE_VIENNACL
    template <class T>
    T dot(GPU_Vector<T> a, GPU_Vector<T> b)
    {
        // Dereference a.GPUptr and b.GPUptr to vcl_vector?
        // viennacl::linalg::inner_prod(vcl_vector_a, vcl_vector_b);
        // Transfer back to CPU end???
    }
   // similarly, other methods
  #else
   template <class T>
   T dot(GPU_Vector<T> a, GPU_Vector<T> b)
   {
       throw std::runtime_error("user did not register GPU backend");
   }
 #endif
 };

class LinalgRefactor
{
    CPUBackend* cpubackend;
    GPUBackend* gpubackend;

public:
    LinalgRefactor():cpubackend(nullptr), gpubackend(nullptr){}

    LinalgRefactor(CPUBackend* cpu_backend):cpubackend(cpu_backend), gpubackend(nullptr){}

    LinalgRefactor(GPUBackend* gpu_backend):cpubackend(nullptr), gpubackend(gpu_backend){}

    LinalgRefactor(CPUBackend* cpu_backend, GPUBackend* gpu_backend)
    :cpubackend(cpu_backend), gpubackend(gpu_backend){}

    template <class T>
    T dot(BaseVector<T>* a, BaseVector<T>* b)
    {
        if (a->onGPU() && b->onGPU())
        {
            if (this->hasGPUBackend())
            {
                // do the gpu backend dot product
                // you shouldn't care whether it's viennacl or some other GPU backend.
                return this->gpubackend->dot<T>(*static_cast<GPU_Vector<T>*>(a),
                                                *static_cast<GPU_Vector<T>*>(b));
            } else {
                throw std::runtime_error("user did not register GPU backend");
            }
        }
        else {
            // take care that the matricies are on the same backend
            if (a->onGPU()){ }//Transfer back to CPU || throw error ??? }
            else if (b->onGPU()) { }//Transfer back to CPU || throw error }

            // do the non-gpu based default backend:
            // this should be actually as well implemented in a separate class's function and just that being called here:
            // like:
            return this->cpubackend->dot<T>(*static_cast<CPU_Vector<T>*>(a),
                                            *static_cast<CPU_Vector<T>*>(b));
        }
    }

    bool hasGPUBackend()
    {
        return gpubackend != nullptr;
    }
};
