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
 * Authors:
 */

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalgrefactor/CPUBackend.h>
#include <shogun/mathematics/linalgrefactor/GPUBackend.h>
#include <shogun/mathematics/linalgrefactor/CPUVector.h>
#include <shogun/mathematics/linalgrefactor/GPU_Vector.h>
#include <memory>

#ifndef LINALGR_H__
#define LINALGR_H__

namespace shogun
{
/** Linalg Class **/
class Linalg
{
    CPUBackend* cpubackend;
    GPUBackend* gpubackend;

public:
    Linalg();

    Linalg(CPUBackend* cpu_backend);

    Linalg(GPUBackend* gpu_backend);

    Linalg(CPUBackend* cpu_backend, GPUBackend* gpu_backend);

    void set_cpu_backend(const CPUBackend* cpu_backend);

    CPUBackend* get_cpu_backend();

    void set_gpu_backend(const GPUBackend* gpu_backend);

    GPUBackend* get_gpu_backend();

    template <class T>
    T dot(BaseVector<T>* a, BaseVector<T>* b);

    bool hasGPUBackend();

    //virtual const char* get_name() const { return "Linalg"; }
};

}

namespace shogun
{
    extern std::unique_ptr<Linalg> sg_linalg;
}
#endif
