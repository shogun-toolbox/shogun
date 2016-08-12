## Internal linear algebra library

#### Table of Contents

- [Motivation](#motivation)
- [For SHOGUN developers](#For SHOGUN developers)
  - [Setting `linalg` backend] (#Setting `linalg` backend)
  - [Using `linalg` operations] (#Using `linalg` operations)
  - [Examples] (#Examples)
- [For `linalg` developers] (#For `linalg` developers)
  - [Understanding operation interface `LinalgNameSpace.h`] (#Understanding operation interface `LinalgNameSpace.h`)
  - [Understanding backend interfaces] (#Understanding backend interfaces)
  - [Understanding operation implementations of different backends] (#Understanding operation implementations of different backends)
  - [Extend external libraries] (#Extend external libraries)

### Motivation

Linear algebra operations form the backbone for most of the computation components in any Machine Learning library. However, writing all of the required linear algebra operations from scratch is rather redundant and undesired, especially when we have some excellent open source alternatives. In Shogun, we prefer

- [`Eigen3`](http://eigen.tuxfamily.org/index.php?title=Main_Page) for its speed and simplicity at the usage level,
- [`ViennaCL`](http://viennacl.sourceforge.net/) version 1.5 for GPU powered linear algebra operations.

For Shogun maintainers, however, the usage of different external libraries for different operations can lead to a painful task.

- For example, consider some part of an algorithm originally written using `Eigen3` API. But a Shogun user wishes to use `ViennaCL` for that algorithm instead, hoping to obtain boosted performance utilizing a GPU powered platform. There is no way of doing that without having the algorithm _rewritten_ by the developers using `ViennaCL`, which leads to _duplication_ of code and effort.
- Also, there is no way to do a _performance comparison_ for the developers while using _different_ external linear algebra libraries for the _same_ algorithm in Shogun code.
- It is also somewhat frustrating for a _new_ developer who has to invest significant amount of time and effort to learn each of these external APIs _just_ to add a new algorithm in Shogun.


### Features of internal linear algebra library

Shogun's **internal linear algebra library** (will be referred as `linalg` hereinafter) is a work-in-progress attempt to overcome these issues. We designed `linalg` as a modularized internal **header only** library in order to

- provide a uniform API for Shogun developers to choose any supported backend without having to worry about the syntactical differences in the external libraries' operations,
- have the backend set for each operations at compile-time (for lesser runtime overhead) and therefore intended to be used internally by Shogun developers,
- allow Shogun developers to add new linear algebra backend plug-ins easily.

### For Shogun developers
#### Setting `linalg` backend
Users can switch between `linalg` backends via global variable `sg_linalg`.
- Shogun uses `Eigen3` backend as default linear algebra backend.
- Enabling of GPU backend allows the data transfer between CPU and GPU, as well as the operations on GPU. `ViennaCL`(GPU) backend can be enabled by assigning new `ViennaCL` backend class to `sg_linalg` or canceled by:
```
   sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());
   sg_linalg->set_gpu_backend(nullptr);
```

- Though backends can be extended, only one CPU backend and one GPU backend are allowed to be registered each time. 

#### Using `linalg` operations
`linalg` library works for both `SGVectors` and `SGMatrices`. The operations can be called by:

```
#include <shogun/mathematics/linalg/LinalgNamespace.h>
shogun::linalg::operation(args)
```

- To use `linalg` operations on GPU data (vectors or matrices) and transfer data between GPU, one can call `to_gpu` and `from_gpu` methods. The methods return results as new instances.

  ```
  auto result = linalg::to_gpu(arg)
  auto result = linalg::from_gpu(arg_on_gpu)
  ```
- The `to_gpu` method will return the original CPU vector or matrix if no GPU backend is available. The `from_gpu` method will return the input argument if it is already on CPU and raise error if no GPU backend is available anymore.

- The status of data can be checked by: `data.on_gpu()`. `True` means the data is on GPU and `false` means the data is on CPU.

- The operations will be carried out on GPU __only if__ the data passed to the operations are on GPU __and__ GPU backend is registered: `sg_linalg->get_gpu_backend() == true`. The `linalg` will be conducted on CPU if the data is on CPU.

- `linalg` will report errors if the data is on GPU but no GPU backend is available anymore. Errors will also occur when an operation requires multiple inputs but the inputs are not on the same backend. 

- A warning will be generated if an operation is not available on specific backend.

#### Examples

 Here we show how to do vector dot with `linalg` library operations on CPU and GPU.
 
```
// CPU dot operation

#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namesapce shogun;

// Create SGVectors
const index_t size = 3;
SGVector<int32_t> a(size), b(size);
a.range_fill(0);
b.range_fill(0);

auto result = linalg::dot(a, b);
```

```
// GPU dot operation

#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>

using namesapce shogun;

// Set gpu backend
sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

// Create SGVectors
const index_t size = 3;
SGVector<int32_t> a(size), b(size), a_gpu, b_gpu;
a.range_fill(0);
b.range_fill(0);

// Transfer vectors to GPU
a_gpu = linalg::to_gpu(a);
b_gpu = linalg::to_gpu(b);

// run dot operation
auto result = linalg::dot(a_gpu, b_gpu);`
```
If the result is a vector or matrix, it needs to be transferred back
```
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>

using namesapce shogun;

// set gpu backend
sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

// Create a SGVector
SGVector<float32_t> a(5), a_gpu;
a.range_fill(0);

// Transfer the vector to gpu
a_gpu = linalg::to_gpu(a);

// Run sacle operation and transfer the result back to CPU
auto result_gpu = linalg::scale(a_gpu, 0.3);
auto result = linalg::from_gpu(result_gpu);
```


### For `linalg` developers
The structure of `linalg` consists of three groups of components:
- The interface that decides which backend to use for each operation (`LinalgNameSpace.h`)
- The structure serves as interface of backend libraries (`GPUMemory*.h`)
- The operation implementations in each backend (`LinalgBackend*.h`). 

#### Understanding operation interface `LinalgNameSpace.h`

- `LinalgNameSpace.h` defines multiple `linalg` operation interfaces in namespace `linalg`. All operation methods will call `infer_backend()` method on the inputs, and decide the backend to call.

#### Understanding backend interfaces

- `GPUMemoryBase` class is a generic base class serving as GPU memory library interface.
The GPU data is referred as `GPUMemoryBase` pointer once it is generated by `to_GPU()` method, and is cast back to specific GPU memory type during operations.

- `GPUMemoryViennaCL` is `ViennaCL` specific GPU memory library interface, which defines the operations to access and manipulate data on GPU with `ViennaCL` operations.

#### Understanding operation implementations of different backends

- `LinalgBackendBase` is the base class for operations on all different backends. The macros in `LinalgBackendBase` class defined the `linalg` operations and data transfer operations available in at least one backend.

- `LinalgBackendGPUBase` has two pure virtual methods: `to_gpu()` and `from_gpu()`. `LinalgBackendViennaCL` and other user-defined GPU backend classes are required to be derived from `LinalgBackendGPUBase` class, and thus GPU transfer methods are required to be implemented.

- `LinalgBackendEigen` and `LinalgBackendViennaCL*` classes provide the specific implementations of linear algebra operations with `Eigen3` library and `ViennaCL` library. 

#### Extend external libraries 

Current `linalg` framework allows easy addition of external linear algebra libraries. To add CPU-based algebra libraries, users just need to derive from `LinalgBackendBase` and re-implement the methods with new library. For GPU-based libraries, users need to add new class derived from `LinalgBackendGPUBase`, as well as the GPU memory library interface class derived from 'GPUMemoryBase` class.