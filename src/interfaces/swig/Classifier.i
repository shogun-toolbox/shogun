/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Saloni Nigam, Sergey Lisitsyn
 */

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
%feature("autodoc", "get_support_vectors(self) -> [] of int") get_support_vectors;
%feature("autodoc", "get_alphas(self) -> [] of float") get_alphas;
#endif

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::DirectorLinearMachine;
%feature("director") shogun::DirectorKernelMachine;
%feature("director:except") {
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
#endif

/* Remove C Prefix */
%shared_ptr(shogun::Machine)
%shared_ptr(shogun::KernelMachine)
%shared_ptr(shogun::LinearMachine)
%shared_ptr(shogun::SVM)
%shared_ptr(shogun::DirectorLinearMachine)
%shared_ptr(shogun::DirectorKernelMachine)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/Machine.h>
%include <shogun/machine/KernelMachine.h>
%include <shogun/machine/LinearMachine.h>
%include <shogun/classifier/svm/SVM.h>
%include <shogun/machine/DirectorLinearMachine.h>
%include <shogun/machine/DirectorKernelMachine.h>
