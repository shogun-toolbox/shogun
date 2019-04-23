/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
 */

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::CDirectorLatentModel;
#endif

/* Remove C Prefix */
%shared_ptr(shogun::LatentModel)

%shared_ptr(shogun::LinearLatentMachine)

%shared_ptr(shogun::LatentSVM)

%shared_ptr(shogun::DirectorLatentModel)


/* Include Class Headers to make them visible from within the target language */
%include <shogun/latent/LatentModel.h>

%include <shogun/latent/DirectorLatentModel.h>

%include <shogun/machine/LinearLatentMachine.h>

#ifdef USE_GPL_SHOGUN
%include <shogun/latent/LatentSVM.h>
#endif //USE_GPL_SHOGUN
