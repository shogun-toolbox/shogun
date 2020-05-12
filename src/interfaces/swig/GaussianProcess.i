/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%shared_ptr(shogun::MeanFunction)
%shared_ptr(shogun::Inference)
SHARED_RANDOM_INTERFACE(shogun::Inference)
%shared_ptr(shogun::LikelihoodModel)
SHARED_RANDOM_INTERFACE(shogun::LikelihoodModel)
%shared_ptr(shogun::GaussianProcess)
SHARED_RANDOM_INTERFACE(shogun::GaussianProcess)


/* These functions return new Objects */
RANDOM_INTERFACE(Machine)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/gp/LikelihoodModel.h>
RANDOM_INTERFACE(LikelihoodModel)

%include <shogun/machine/gp/MeanFunction.h>


%include <shogun/machine/gp/Inference.h>
RANDOM_INTERFACE(Inference)
%include <shogun/machine/GaussianProcess.h>
RANDOM_INTERFACE(GaussianProcess)