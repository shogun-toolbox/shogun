/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben, Heiko Strathmann, Sergey Lisitsyn
 */

// have to be declared after transformers
%shared_ptr(shogun::PipelineBuilder)
%shared_ptr(shogun::Pipeline)
%shared_ptr(shogun::Composite)
%include "shogun/machine/Pipeline.h"
%include "shogun/machine/Composite.h"

#if defined(SWIGPYTHON) || defined(SWIGOCTAVE) || defined(SWIGRUBY) || defined(SWIGLUA) || defined(SWIGR)

%define APPLY_MULTICLASS(CLASS)
    %extend CLASS
    {
        std::shared_ptr<MulticlassLabels> apply(std::shared_ptr<Features> data=NULL)
        {
            return $self->apply_multiclass(data);
        }
    }
%enddef

%define APPLY_BINARY(CLASS)
    %extend CLASS
    {
        std::shared_ptr<BinaryLabels> apply(std::shared_ptr<Features> data=NULL)
        {
            return $self->apply_binary(data);
        }
    }
%enddef

%define APPLY_REGRESSION(CLASS)
    %extend CLASS
    {
        std::shared_ptr<RegressionLabels> apply(std::shared_ptr<Features> data=NULL)
        {
            return $self->apply_regression(data);
        }
    }
%enddef

%define APPLY_STRUCTURED(CLASS)
    %extend CLASS
    {
        std::shared_ptr<StructuredLabels> apply(std::shared_ptr<Features> data=NULL)
        {
            return $self->apply_structured(data);
        }
    }
%enddef

%define APPLY_LATENT(CLASS)
    %extend CLASS
    {
        std::shared_ptr<LatentLabels> apply(std::shared_ptr<Features> data=NULL)
        {
            return $self->apply_latent(data);
        }
    }
%enddef

#ifdef USE_GPL_SHOGUN
APPLY_BINARY(shogun::GaussianProcessClassification);
APPLY_REGRESSION(shogun::GaussianProcessRegression);
#endif //USE_GPL_SHOGUN

APPLY_STRUCTURED(shogun::StructuredOutputMachine);
APPLY_STRUCTURED(shogun::LinearStructuredOutputMachine);
APPLY_STRUCTURED(shogun::KernelStructuredOutputMachine);
#ifdef USE_MOSEK
APPLY_STRUCTURED(shogun::PrimalMosekSOSVM);
#endif
#ifdef USE_GPL_SHOGUN
APPLY_STRUCTURED(shogun::DualLibQPBMSOSVM);
APPLY_LATENT(shogun::LatentSVM);
#endif //USE_GPL_SHOGUN

#undef APPLY_MULTICLASS
#undef APPLY_BINARY
#undef APPLY_REGRESSION
#undef APPLY_STRUCTURED
#undef APPLY_LATENT

#endif

