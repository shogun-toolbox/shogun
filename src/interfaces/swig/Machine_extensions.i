/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben, Heiko Strathmann, Sergey Lisitsyn
 */

// have to be declared after transformers
%shared_ptr(shogun::PipelineBuilder)
%shared_ptr(shogun::Pipeline)
%include "shogun/machine/Pipeline.h"

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

APPLY_MULTICLASS(shogun::MulticlassMachine);
APPLY_MULTICLASS(shogun::KernelMulticlassMachine);
APPLY_MULTICLASS(shogun::LinearMulticlassMachine);
APPLY_MULTICLASS(shogun::DistanceMachine);

APPLY_BINARY(shogun::LinearMachine);
APPLY_BINARY(shogun::KernelMachine);
#ifdef USE_GPL_SHOGUN
APPLY_BINARY(shogun::WDSVMOcas);
#endif //USE_GPL_SHOGUN
APPLY_BINARY(shogun::PluginEstimate);
#ifdef USE_GPL_SHOGUN
APPLY_BINARY(shogun::GaussianProcessClassification);
#endif //USE_GPL_SHOGUN

#if USE_SVMLIGHT
APPLY_REGRESSION(shogun::SVRLight);
#endif //USE_SVMLIGHT
APPLY_REGRESSION(shogun::MKLRegression);
#ifdef HAVE_LAPACK
APPLY_REGRESSION(shogun::LeastSquaresRegression);
APPLY_REGRESSION(shogun::LeastAngleRegression);
#endif
#ifdef USE_GPL_SHOGUN
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

%rename(apply_generic) Machine::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) MulticlassMachine::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) KernelMulticlassMachine::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) LinearMulticlassMachine::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) DistanceMachineMachine::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) LinearMachine::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) KernelMachine::apply(std::shared_ptr<Features> data=NULL);
#ifdef USE_GPL_SHOGUN
%rename(apply_generic) WDSVMOcas::apply(std::shared_ptr<Features> data=NULL);
#endif //USE_GPL_SHOGUN
%rename(apply_generic) PluginEstimate::apply(std::shared_ptr<Features> data=NULL);
#ifdef USE_SVMLIGHT
%rename(apply_generic) SVRLight::apply(std::shared_ptr<Features> data=NULL);
#endif //USE_SVMLIGHT
%rename(apply_generic) MKLRegression::apply(std::shared_ptr<Features> data=NULL);
#ifdef HAVE_LAPACK
%rename(apply_generic) LeastSquaresRegression::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) LeastAngleRegression::apply(std::shared_ptr<Features> data=NULL);
#endif
%rename(apply_generic) GaussianProcessRegression::apply(std::shared_ptr<Features> data=NULL);

%rename(apply_generic) StructuredOutputMachine::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) LinearStructuredOutputMachine::apply(std::shared_ptr<Features> data=NULL);
%rename(apply_generic) KernelStructuredOutputMachine::apply(std::shared_ptr<Features> data=NULL);
#ifdef USE_MOSEK
%rename(apply_generic) CPrimalMosekSOSVM::apply(std::shared_ptr<Features> data=NULL);
#endif

#undef APPLY_MULTICLASS
#undef APPLY_BINARY
#undef APPLY_REGRESSION
#undef APPLY_STRUCTURED
#undef APPLY_LATENT

#endif

/** Instantiate RandomMixin */
%template(RandomMixinMachine) shogun::RandomMixin<shogun::Machine, std::mt19937_64>;

