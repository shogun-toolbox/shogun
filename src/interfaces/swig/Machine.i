/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
 */

/*%warnfilter(302) apply;
%warnfilter(302) apply_generic;*/

%shared_ptr(shogun::Machine)
%shared_ptr(shogun::PipelineBuilder)
%shared_ptr(shogun::Pipeline)
%shared_ptr(shogun::LinearMachine)
%shared_ptr(shogun::DistanceMachine)
%shared_ptr(shogun::IterativeMachine<LinearMachine>)

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
        std::shared_ptr<CLatentLabels> apply(std::shared_ptr<Features> data=NULL)
        {
            return $self->apply_latent(data);
        }
    }
%enddef

APPLY_MULTICLASS(MulticlassMachine);
APPLY_MULTICLASS(KernelMulticlassMachine);
APPLY_MULTICLASS(LinearMulticlassMachine);
APPLY_MULTICLASS(DistanceMachine);

APPLY_BINARY(LinearMachine);
APPLY_BINARY(KernelMachine);
#ifdef USE_GPL_SHOGUN
APPLY_BINARY(WDSVMOcas);
#endif //USE_GPL_SHOGUN
APPLY_BINARY(PluginEstimate);
#ifdef USE_GPL_SHOGUN
APPLY_BINARY(GaussianProcessClassification);
#endif //USE_GPL_SHOGUN

APPLY_REGRESSION(MKLRegression);
#ifdef HAVE_LAPACK
APPLY_REGRESSION(LeastSquaresRegression);
APPLY_REGRESSION(LeastAngleRegression);
#endif
#ifdef USE_GPL_SHOGUN
APPLY_REGRESSION(GaussianProcessRegression);
#endif //USE_GPL_SHOGUN

APPLY_STRUCTURED(StructuredOutputMachine);
APPLY_STRUCTURED(LinearStructuredOutputMachine);
APPLY_STRUCTURED(KernelStructuredOutputMachine);
#ifdef USE_MOSEK
APPLY_STRUCTURED(PrimalMosekSOSVM);
#endif
#ifdef USE_GPL_SHOGUN
APPLY_STRUCTURED(DualLibQPBMSOSVM);
APPLY_LATENT(LatentSVM);
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

%include <shogun/machine/Machine.h>

/** Instantiate RandomMixin */
%template(SeedableMachine) shogun::Seedable<shogun::CMachine>;
%template(RandomMixinMachine) shogun::RandomMixin<shogun::CMachine, std::mt19937_64>;

%include <shogun/machine/Pipeline.h>
