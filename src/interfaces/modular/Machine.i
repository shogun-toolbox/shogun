/*%warnfilter(302) apply;
%warnfilter(302) apply_generic;*/
#if defined(SWIGPYTHON) || defined(SWIGOCTAVE) || defined(SWIGRUBY) || defined(SWIGLUA) || defined(SWIGR)

%define APPLY_MULTICLASS(CLASS)
    %extend CLASS
    {
        CMulticlassLabels* apply(CFeatures* data=NULL)
        {
            return $self->apply_multiclass(data);
        }
    }
%enddef

%define APPLY_BINARY(CLASS)
    %extend CLASS
    {
        CBinaryLabels* apply(CFeatures* data=NULL)
        {
            return $self->apply_binary(data);
        }
    }
%enddef

%define APPLY_REGRESSION(CLASS)
    %extend CLASS
    {
        CRegressionLabels* apply(CFeatures* data=NULL)
        {
            return $self->apply_regression(data);
        }
    }
%enddef

%define APPLY_STRUCTURED(CLASS)
    %extend CLASS
    {
        CStructuredLabels* apply(CFeatures* data=NULL)
        {
            return $self->apply_structured(data);
        }
    }
%enddef

%define APPLY_LATENT(CLASS)
    %extend CLASS
    {
        CLatentLabels* apply(CFeatures* data=NULL)
        {
            return $self->apply_latent(data);
        }
    }
%enddef

namespace shogun {
APPLY_MULTICLASS(CMulticlassMachine);
APPLY_MULTICLASS(CKernelMulticlassMachine);
APPLY_MULTICLASS(CLinearMulticlassMachine);
APPLY_MULTICLASS(CDistanceMachine);
APPLY_MULTICLASS(CConjugateIndex);

APPLY_BINARY(CLinearMachine);
APPLY_BINARY(CKernelMachine);
APPLY_BINARY(CWDSVMOcas);
APPLY_BINARY(CPluginEstimate);

APPLY_REGRESSION(CKernelRidgeRegression);
APPLY_REGRESSION(CLibSVR);
APPLY_REGRESSION(CSVRLight);
APPLY_REGRESSION(CMKLRegression);
APPLY_REGRESSION(CKernelRidgeRegression);
APPLY_REGRESSION(CLinearRidgeRegression);
APPLY_REGRESSION(CLeastSquaresRegression);
APPLY_REGRESSION(CLeastAngleRegression);
#ifdef HAVE_EIGEN3
APPLY_REGRESSION(CGaussianProcessRegression);
#endif //HAVE_EIGEN3

APPLY_STRUCTURED(CStructuredOutputMachine);
APPLY_STRUCTURED(CLinearStructuredOutputMachine);
APPLY_STRUCTURED(CKernelStructuredOutputMachine);
#ifdef USE_MOSEK
APPLY_STRUCTURED(CPrimalMosekSOSVM);
#endif
APPLY_STRUCTURED(CDualLibQPBMSOSVM);

APPLY_LATENT(CLatentSVM);
}

%rename(apply_generic) CMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CMulticlassMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CKernelMulticlassMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CLinearMulticlassMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CCDistanceMachineMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CLinearMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CKernelMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CWDSVMOcas::apply(CFeatures* data=NULL);
%rename(apply_generic) CPluginEstimate::apply(CFeatures* data=NULL);
%rename(apply_generic) CKernelRidgeRegression::apply(CFeatures* data=NULL);
%rename(apply_generic) CSVRLight::apply(CFeatures* data=NULL);
%rename(apply_generic) CLibSVR::apply(CFeatures* data=NULL);
%rename(apply_generic) CMKLRegression::apply(CFeatures* data=NULL);
%rename(apply_generic) CKernelRidgeRegression::apply(CFeatures* data=NULL);
%rename(apply_generic) CLinearRidgeRegression::apply(CFeatures* data=NULL);
%rename(apply_generic) CLeastSquaresRegression::apply(CFeatures* data=NULL);
%rename(apply_generic) CLeastAngleRegression::apply(CFeatures* data=NULL);
%rename(apply_generic) CGaussianProcessRegression::apply(CFeatures* data=NULL);
%rename(apply_generic) CConjugateIndex::apply(CFeatures* data=NULL);

%rename(apply_generic) CStructuredOutputMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CLinearStructuredOutputMachine::apply(CFeatures* data=NULL);
%rename(apply_generic) CKernelStructuredOutputMachine::apply(CFeatures* data=NULL);
#ifdef USE_MOSEK
%rename(apply_generic) CPrimalMosekSOSVM::apply(CFeatures* data=NULL);
#endif

#undef APPLY_MULTICLASS
#undef APPLY_BINARY
#undef APPLY_REGRESSION
#undef APPLY_STRUCTURED
#undef APPLY_LATENT
#endif
#endif

ARRAY(0x28a8960)