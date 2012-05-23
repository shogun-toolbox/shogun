/*%warnfilter(302) apply;
%warnfilter(302) apply_generic;*/
#if defined(SWIGPYTHON) || defined(SWIGOCTAVE) || defined(SWIGRUBY) || defined(SWIGLUA) || defined(SWIGR)
%rename(apply_generic) shogun::CMachine::apply();
%rename(apply_generic) shogun::CMulticlassMachine::apply();
%rename(apply_generic) shogun::CKernelMulticlassMachine::apply();
%rename(apply_generic) shogun::CLinearMulticlassMachine::apply();
%rename(apply_generic) shogun::CCDistanceMachineMachine::apply();
%rename(apply_generic) shogun::CLinearMachine::apply();
%rename(apply_generic) shogun::CKernelMachine::apply();
%rename(apply_generic) shogun::CWDSVMOcas::apply();
%rename(apply_generic) shogun::CPluginEstimate::apply();
%rename(apply_generic) shogun::CKernelRidgeRegression::apply();
%rename(apply_generic) shogun::CSVRLight::apply();
%rename(apply_generic) shogun::CMKLRegression::apply();
%rename(apply_generic) shogun::CKernelRidgeRegression::apply();
%rename(apply_generic) shogun::CLinearRidgeRegression::apply();
%rename(apply_generic) shogun::CLeastSquaresRegression::apply();
%rename(apply_generic) shogun::CLeastAngleRegression::apply();
%rename(apply_generic) shogun::CGaussianProcessRegression::apply();
%rename(apply_generic) shogun::CConjugateIndex::apply();

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
APPLY_REGRESSION(CSVRLight);
APPLY_REGRESSION(CMKLRegression);
APPLY_REGRESSION(CKernelRidgeRegression);
APPLY_REGRESSION(CLinearRidgeRegression);
APPLY_REGRESSION(CLeastSquaresRegression);
APPLY_REGRESSION(CLeastAngleRegression);
APPLY_REGRESSION(CGaussianProcessRegression);
}

#undef APPLY_MULTICLASS
#undef APPLY_BINARY
#undef APPLY_REGRESSION
#endif
