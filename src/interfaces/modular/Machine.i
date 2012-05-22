/*%warnfilter(302) apply;
%warnfilter(302) apply_generic;*/
%rename(apply_generic) shogun::CMachine::apply(CFeatures*);
%rename(apply_generic) shogun::CMulticlassMachine::apply(CFeatures*);
%rename(apply_generic) shogun::CKernelMulticlassMachine::apply(CFeatures*);
%rename(apply_generic) shogun::CLinearMulticlassMachine::apply(CFeatures*);
%rename(apply_generic) shogun::CCDistanceMachineMachine::apply(CFeatures*);
%rename(apply_generic) shogun::CLinearMachine::apply(CFeatures*);
%rename(apply_generic) shogun::CKernelMachine::apply(CFeatures*);
%rename(apply_generic) shogun::CWDSVMOcas::apply(CFeatures*);
%rename(apply_generic) shogun::CPluginEstimate::apply(CFeatures*);
%rename(apply_generic) shogun::CKernelRidgeRegression::apply(CFeatures*);
%rename(apply_generic) shogun::CSVRLight::apply(CFeatures*);
%rename(apply_generic) shogun::CMKLRegression::apply(CFeatures*);
%rename(apply_generic) shogun::CKernelRidgeRegression::apply(CFeatures*);
%rename(apply_generic) shogun::CLinearRidgeRegression::apply(CFeatures*);
%rename(apply_generic) shogun::CLeastSquaresRegression::apply(CFeatures*);
%rename(apply_generic) shogun::CLeastAngleRegression::apply(CFeatures*);
%rename(apply_generic) shogun::CGaussianProcessRegression::apply(CFeatures*);
%rename(apply_generic) shogun::CConjugateIndex::apply(CFeatures*);

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

    /*%extend COnlineLinearMachine
    {
        CRegressionLabels* apply(CFeatures* data=NULL)
        {
            return CRegressionLabels::obtain_from_generic($self->apply_binary(data));
        }
    }*/
