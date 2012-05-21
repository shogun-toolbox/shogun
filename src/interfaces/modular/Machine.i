/*%warnfilter(302) apply;
%warnfilter(302) apply_generic;*/
%rename(apply_generic) apply(CFeatures*);

namespace shogun {
    %extend CMulticlassMachine
    {
        CMulticlassLabels* apply(CFeatures* data=NULL)
        {
            return CMulticlassLabels::obtain_from_generic($self->apply_multiclass(data));
        }
    }

    %extend CKernelMulticlassMachine
    {
        CMulticlassLabels* apply(CFeatures* data=NULL)
        {
            return CMulticlassLabels::obtain_from_generic($self->apply_multiclass(data));
        }
    }
    
    %extend CLinearMulticlassMachine
    {
        CMulticlassLabels* apply(CFeatures* data=NULL)
        {
            return CMulticlassLabels::obtain_from_generic($self->apply_multiclass(data));
        }
    }


    /*%extend COnlineLinearMachine
    {
        CRealLabels* apply(CFeatures* data=NULL)
        {
            return CRealLabels::obtain_from_generic($self->apply_binary(data));
        }
    }*/

    %extend CLinearMachine
    {
        CRegressionLabels* apply(CFeatures* data=NULL)
        {
            return CRegressionLabels::obtain_from_generic($self->apply_regression(data));
        }
    }

    %extend CKernelMachine
    {
        CRegressionLabels* apply(CFeatures* data=NULL)
        {
            return CRegressionLabels::obtain_from_generic($self->apply_regression(data));
        }
    }

    %extend CDistanceMachine
    {
        CMulticlassLabels* apply(CFeatures* data=NULL)
        {
            return CMulticlassLabels::obtain_from_generic($self->apply_multiclass(data));
        }
    }
}


