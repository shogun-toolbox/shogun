#include "lib/common.h"
#include "kernel/WaveletKernel.h"
#include "features/SimpleFeatures.h"

using namespace shogun;

CWaveletKernel::CWaveletKernel() : CDotKernel()
{
	init();
}

CWaveletKernel::CWaveletKernel(int32_t size, float64_t a, float64_t c)
: CDotKernel(size)
{
	init();
	Wdilation=a; //Wavelet dilation coefficient
	Wtranslation=c; //Wavelet translation coefficient
}

CWaveletKernel::CWaveletKernel(
	CDotFeatures* l, CDotFeatures* r, int32_t size, float64_t a, float64_t c)
: CDotKernel(size)
{
	init();
	Wdilation=a;
	Wtranslation=c;

	init(l,r);
}

CWaveletKernel::~CWaveletKernel()
{
	cleanup();
}

void CWaveletKernel::cleanup()
{
}

bool CWaveletKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	return init_normalizer();
}

void CWaveletKernel::init()
{
	Wdilation=0.0;
	Wtranslation=0.0;

	m_parameters->add(&Wdilation, "Wdilation", "Wdil");
	m_parameters->add(&Wtranslation, "Wtranslaton", "Wtrans");
}
float64_t CWaveletKernel::MotherWavelet(float64_t h)
{
	float64_t res=cos(1.75*h)*exp(-h*h/2);
	return res;
}
float64_t CWaveletKernel::compute(int32_t idx_a, int32_t idx_b)
{
        int32_t alen, blen;
        bool afree, bfree;

        float64_t* avec=
                ((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
        float64_t* bvec=
                ((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
        ASSERT(alen==blen);

        float64_t result=1;
	
        for (int32_t i=0; i<alen; i++)
        {
		if (Wtranslation !=0)
		{
                	float64_t h1=(avec[i]-Wdilation)/Wtranslation;
                	float64_t h2=(bvec[i]-Wdilation)/Wtranslation;
			float64_t res1=MotherWavelet(h1);
			float64_t res2=MotherWavelet(h2);
                	result=result*res1*res2;
		}
                        
        }

        ((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
        ((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

        return result;
}

