/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/SGVector.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distributions/MixtureModel.h>
#include <shogun/distributions/Gaussian.h> 
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_LAPACK

TEST(MixtureModel,gaussian_mixture_model)
{
	sg_rand->set_seed(2);	
	SGMatrix<float64_t> data(1,400);

	for (int32_t i=0;i<100;i++)
		data(0,i)=CMath::randn_double();
	for (int32_t i=100;i<400;i++)
		data(0,i)=CMath::randn_double()+10;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	CDynamicObjectArray* comps=new CDynamicObjectArray();
	SGVector<float64_t> mean1(1);
	mean1[0]=5;
	SGMatrix<float64_t> cov1(1,1);
	cov1(0,0)=5;
	CGaussian* g1=new CGaussian(mean1,cov1,DIAG);

	SGVector<float64_t> mean2(1);
	mean2[0]=4;
	SGMatrix<float64_t> cov2(1,1);
	cov2(0,0)=3;
	CGaussian* g2=new CGaussian(mean2,cov2,DIAG);

	comps->push_back(g1);
	comps->push_back(g2);

	SGVector<float64_t> weights(2);
	weights[0]=0.5;
	weights[1]=0.5;

	CMixtureModel* mix=new CMixtureModel(comps,weights);
	mix->train(feats);

	CDistribution* distr=CDistribution::obtain_from_generic(comps->get_element(0));
	CGaussian* outg=CGaussian::obtain_from_generic(distr);
	SGVector<float64_t> m=outg->get_mean();
	SGMatrix<float64_t> cov=outg->get_cov();

	float64_t eps=1e-8;
	EXPECT_NEAR(m[0],9.863760378,eps);
	EXPECT_NEAR(cov(0,0),0.956568199,eps);	

	SG_FREE(cov.matrix);
	SG_UNREF(outg);
	SG_UNREF(distr);

	distr=CDistribution::obtain_from_generic(comps->get_element(1));
	outg=CGaussian::obtain_from_generic(distr);
	m=outg->get_mean();
	cov=outg->get_cov();

	EXPECT_NEAR(m[0],-0.208122793,eps);
	EXPECT_NEAR(cov(0,0),1.095106568,eps);	

	SG_FREE(cov.matrix);
	SG_UNREF(outg);
	SG_UNREF(distr);
	SG_UNREF(mix)
}

#endif /* HAVE_LAPACK */