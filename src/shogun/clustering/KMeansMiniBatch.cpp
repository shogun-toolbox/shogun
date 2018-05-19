/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Michele Mazzoni, Heiko Strathmann, Viktor Gal
 */

#include <shogun/clustering/KMeansMiniBatch.h>
#include <shogun/mathematics/Math.h>
#include <shogun/distance/Distance.h>
#include <shogun/base/progress.h>
#include <shogun/features/DenseFeatures.h>

#ifdef _WIN32
#undef far
#undef near
#endif

using namespace shogun;

namespace shogun
{
CKMeansMiniBatch::CKMeansMiniBatch():CKMeansBase()
{
	init_mb_params();
}

CKMeansMiniBatch::CKMeansMiniBatch(int32_t k_i, CDistance* d_i, bool use_kmpp_i):CKMeansBase(k_i, d_i, use_kmpp_i)
{
	init_mb_params();
}

CKMeansMiniBatch::CKMeansMiniBatch(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i):CKMeansBase(k_i, d_i, centers_i)
{
	init_mb_params();
}

CKMeansMiniBatch::~CKMeansMiniBatch()
{
}

void CKMeansMiniBatch::set_batch_size(int32_t b)
{
	REQUIRE(b>0, "Parameter bach size should be > 0");
	batch_size=b;
}

int32_t CKMeansMiniBatch::get_batch_size() const
{
	return batch_size;
}

void CKMeansMiniBatch::set_mb_params(int32_t b, int32_t t)
{
	REQUIRE(b>0, "Parameter bach size should be > 0");
	REQUIRE(t>0, "Parameter number of iterations should be > 0");
	batch_size=b;
	max_iter = t;
}

void CKMeansMiniBatch::minibatch_KMeans()
{
	REQUIRE(batch_size>0,
		"batch size not set to positive value. Current batch size %d \n", batch_size);
	REQUIRE(
		max_iter > 0, "number of iterations not set to positive value. Current "
		              "iterations %d \n",
		max_iter);

	CDenseFeatures<float64_t>* lhs=
		distance->get_lhs()->as<CDenseFeatures<float64_t>>();
	CDenseFeatures<float64_t>* rhs_mus=new CDenseFeatures<float64_t>(mus);
	CFeatures* rhs_cache=distance->replace_rhs(rhs_mus);
	int32_t XSize=lhs->get_num_vectors();
	int32_t dims=lhs->get_num_features();

	SGVector<float64_t> v=SGVector<float64_t>(k);
	v.zero();

	for (auto i: progress(range(max_iter)))
	{
		SGVector<int32_t> M=mbchoose_rand(batch_size,XSize);
		SGVector<int32_t> ncent=SGVector<int32_t>(batch_size);
		for (int32_t j=0; j<batch_size; j++)
		{
			SGVector<float64_t> dists=SGVector<float64_t>(k);
			for (int32_t p=0; p<k; p++)
				dists[p]=distance->distance(M[j],p);

			int32_t imin=0;
			float64_t min=dists[0];
			for (int32_t p=1; p<k; p++)
			{
				if (dists[p]<min)
				{
					imin=p;
					min=dists[p];
				}
			}
			ncent[j]=imin;
		}
		for (int32_t j=0; j<batch_size; j++)
		{
			int32_t near=ncent[j];
			SGVector<float64_t> c_alive=rhs_mus->get_feature_vector(near);
			SGVector<float64_t> x=lhs->get_feature_vector(M[j]);
			v[near]+=1.0;
			float64_t eta=1.0/v[near];
			for (int32_t c=0; c<dims; c++)
			{
				c_alive[c]=(1.0-eta)*c_alive[c]+eta*x[c];
			}
		}
	}
	SG_UNREF(lhs);
	distance->replace_rhs(rhs_cache);
	delete rhs_mus;
}

SGVector<int32_t> CKMeansMiniBatch::mbchoose_rand(int32_t b, int32_t num)
{
	SGVector<int32_t> chosen=SGVector<int32_t>(num);
	SGVector<int32_t> ret=SGVector<int32_t>(b);
	chosen.zero();
	int32_t ch=0;
	while (ch<b)
	{
		const int32_t n=CMath::random(0,num-1);
		if (chosen[n]==0)
		{
			chosen[n]+=1;
			ret[ch]=n;
			ch++;
		}
	}
	return ret;
}

void CKMeansMiniBatch::init_mb_params()
{
	batch_size=-1;

	SG_ADD(
		&batch_size, "batch_size", "batch size for mini-batch KMeans",
		MS_NOT_AVAILABLE);
}

bool CKMeansMiniBatch::train_machine(CFeatures* data)
{
	initialize_training(data);
	minibatch_KMeans();
	compute_cluster_variances();
	return true;
}

}
