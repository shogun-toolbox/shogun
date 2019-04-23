/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Michele Mazzoni, Heiko Strathmann, Viktor Gal
 */

#include <shogun/base/progress.h>
#include <shogun/clustering/KMeansMiniBatch.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/mathematics/UniformIntDistribution.h>

#ifdef _WIN32
#undef far
#undef near
#endif

using namespace shogun;

namespace shogun
{
KMeansMiniBatch::KMeansMiniBatch():KMeansBase()
{
	init_mb_params();
}

KMeansMiniBatch::KMeansMiniBatch(int32_t k_i, std::shared_ptr<Distance> d_i, bool use_kmpp_i):KMeansBase(k_i, d_i, use_kmpp_i)
{
	init_mb_params();
}

KMeansMiniBatch::KMeansMiniBatch(int32_t k_i, std::shared_ptr<Distance> d_i, SGMatrix<float64_t> centers_i):KMeansBase(k_i, d_i, centers_i)
{
	init_mb_params();
}

KMeansMiniBatch::~KMeansMiniBatch()
{
}

void KMeansMiniBatch::minibatch_KMeans()
{
	require(batch_size>0,
		"batch size not set to positive value. Current batch size {} ", batch_size);
	require(
		max_iter > 0, "number of iterations not set to positive value. Current "
		              "iterations {} ",
		max_iter);

	auto lhs=
		distance->get_lhs()->as<DenseFeatures<float64_t>>();
	auto rhs_mus=std::make_shared<DenseFeatures<float64_t>>(mus);
	auto rhs_cache=distance->replace_rhs(rhs_mus);
	int32_t XSize=lhs->get_num_vectors();
	int32_t dims=lhs->get_num_features();

	SGVector<float64_t> v=SGVector<float64_t>(k);
	v.zero();

	for (auto i : SG_PROGRESS(range(max_iter)))
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
		mus = rhs_mus->get_feature_matrix();
		observe<SGMatrix<float64_t>>(i, "mus");
	}

	distance->replace_rhs(rhs_cache);
}

SGVector<int32_t> KMeansMiniBatch::mbchoose_rand(int32_t b, int32_t num)
{
	SGVector<int32_t> chosen=SGVector<int32_t>(num);
	SGVector<int32_t> ret=SGVector<int32_t>(b);
	chosen.zero();
	int32_t ch=0;
	UniformIntDistribution<int32_t> uniform_int_dist(0, num-1);
	while (ch<b)
	{
		const int32_t n=uniform_int_dist(m_prng);
		if (chosen[n]==0)
		{
			chosen[n]+=1;
			ret[ch]=n;
			ch++;
		}
	}
	return ret;
}

void KMeansMiniBatch::init_mb_params()
{
	batch_size = 100;

	SG_ADD(
	&batch_size, "batch_size", "batch size for mini-batch KMeans",
	ParameterProperties::HYPER | ParameterProperties::SETTING);
}

bool KMeansMiniBatch::train_machine(std::shared_ptr<Features> data)
{
	initialize_training(data);
	minibatch_KMeans();
	compute_cluster_variances();
	return true;
}

}
