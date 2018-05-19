/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Alesis Novik, Weijie Lin, Sergey Lisitsyn,
 *          Heiko Strathmann, Evgeniy Andreev, Chiyuan Zhang, Evan Shelhamer,
 *          Wuwei Lin, Marcus Edel
 */
#include <shogun/lib/config.h>

#include <shogun/base/some.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/progress.h>
#include <shogun/clustering/GMM.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/multiclass/KNN.h>
#include <vector>

using namespace shogun;
using namespace std;

CGMM::CGMM() : CDistribution(), m_components(),	m_coefficients()
{
	register_params();
}

CGMM::CGMM(int32_t n, ECovType cov_type) : CDistribution(), m_components(), m_coefficients()
{
	m_coefficients = SGVector<float64_t>(n);
	m_components = vector<CGaussian*>(n);

	for (int32_t i=0; i<n; i++)
	{
		m_components[i]=new CGaussian();
		SG_REF(m_components[i]);
		m_components[i]->set_cov_type(cov_type);
	}

	register_params();
}

CGMM::CGMM(vector<CGaussian*> components, SGVector<float64_t> coefficients, bool copy) : CDistribution()
{
	ASSERT(int32_t(components.size())==coefficients.vlen)

	if (!copy)
	{
		m_components=components;
		m_coefficients=coefficients;
		for (int32_t i=0; i<int32_t(components.size()); i++)
		{
			SG_REF(m_components[i]);
		}
	}
	else
	{
		m_coefficients = coefficients;
		m_components = vector<CGaussian*>(components.size());

		for (int32_t i=0; i<int32_t(components.size()); i++)
		{
			m_components[i]=new CGaussian();
			SG_REF(m_components[i]);
			m_components[i]->set_cov_type(components[i]->get_cov_type());

			SGVector<float64_t> old_mean=components[i]->get_mean();
			SGVector<float64_t> new_mean = old_mean.clone();
			m_components[i]->set_mean(new_mean);

			SGVector<float64_t> old_d=components[i]->get_d();
			SGVector<float64_t> new_d = old_d.clone();
			m_components[i]->set_d(new_d);

			if (components[i]->get_cov_type()==FULL)
			{
				SGMatrix<float64_t> old_u=components[i]->get_u();
				SGMatrix<float64_t> new_u = old_u.clone();
				m_components[i]->set_u(new_u);
			}

			m_coefficients[i]=coefficients[i];
		}
	}

	register_params();
}

CGMM::~CGMM()
{
	if (!m_components.empty())
		cleanup();
}

void CGMM::cleanup()
{
	for (int32_t i = 0; i < int32_t(m_components.size()); i++)
		SG_UNREF(m_components[i]);

	m_components = vector<CGaussian*>();
	m_coefficients = SGVector<float64_t>();
}

bool CGMM::train(CFeatures* data)
{
	ASSERT(m_components.size() != 0)

	/** init features with data if necessary and assure type is correct */
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features(data);
	}

	return true;
}

float64_t CGMM::train_em(float64_t min_cov, int32_t max_iter, float64_t min_change)
{
	if (!features)
		SG_ERROR("No features to train on.\n")

	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_vectors=dotdata->get_num_vectors();

	SGMatrix<float64_t> alpha;

	/* compute initialization via kmeans if none is present */
	if (m_components[0]->get_mean().vector==NULL)
	{
		CKMeans* init_k_means=new CKMeans(int32_t(m_components.size()), new CEuclideanDistance());
		init_k_means->train(dotdata);
		SGMatrix<float64_t> init_means=init_k_means->get_cluster_centers();

		alpha=alpha_init(init_means);

		SG_UNREF(init_k_means);

		max_likelihood(alpha, min_cov);
	}
	else
		alpha=SGMatrix<float64_t>(num_vectors,int32_t(m_components.size()));

	int32_t iter=0;
	float64_t log_likelihood_prev=0;
	float64_t log_likelihood_cur=0;
	SGVector<float64_t> logPxy(num_vectors * m_components.size());
	SGVector<float64_t> logPx(num_vectors);
	//float64_t* logPost=SG_MALLOC(float64_t, num_vectors*m_components.vlen);
	auto pb = progress(range(max_iter));
	while (iter<max_iter)
	{
		log_likelihood_prev=log_likelihood_cur;
		log_likelihood_cur=0;

		for (int32_t i=0; i<num_vectors; i++)
		{
			logPx[i]=0;
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
			for (int32_t j=0; j<int32_t(m_components.size()); j++)
			{
				logPxy[index_t(i * m_components.size() + j)] =
				    m_components[j]->compute_log_PDF(v) +
				    std::log(m_coefficients[j]);
				logPx[i] +=
				    std::exp(logPxy[index_t(i * m_components.size() + j)]);
			}

			logPx[i] = std::log(logPx[i]);
			log_likelihood_cur+=logPx[i];

			for (int32_t j=0; j<int32_t(m_components.size()); j++)
			{
				//logPost[i*m_components.vlen+j]=logPxy[i*m_components.vlen+j]-logPx[i];
				alpha.matrix[i * m_components.size() + j] = std::exp(
				    logPxy[index_t(i * m_components.size() + j)] - logPx[i]);
			}
		}

		if (iter>0 && log_likelihood_cur-log_likelihood_prev<min_change)
			break;
		pb.print_progress();
		max_likelihood(alpha, min_cov);

		iter++;
	}
	pb.complete();
	return log_likelihood_cur;
}

float64_t CGMM::train_smem(int32_t max_iter, int32_t max_cand, float64_t min_cov, int32_t max_em_iter, float64_t min_change)
{
	if (!features)
		SG_ERROR("No features to train on.\n")

	if (m_components.size()<3)
		SG_ERROR("Can't run SMEM with less than 3 component mixture model.\n")

	CDotFeatures* dotdata = features->as<CDotFeatures>();
	auto num_vectors = dotdata->get_num_vectors();

	float64_t cur_likelihood=train_em(min_cov, max_em_iter, min_change);

	int32_t iter=0;
	SGVector<float64_t> logPxy(num_vectors * m_components.size());
	SGVector<float64_t> logPx(num_vectors);
	SGVector<float64_t> logPost(num_vectors * m_components.size());
	SGVector<float64_t> logPostSum(m_components.size());
	SGVector<float64_t> logPostSum2(m_components.size());
	SGVector<float64_t> logPostSumSum(
	    m_components.size() * (m_components.size() - 1) / 2);
	SGVector<float64_t> split_crit(m_components.size());
	SGVector<float64_t> merge_crit(
	    m_components.size() * (m_components.size() - 1) / 2);
	SGVector<int32_t> split_ind(m_components.size());
	SGVector<int32_t> merge_ind(
	    m_components.size() * (m_components.size() - 1) / 2);

	auto pb = progress(range(max_iter));
	while (iter<max_iter)
	{
		linalg::zero(logPostSum);
		linalg::zero(logPostSum2);
		linalg::zero(logPostSumSum);
		for (int32_t i=0; i<num_vectors; i++)
		{
			logPx[i]=0;
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
			for (int32_t j=0; j<int32_t(m_components.size()); j++)
			{
				logPxy[index_t(i * m_components.size() + j)] =
				    m_components[j]->compute_log_PDF(v) +
				    std::log(m_coefficients[j]);
				logPx[i] +=
				    std::exp(logPxy[index_t(i * m_components.size() + j)]);
			}

			logPx[i] = std::log(logPx[i]);

			for (int32_t j=0; j<int32_t(m_components.size()); j++)
			{
				logPost[index_t(i * m_components.size() + j)] =
				    logPxy[index_t(i * m_components.size() + j)] - logPx[i];
				logPostSum[j] +=
				    std::exp(logPost[index_t(i * m_components.size() + j)]);
				logPostSum2[j] +=
				    std::exp(2 * logPost[index_t(i * m_components.size() + j)]);
			}

			int32_t counter=0;
			for (int32_t j=0; j<int32_t(m_components.size()); j++)
			{
				for (int32_t k=j+1; k<int32_t(m_components.size()); k++)
				{
					logPostSumSum[counter] += std::exp(
					    logPost[index_t(i * m_components.size() + j)] +
					    logPost[index_t(i * m_components.size() + k)]);
					counter++;
				}
			}
		}

		int32_t counter=0;
		for (int32_t i=0; i<int32_t(m_components.size()); i++)
		{
			logPostSum[i] = std::log(logPostSum[i]);
			split_crit[i]=0;
			split_ind[i]=i;
			for (int32_t j=0; j<num_vectors; j++)
			{
				split_crit[i] +=
				    (logPost[index_t(j * m_components.size() + i)] -
				     logPostSum[i] -
				     logPxy[index_t(j * m_components.size() + i)] +
				     std::log(m_coefficients[i])) *
				    (std::exp(logPost[index_t(j * m_components.size() + i)]) /
				     std::exp(logPostSum[i]));
			}
			for (int32_t j=i+1; j<int32_t(m_components.size()); j++)
			{
				merge_crit[counter] = std::log(logPostSumSum[counter]) -
				                      (0.5 * std::log(logPostSum2[i])) -
				                      (0.5 * std::log(logPostSum2[j]));
				merge_ind[counter]=i*m_components.size()+j;
				counter++;
			}
		}
		CMath::qsort_backward_index(
		    split_crit.vector, split_ind.vector, int32_t(m_components.size()));
		CMath::qsort_backward_index(
		    merge_crit.vector, merge_ind.vector,
		    int32_t(m_components.size() * (m_components.size() - 1) / 2));

		bool better_found=false;
		int32_t candidates_checked=0;
		for (int32_t i=0; i<int32_t(m_components.size()); i++)
		{
			for (int32_t j=0; j<int32_t(m_components.size()*(m_components.size()-1)/2); j++)
			{
				if (merge_ind[j]/int32_t(m_components.size()) != split_ind[i] && int32_t(merge_ind[j]%m_components.size()) != split_ind[i])
				{
					candidates_checked++;
					CGMM* candidate=new CGMM(m_components, m_coefficients, true);
					candidate->train(features);
					candidate->partial_em(split_ind[i], merge_ind[j]/int32_t(m_components.size()), merge_ind[j]%int32_t(m_components.size()), min_cov, max_em_iter, min_change);
					float64_t cand_likelihood=candidate->train_em(min_cov, max_em_iter, min_change);

					if (cand_likelihood>cur_likelihood)
					{
						cur_likelihood=cand_likelihood;
						set_comp(candidate->get_comp());
						set_coef(candidate->get_coef());

						for (int32_t k=0; k<int32_t(candidate->get_comp().size()); k++)
						{
							SG_UNREF(candidate->get_comp()[k]);
						}

						better_found=true;
						delete candidate;
						break;
					}
					else
						delete candidate;

					if (candidates_checked>=max_cand)
						break;
				}
			}
			if (better_found || candidates_checked>=max_cand)
				break;
		}
		if (!better_found)
			break;
		iter++;
		pb.print_progress();
		
	}
	pb.complete();
	return cur_likelihood;
}

void CGMM::partial_em(int32_t comp1, int32_t comp2, int32_t comp3, float64_t min_cov, int32_t max_em_iter, float64_t min_change)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_vectors=dotdata->get_num_vectors();

	SGVector<float64_t> init_logPxy(num_vectors * m_components.size());
	SGVector<float64_t> init_logPx(num_vectors);
	SGVector<float64_t> init_logPx_fix(num_vectors);
	SGVector<float64_t> post_add(num_vectors);

	for (int32_t i=0; i<num_vectors; i++)
	{
		init_logPx[i]=0;
		init_logPx_fix[i]=0;

		SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
		for (int32_t j=0; j<int32_t(m_components.size()); j++)
		{
			init_logPxy[index_t(i * m_components.size() + j)] =
			    m_components[j]->compute_log_PDF(v) +
			    std::log(m_coefficients[j]);
			init_logPx[i] +=
			    std::exp(init_logPxy[index_t(i * m_components.size() + j)]);
			if (j!=comp1 && j!=comp2 && j!=comp3)
			{
				init_logPx_fix[i] +=
				    std::exp(init_logPxy[index_t(i * m_components.size() + j)]);
			}
		}

		init_logPx[i] = std::log(init_logPx[i]);
		post_add[i] = std::log(
		    std::exp(
		        init_logPxy[index_t(i * m_components.size() + comp1)] -
		        init_logPx[i]) +
		    std::exp(
		        init_logPxy[index_t(i * m_components.size() + comp2)] -
		        init_logPx[i]) +
		    std::exp(
		        init_logPxy[index_t(i * m_components.size() + comp3)] -
		        init_logPx[i]));
	}

	vector<CGaussian*> components(3);
	SGVector<float64_t> coefficients(3);
	components[0]=m_components[comp1];
	components[1]=m_components[comp2];
	components[2]=m_components[comp3];
	coefficients.vector[0]=m_coefficients.vector[comp1];
	coefficients.vector[1]=m_coefficients.vector[comp2];
	coefficients.vector[2]=m_coefficients.vector[comp3];
	float64_t coef_sum=coefficients.vector[0]+coefficients.vector[1]+coefficients.vector[2];

	int32_t dim_n=components[0]->get_mean().vlen;

	float64_t alpha1=coefficients.vector[1]/(coefficients.vector[1]+coefficients.vector[2]);
	float64_t alpha2=coefficients.vector[2]/(coefficients.vector[1]+coefficients.vector[2]);

	float64_t noise_mag =
	    SGVector<float64_t>::twonorm(components[0]->get_mean().vector, dim_n) *
	    0.1 / std::sqrt((float64_t)dim_n);

	SGVector<float64_t> mean(dim_n);
	linalg::add(components[1]->get_mean(), components[2]->get_mean(), mean, alpha1, alpha2);
	components[1]->set_mean(mean);

	for (int32_t i=0; i<dim_n; i++)
	{
		components[2]->get_mean().vector[i]=components[0]->get_mean().vector[i]+CMath::randn_double()*noise_mag;
		components[0]->get_mean().vector[i]=components[0]->get_mean().vector[i]+CMath::randn_double()*noise_mag;
	}

	coefficients.vector[1]=coefficients.vector[1]+coefficients.vector[2];
	coefficients.vector[2]=coefficients.vector[0]*0.5;
	coefficients.vector[0]=coefficients.vector[0]*0.5;

	if (components[0]->get_cov_type()==FULL)
	{
		SGMatrix<float64_t> c1=components[1]->get_cov();
		SGMatrix<float64_t> c2=components[2]->get_cov();
		linalg::add(c1, c2, c1, alpha1, alpha2);

		SGVector<float64_t> eigenvalues(dim_n);
		linalg::eigen_solver_symmetric(c1, eigenvalues, c1);

		components[1]->set_d(eigenvalues);
		components[1]->set_u(c1);

		float64_t new_d=0;
		for (int32_t i=0; i<dim_n; i++)
		{
			new_d += std::log(components[0]->get_d().vector[i]);
			for (int32_t j=0; j<dim_n; j++)
			{
				if (i==j)
				{
					components[0]->get_u().matrix[i*dim_n+j]=1;
					components[2]->get_u().matrix[i*dim_n+j]=1;
				}
				else
				{
					components[0]->get_u().matrix[i*dim_n+j]=0;
					components[2]->get_u().matrix[i*dim_n+j]=0;
				}
			}
		}
		new_d = std::exp(new_d * (1. / dim_n));
		for (int32_t i=0; i<dim_n; i++)
		{
			components[0]->get_d().vector[i]=new_d;
			components[2]->get_d().vector[i]=new_d;
		}
	}
	else if(components[0]->get_cov_type()==DIAG)
	{
		auto result_d = components[1]->get_d();
		auto temp_d = components[2]->get_d();
		linalg::add(result_d, temp_d, result_d, alpha1, alpha2);
		components[1]->set_d(result_d);

		float64_t new_d=0;
		for (int32_t i=0; i<dim_n; i++)
		{
			new_d += std::log(components[0]->get_d().vector[i]);
		}
		new_d = std::exp(new_d * (1. / dim_n));
		for (int32_t i=0; i<dim_n; i++)
		{
			components[0]->get_d().vector[i]=new_d;
			components[2]->get_d().vector[i]=new_d;
		}
	}
	else if(components[0]->get_cov_type()==SPHERICAL)
	{
		components[1]->get_d().vector[0]=alpha1*components[1]->get_d().vector[0]+
												alpha2*components[2]->get_d().vector[0];

		components[2]->get_d().vector[0]=components[0]->get_d().vector[0];
	}

	CGMM* partial_candidate=new CGMM(components, coefficients);
	partial_candidate->train(features);

	float64_t log_likelihood_prev=0;
	float64_t log_likelihood_cur=0;
	int32_t iter=0;
	SGMatrix<float64_t> alpha(num_vectors, 3);
	SGVector<float64_t> logPxy(num_vectors * 3);
	SGVector<float64_t> logPx(num_vectors);
	//float64_t* logPost=SG_MALLOC(float64_t, num_vectors*m_components.vlen);

	while (iter<max_em_iter)
	{
		log_likelihood_prev=log_likelihood_cur;
		log_likelihood_cur=0;

		for (int32_t i=0; i<num_vectors; i++)
		{
			logPx[i]=0;
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
			for (int32_t j=0; j<3; j++)
			{
				logPxy[i * 3 + j] = components[j]->compute_log_PDF(v) +
				                    std::log(coefficients[j]);
				logPx[i] += std::exp(logPxy[i * 3 + j]);
			}

			logPx[i] = std::log(logPx[i] + init_logPx_fix[i]);
			log_likelihood_cur+=logPx[i];

			for (int32_t j=0; j<3; j++)
			{
				//logPost[i*m_components.vlen+j]=logPxy[i*m_components.vlen+j]-logPx[i];
				alpha.matrix[i * 3 + j] =
				    std::exp(logPxy[i * 3 + j] - logPx[i] + post_add[i]);
			}
		}

		if (iter>0 && log_likelihood_cur-log_likelihood_prev<min_change)
			break;

		partial_candidate->max_likelihood(alpha, min_cov);
		partial_candidate->get_coef().vector[0]=partial_candidate->get_coef().vector[0]*coef_sum;
		partial_candidate->get_coef().vector[1]=partial_candidate->get_coef().vector[1]*coef_sum;
		partial_candidate->get_coef().vector[2]=partial_candidate->get_coef().vector[2]*coef_sum;
		iter++;
	}

	m_coefficients.vector[comp1]=coefficients.vector[0];
	m_coefficients.vector[comp2]=coefficients.vector[1];
	m_coefficients.vector[comp3]=coefficients.vector[2];

	delete partial_candidate;
}

void CGMM::max_likelihood(SGMatrix<float64_t> alpha, float64_t min_cov)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_dim=dotdata->get_dim_feature_space();

	float64_t alpha_sum;
	float64_t alpha_sum_sum=0;

	for (int32_t i=0; i<alpha.num_cols; i++)
	{
		alpha_sum=0;
		SGVector<float64_t> mean_sum(num_dim);
		linalg::zero(mean_sum);

		for (int32_t j=0; j<alpha.num_rows; j++)
		{
			alpha_sum+=alpha.matrix[j*alpha.num_cols+i];
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(j);
			linalg::add(
			    v, mean_sum, mean_sum, alpha.matrix[j * alpha.num_cols + i],
			    1.0);
		}

		linalg::scale(mean_sum, mean_sum, 1.0 / alpha_sum);

		m_components[i]->set_mean(mean_sum);

		SGMatrix<float64_t> cov_sum;

		ECovType cov_type = m_components[i]->get_cov_type();
		if (cov_type==FULL)
		{
			cov_sum = SGMatrix<float64_t>(num_dim, num_dim);
			linalg::zero(cov_sum);
		}
		else if(cov_type==DIAG)
		{
			cov_sum = SGMatrix<float64_t>(1, num_dim);
			linalg::zero(cov_sum);
		}
		else if(cov_type==SPHERICAL)
		{
			cov_sum = SGMatrix<float64_t>(1, 1);
			linalg::zero(cov_sum);
		}

		for (int32_t j=0; j<alpha.num_rows; j++)
		{
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(j);

			linalg::add(v, mean_sum, v, 1.0, -1.0);
			switch (cov_type)
			{
				case FULL:
				    linalg::dger(
				        alpha.matrix[j * alpha.num_cols + i], v, v, cov_sum);
				    break;
			    case DIAG:
			    {
				    auto temp_matrix = SGMatrix<float64_t>(v.vector, 1, v.vlen);
				    auto temp_result = linalg::matrix_prod(
				        temp_matrix, temp_matrix, true, false);
				    cov_sum = temp_result.get_diagonal_vector().clone();
				    linalg::scale(
				        cov_sum, cov_sum, alpha.matrix[j * alpha.num_cols + i]);
			    }

			    break;
			    case SPHERICAL:
				    float64_t temp = 0;

				    temp = linalg::dot(v, v);

				    cov_sum(0, 0) +=
				        temp * alpha.matrix[j * alpha.num_cols + i];
				    break;
			}
		}

		switch (cov_type)
		{
			case FULL:
		    {
			    linalg::scale(cov_sum, cov_sum, 1.0 / alpha_sum);

			    SGVector<float64_t> d0(num_dim);
			    linalg::eigen_solver_symmetric(cov_sum, d0, cov_sum);

			    for (auto& v: d0)
				    v = CMath::max(min_cov, v);

			    m_components[i]->set_d(d0);
			    m_components[i]->set_u(cov_sum);

			    break;
		    }
		    case DIAG:
			    for (int32_t j = 0; j < num_dim; j++)
			    {
				    cov_sum(0, j) /= alpha_sum;
				    cov_sum(0, j) = CMath::max(min_cov, cov_sum(0, j));
			    }

			    m_components[i]->set_d(cov_sum.get_row_vector(0));

			    break;
		    case SPHERICAL:
			    cov_sum[0] /= alpha_sum * num_dim;
			    cov_sum[0] = CMath::max(min_cov, cov_sum[0]);

			    m_components[i]->set_d(cov_sum.get_row_vector(0));

			    break;
		}

		m_coefficients.vector[i]=alpha_sum;
		alpha_sum_sum+=alpha_sum;
	}

	linalg::scale(m_coefficients, m_coefficients, 1.0 / alpha_sum_sum);
}

int32_t CGMM::get_num_model_parameters()
{
	return 1;
}

float64_t CGMM::get_log_model_parameter(int32_t num_param)
{
	ASSERT(num_param==1)

	return std::log(m_components.size());
}

index_t CGMM::get_num_components() const
{
	return m_components.size();
}

CDistribution* CGMM::get_component(index_t index) const
{
	return m_components[index];
}

float64_t CGMM::get_log_derivative(int32_t num_param, int32_t num_example)
{
	SG_NOTIMPLEMENTED
	return 0;
}

float64_t CGMM::get_log_likelihood_example(int32_t num_example)
{
	SG_NOTIMPLEMENTED
	return 1;
}

float64_t CGMM::get_likelihood_example(int32_t num_example)
{
	float64_t result=0;

	ASSERT(features);
	ASSERT(features->get_feature_class() == C_DENSE);
	ASSERT(features->get_feature_type() == F_DREAL);

	for (auto i: range(index_t(m_components.size())))
	{
		SGVector<float64_t> point= ((CDenseFeatures<float64_t>*) features)->get_feature_vector(num_example);
		result += std::exp(
		    m_components[i]->compute_log_PDF(point) +
		    std::log(m_coefficients[i]));
	}

	return result;
}

SGVector<float64_t> CGMM::get_nth_mean(int32_t num)
{
	ASSERT(num<int32_t(m_components.size()))
	return m_components[num]->get_mean();
}

void CGMM::set_nth_mean(SGVector<float64_t> mean, int32_t num)
{
	ASSERT(num<int32_t(m_components.size()))
	m_components[num]->set_mean(mean);
}

SGMatrix<float64_t> CGMM::get_nth_cov(int32_t num)
{
	ASSERT(num<int32_t(m_components.size()))
	return m_components[num]->get_cov();
}

void CGMM::set_nth_cov(SGMatrix<float64_t> cov, int32_t num)
{
	ASSERT(num<int32_t(m_components.size()))
	m_components[num]->set_cov(cov);
}

SGVector<float64_t> CGMM::get_coef()
{
	return m_coefficients;
}

void CGMM::set_coef(const SGVector<float64_t> coefficients)
{
	m_coefficients=coefficients;
}

vector<CGaussian*> CGMM::get_comp()
{
	return m_components;
}

void CGMM::set_comp(vector<CGaussian*> components)
{
	for (int32_t i=0; i<int32_t(m_components.size()); i++)
	{
		SG_UNREF(m_components[i]);
	}

	m_components=components;

	for (int32_t i=0; i<int32_t(m_components.size()); i++)
	{
		SG_REF(m_components[i]);
	}
}

SGMatrix<float64_t> CGMM::alpha_init(SGMatrix<float64_t> init_means)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	auto num_vectors=dotdata->get_num_vectors();

	SGVector<float64_t> label_num(init_means.num_cols);
	linalg::range_fill(label_num);

	auto knn=some<CKNN>(1, new CEuclideanDistance(), new CMulticlassLabels(label_num));
	knn->train(new CDenseFeatures<float64_t>(init_means));
	auto init_labels = knn->apply(features)->as<CMulticlassLabels>();

	SGMatrix<float64_t> alpha(num_vectors, index_t(m_components.size()));
	for (auto i: range(num_vectors))
		alpha[i * m_components.size() + init_labels->get_int_label(i)] = 1;

	SG_UNREF(init_labels);
	return alpha;
}

SGVector<float64_t> CGMM::sample()
{
	REQUIRE(m_components.size()>0, "Number of mixture components is %d but "
			"must be positive\n", m_components.size());
	float64_t rand_num = CMath::random(0.0, 1.0);
	float64_t cum_sum=0;
	for (auto i: range(m_coefficients.vlen))
	{
		cum_sum+=m_coefficients.vector[i];
		if (cum_sum>=rand_num)
		{
			SG_DEBUG("Sampling from mixture component %d\n", i);
			return m_components[i]->sample();
		}
	}
	return m_components[m_coefficients.vlen-1]->sample();
}

SGVector<float64_t> CGMM::cluster(SGVector<float64_t> point)
{
	SGVector<float64_t> answer(m_components.size()+1);
	answer.vector[m_components.size()]=0;

	for (auto i: range(index_t(m_components.size())))
	{
		answer.vector[i] = m_components[i]->compute_log_PDF(point) +
		                   std::log(m_coefficients[i]);
		answer.vector[m_components.size()] += std::exp(answer.vector[i]);
	}
	answer.vector[m_components.size()] =
	    std::log(answer.vector[m_components.size()]);

	return answer;
}

void CGMM::register_params()
{
	//TODO serialization broken
	//m_parameters->add((SGVector<CSGObject*>*) &m_components, "m_components", "Mixture components");
	SG_ADD(
	    &m_coefficients, "m_coefficients", "Mixture coefficients.",
	    MS_NOT_AVAILABLE);
}
