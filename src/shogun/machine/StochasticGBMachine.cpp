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

#include <shogun/base/progress.h>
#include <shogun/lib/View.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/optimization/lbfgs/lbfgs.h>

using namespace shogun;

StochasticGBMachine::StochasticGBMachine(std::shared_ptr<Machine> machine, std::shared_ptr<LossFunction> loss, int32_t num_iterations,
						float64_t learning_rate, float64_t subset_fraction)
: RandomMixin<Machine>()
{
	init();

	if (machine!=NULL)
	{

		m_machine=machine;
	}

	if (loss!=NULL)
	{

		m_loss=loss;
	}

	m_num_iter=num_iterations;
	m_subset_frac=subset_fraction;
	m_learning_rate=learning_rate;
}

StochasticGBMachine::~StochasticGBMachine()
{




}

void StochasticGBMachine::set_machine(std::shared_ptr<Machine> machine)
{
	require(machine,"Supplied machine is NULL");

	m_machine=machine;
}

std::shared_ptr<Machine> StochasticGBMachine::get_machine() const
{
	if (m_machine==NULL)
		error("machine not set yet!");


	return m_machine;
}

void StochasticGBMachine::set_loss_function(std::shared_ptr<LossFunction> f)
{
	require(f,"Supplied loss function is NULL");
	m_loss=f;
}

std::shared_ptr<LossFunction> StochasticGBMachine::get_loss_function() const
{
	if (m_loss==NULL)
		error("Loss function not set yet!");


	return m_loss;
}

void StochasticGBMachine::set_num_iterations(int32_t iter)
{
	require(iter,"Number of iterations");
	m_num_iter=iter;
}

int32_t StochasticGBMachine::get_num_iterations() const
{
	return m_num_iter;
}

void StochasticGBMachine::set_subset_fraction(float64_t frac)
{
	require((frac>0)&&(frac<=1),"subset fraction should lie between 0 and 1. Supplied value is {}",frac);

	m_subset_frac=frac;
}

float64_t StochasticGBMachine::get_subset_fraction() const
{
	return m_subset_frac;
}

void StochasticGBMachine::set_learning_rate(float64_t lr)
{
	require((lr>0)&&(lr<=1),"learning rate should lie between 0 and 1. Supplied value is {}",lr);

	m_learning_rate=lr;
}

float64_t StochasticGBMachine::get_learning_rate() const
{
	return m_learning_rate;
}

std::shared_ptr<RegressionLabels> StochasticGBMachine::apply_regression(std::shared_ptr<Features> data)
{
	require(data,"test data supplied is NULL");
	auto feats=data->as<DenseFeatures<float64_t>>();

	SGVector<float64_t> retlabs(feats->get_num_vectors());
	retlabs.fill_vector(retlabs.vector,retlabs.vlen,0);
	for (int32_t i=0;i<m_num_iter;i++)
	{
		float64_t gamma=m_gamma->get_element(i);

		auto machine=m_weak_learners->get_element<Machine>(i);
		require(machine, "{} element of the array of weak learners is NULL. This is not expected",i);

		auto dlabels=machine->apply_regression(feats);
		SGVector<float64_t> delta=dlabels->get_labels();

		for (int32_t j=0;j<retlabs.vlen;j++)
			retlabs[j]+=delta[j]*gamma*m_learning_rate;



	}

	return std::make_shared<RegressionLabels>(retlabs);
}

bool StochasticGBMachine::train_machine(std::shared_ptr<Features> data)
{
	require(data,"training data not supplied!");
	require(m_machine,"machine not set!");
	require(m_loss,"loss function not specified");
	require(m_labels, "labels not specified");

	auto feats=data->as<DenseFeatures<float64_t>>();

	// initialize weak learners array and gamma array
	initialize_learners();

	// cache predicted labels for intermediate models
	auto interf=std::make_shared<RegressionLabels>(feats->get_num_vectors());

	for (int32_t i=0;i<interf->get_num_labels();i++)
		interf->set_label(i,0);

	for (auto i : SG_PROGRESS(range(m_num_iter)))
	{
		const auto result = get_subset(feats, interf);
		const auto& feats_iter = std::get<0>(result);
		const auto& interf_iter = std::get<1>(result);
		const auto& labels_iter = std::get<2>(result);

		// compute pseudo-residuals
		auto pres =
		    compute_pseudo_residuals(interf_iter, labels_iter);

		// fit learner
		auto wlearner = fit_model(feats_iter, pres);
		m_weak_learners->push_back(wlearner);

		// compute multiplier
		auto hm = wlearner->apply_regression(feats_iter);

		float64_t gamma = compute_multiplier(interf_iter, hm, labels_iter);
		m_gamma->push_back(gamma);

		// update intermediate function value
		auto dlabels=wlearner->apply_regression(feats);
		SGVector<float64_t> delta=dlabels->get_labels();
		for (int32_t j=0;j<interf->get_num_labels();j++)
			interf->set_label(j,interf->get_label(j)+delta[j]*gamma*m_learning_rate);




	}


	return true;
}

float64_t StochasticGBMachine::compute_multiplier(
    std::shared_ptr<RegressionLabels> f, std::shared_ptr<RegressionLabels> hm, std::shared_ptr<Labels> labs)
{
	require(f->get_num_labels()==hm->get_num_labels(),"The number of labels in both input parameters should be equal");

	auto instance=std::make_shared<DynamicObjectArray>();
	instance->push_back(labs);
	instance->push_back(f);
	instance->push_back(hm);
	instance->push_back(m_loss);

	float64_t ret=get_gamma(instance.get());


	return ret;
}

std::shared_ptr<Machine> StochasticGBMachine::fit_model(std::shared_ptr<DenseFeatures<float64_t>> feats, std::shared_ptr<RegressionLabels> labels)
{
	// clone base machine
	auto c=m_machine->clone()->as<Machine>();
	// train cloned machine
	c->set_labels(labels);
	c->train(feats);

	return c;
}

std::shared_ptr<RegressionLabels> StochasticGBMachine::compute_pseudo_residuals(
    std::shared_ptr<RegressionLabels> inter_f, std::shared_ptr<Labels> labs)
{
	auto labels = labs->as<DenseLabels>()->get_labels();
	SGVector<float64_t> f=inter_f->get_labels();

	SGVector<float64_t> residuals(f.vlen);
	for (int32_t i=0;i<residuals.vlen;i++)
		residuals[i]=-m_loss->first_derivative(f[i],labels[i]);

	return std::make_shared<RegressionLabels>(residuals);
}

std::tuple<std::shared_ptr<DenseFeatures<float64_t>>, std::shared_ptr<RegressionLabels>,
           std::shared_ptr<Labels>>
StochasticGBMachine::get_subset(
    std::shared_ptr<DenseFeatures<float64_t>> f, std::shared_ptr<RegressionLabels> interf)
{
	if (m_subset_frac == 1.0)
		return std::make_tuple(f, interf, m_labels);

	int32_t subset_size=m_subset_frac*(f->get_num_vectors());
	SGVector<index_t> idx(f->get_num_vectors());
	idx.range_fill();
	random::shuffle(idx, m_prng);

	SGVector<index_t> subset(subset_size);
	sg_memcpy(subset.vector,idx.vector,subset.vlen*sizeof(index_t));

	return std::make_tuple(
	    view(f, subset), view(interf, subset),
	    view(m_labels, subset));
}

void StochasticGBMachine::initialize_learners()
{

	m_weak_learners=std::make_shared<DynamicObjectArray>();



	m_gamma=std::make_shared<DynamicArray<float64_t>>();

}

float64_t StochasticGBMachine::get_gamma(void* instance)
{
	lbfgs_parameter_t lbfgs_param;
	lbfgs_parameter_init(&lbfgs_param);
	lbfgs_param.linesearch=2;

	float64_t gamma=0;
	lbfgs(1,&gamma,NULL,StochasticGBMachine::lbfgs_evaluate,NULL,instance,&lbfgs_param);

	return gamma;
}

float64_t StochasticGBMachine::lbfgs_evaluate(void *obj, const float64_t *parameters, float64_t *gradient, const int dim,
												const float64_t step)
{
	require(obj,"object cannot be NULL");
	auto objects=(DynamicObjectArray*)obj;
	require((objects->get_num_elements()==2) || (objects->get_num_elements()==4),"Number of elements in obj array"
	" ({}) does not match expectations(2 or 4)",objects->get_num_elements());

	if (objects->get_num_elements()==2)
	{
		// extract labels
		auto lab=objects->get_element<DenseLabels>(0);
		require(lab,"0 index element of objects is NULL");
		SGVector<float64_t> labels=lab->get_labels();

		// extract loss function
		auto lossf =objects->get_element<LossFunction>(1);
		require(lossf,"1 index element of objects is NULL");

		*gradient=0;
		float64_t ret=0;
		for (int32_t i=0;i<labels.vlen;i++)
		{
			*gradient+=lossf->first_derivative((*parameters),labels[i]);
			ret+=lossf->loss((*parameters),labels[i]);
		}



		return ret;
	}

	// extract labels
	auto lab=objects->get_element<DenseLabels>(0);
	require(lab,"0 index element of objects is NULL");
	SGVector<float64_t> labels=lab->get_labels();

	// extract f
	auto func=objects->get_element<DenseLabels>(1);
	require(func,"1 index element of objects is NULL");
	SGVector<float64_t> f=func->get_labels();

	// extract hm
	auto delta=objects->get_element<DenseLabels>(2);
	require(delta,"2 index element of objects is NULL");
	SGVector<float64_t> hm=delta->get_labels();

	// extract loss function
	auto lossf=objects->get_element<LossFunction>(3);
	require(lossf,"3 index element of objects is NULL");

	*gradient=0;
	float64_t ret=0;
	for (int32_t i=0;i<labels.vlen;i++)
	{
		*gradient+=lossf->first_derivative((*parameters)*hm[i]+f[i],labels[i]);
		ret+=lossf->loss((*parameters)*hm[i]+f[i],labels[i]);
	}





	return ret;
}

void StochasticGBMachine::init()
{
	m_machine=nullptr;
	m_loss=nullptr;
	m_num_iter=0;
	m_subset_frac=0;
	m_learning_rate=0;

	m_weak_learners=std::make_shared<DynamicObjectArray>();
	m_gamma=std::make_shared<DynamicArray<float64_t>>();

	SG_ADD((std::shared_ptr<SGObject>*)&m_machine,"m_machine","machine");
	SG_ADD((std::shared_ptr<SGObject>*)&m_loss,"m_loss","loss function");
	SG_ADD(&m_num_iter,"m_num_iter","number of iterations");
	SG_ADD(&m_subset_frac,"m_subset_frac","subset fraction");
	SG_ADD(&m_learning_rate,"m_learning_rate","learning rate");
	SG_ADD((std::shared_ptr<SGObject>*)&m_weak_learners,"m_weak_learners","array of weak learners");
	SG_ADD((std::shared_ptr<SGObject>*)&m_gamma,"m_gamma","array of learner weights");
}
