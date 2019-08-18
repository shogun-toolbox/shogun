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
#include <shogun/base/some.h>
#include <shogun/lib/View.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/optimization/lbfgs/lbfgs.h>

using namespace shogun;

CStochasticGBMachine::CStochasticGBMachine(CMachine* machine, CLossFunction* loss, int32_t num_iterations,
						float64_t learning_rate, float64_t subset_fraction)
: RandomMixin<CMachine>()
{
	init();

	if (machine!=NULL)
	{
		SG_REF(machine);
		m_machine=machine;
	}

	if (loss!=NULL)
	{
		SG_REF(loss);
		m_loss=loss;
	}

	m_num_iter=num_iterations;
	m_subset_frac=subset_fraction;
	m_learning_rate=learning_rate;
}

CStochasticGBMachine::~CStochasticGBMachine()
{
	SG_UNREF(m_machine);
	SG_UNREF(m_loss);
	SG_UNREF(m_weak_learners);
	SG_UNREF(m_gamma);
}

void CStochasticGBMachine::set_machine(CMachine* machine)
{
	require(machine,"Supplied machine is NULL");

	if (m_machine!=NULL)
		SG_UNREF(m_machine);

	SG_REF(machine);
	m_machine=machine;
}

CMachine* CStochasticGBMachine::get_machine() const
{
	if (m_machine==NULL)
		error("machine not set yet!");

	SG_REF(m_machine);
	return m_machine;
}

void CStochasticGBMachine::set_loss_function(CLossFunction* f)
{
	require(f,"Supplied loss function is NULL");
	if (m_loss!=NULL)
		SG_UNREF(m_loss);

	SG_REF(f);
	m_loss=f;
}

CLossFunction* CStochasticGBMachine::get_loss_function() const
{
	if (m_loss==NULL)
		error("Loss function not set yet!");

	SG_REF(m_loss)
	return m_loss;
}

void CStochasticGBMachine::set_num_iterations(int32_t iter)
{
	require(iter,"Number of iterations");
	m_num_iter=iter;
}

int32_t CStochasticGBMachine::get_num_iterations() const
{
	return m_num_iter;
}

void CStochasticGBMachine::set_subset_fraction(float64_t frac)
{
	require((frac>0)&&(frac<=1),"subset fraction should lie between 0 and 1. Supplied value is {}",frac);

	m_subset_frac=frac;
}

float64_t CStochasticGBMachine::get_subset_fraction() const
{
	return m_subset_frac;
}

void CStochasticGBMachine::set_learning_rate(float64_t lr)
{
	require((lr>0)&&(lr<=1),"learning rate should lie between 0 and 1. Supplied value is {}",lr);

	m_learning_rate=lr;
}

float64_t CStochasticGBMachine::get_learning_rate() const
{
	return m_learning_rate;
}

CRegressionLabels* CStochasticGBMachine::apply_regression(CFeatures* data)
{
	require(data,"test data supplied is NULL");
	CDenseFeatures<float64_t>* feats=data->as<CDenseFeatures<float64_t>>();

	SGVector<float64_t> retlabs(feats->get_num_vectors());
	retlabs.fill_vector(retlabs.vector,retlabs.vlen,0);
	for (int32_t i=0;i<m_num_iter;i++)
	{
		float64_t gamma=m_gamma->get_element(i);

		CSGObject* element=m_weak_learners->get_element(i);
		require(element,"{} element of the array of weak learners is NULL. This is not expected",i);
		CMachine* machine=dynamic_cast<CMachine*>(element);

		CRegressionLabels* dlabels=machine->apply_regression(feats);
		SGVector<float64_t> delta=dlabels->get_labels();

		for (int32_t j=0;j<retlabs.vlen;j++)
			retlabs[j]+=delta[j]*gamma*m_learning_rate;

		SG_UNREF(dlabels);
		SG_UNREF(element);
	}

	return new CRegressionLabels(retlabs);
}

bool CStochasticGBMachine::train_machine(CFeatures* data)
{
	require(data,"training data not supplied!");
	require(m_machine,"machine not set!");
	require(m_loss,"loss function not specified");
	require(m_labels, "labels not specified");

	CDenseFeatures<float64_t>* feats=data->as<CDenseFeatures<float64_t>>();

	// initialize weak learners array and gamma array
	initialize_learners();

	// cache predicted labels for intermediate models
	CRegressionLabels* interf=new CRegressionLabels(feats->get_num_vectors());
	SG_REF(interf);
	for (int32_t i=0;i<interf->get_num_labels();i++)
		interf->set_label(i,0);

	for (auto i : SG_PROGRESS(range(m_num_iter)))
	{
		const auto result = get_subset(feats, interf);
		const auto& feats_iter = std::get<0>(result);
		const auto& interf_iter = std::get<1>(result);
		const auto& labels_iter = std::get<2>(result);

		// compute pseudo-residuals
		CRegressionLabels* pres =
		    compute_pseudo_residuals(interf_iter, labels_iter);

		// fit learner
		CMachine* wlearner = fit_model(feats_iter, pres);
		m_weak_learners->push_back(wlearner);

		// compute multiplier
		CRegressionLabels* hm = wlearner->apply_regression(feats_iter);
		SG_REF(hm);
		float64_t gamma = compute_multiplier(interf_iter, hm, labels_iter);
		m_gamma->push_back(gamma);

		// update intermediate function value
		CRegressionLabels* dlabels=wlearner->apply_regression(feats);
		SGVector<float64_t> delta=dlabels->get_labels();
		for (int32_t j=0;j<interf->get_num_labels();j++)
			interf->set_label(j,interf->get_label(j)+delta[j]*gamma*m_learning_rate);

		SG_UNREF(dlabels);
		SG_UNREF(hm);
		SG_UNREF(wlearner);
	}

	SG_UNREF(interf);
	return true;
}

float64_t CStochasticGBMachine::compute_multiplier(
    CRegressionLabels* f, CRegressionLabels* hm, CLabels* labs)
{
	require(f->get_num_labels()==hm->get_num_labels(),"The number of labels in both input parameters should be equal");

	CDynamicObjectArray* instance=new CDynamicObjectArray();
	instance->push_back(labs);
	instance->push_back(f);
	instance->push_back(hm);
	instance->push_back(m_loss);

	float64_t ret=get_gamma(instance);

	SG_UNREF(instance);
	return ret;
}

CMachine* CStochasticGBMachine::fit_model(CDenseFeatures<float64_t>* feats, CRegressionLabels* labels)
{
	// clone base machine
	CSGObject* obj=m_machine->clone();
	CMachine* c=NULL;
	if (obj)
		c=dynamic_cast<CMachine*>(obj);
	else
		error("Machine could not be cloned!");

	// train cloned machine
	c->set_labels(labels);
	c->train(feats);

	return c;
}

CRegressionLabels* CStochasticGBMachine::compute_pseudo_residuals(
    CRegressionLabels* inter_f, CLabels* labs)
{
	auto labels = labs->as<CDenseLabels>()->get_labels();
	SGVector<float64_t> f=inter_f->get_labels();

	SGVector<float64_t> residuals(f.vlen);
	for (int32_t i=0;i<residuals.vlen;i++)
		residuals[i]=-m_loss->first_derivative(f[i],labels[i]);

	return new CRegressionLabels(residuals);
}

std::tuple<Some<CDenseFeatures<float64_t>>, Some<CRegressionLabels>,
           Some<CLabels>>
CStochasticGBMachine::get_subset(
    CDenseFeatures<float64_t>* f, CRegressionLabels* interf)
{
	if (m_subset_frac == 1.0)
		return std::make_tuple(wrap(f), wrap(interf), wrap(m_labels));

	int32_t subset_size=m_subset_frac*(f->get_num_vectors());
	SGVector<index_t> idx(f->get_num_vectors());
	idx.range_fill();
	random::shuffle(idx, m_prng);

	SGVector<index_t> subset(subset_size);
	sg_memcpy(subset.vector,idx.vector,subset.vlen*sizeof(index_t));

	return std::make_tuple(
	    wrap(view(f, subset)), wrap(view(interf, subset)),
	    wrap(view(m_labels, subset)));
}

void CStochasticGBMachine::initialize_learners()
{
	SG_UNREF(m_weak_learners);
	m_weak_learners=new CDynamicObjectArray();
	SG_REF(m_weak_learners);

	SG_UNREF(m_gamma);
	m_gamma=new CDynamicArray<float64_t>();
	SG_REF(m_gamma);
}

float64_t CStochasticGBMachine::get_gamma(void* instance)
{
	lbfgs_parameter_t lbfgs_param;
	lbfgs_parameter_init(&lbfgs_param);
	lbfgs_param.linesearch=2;

	float64_t gamma=0;
	lbfgs(1,&gamma,NULL,CStochasticGBMachine::lbfgs_evaluate,NULL,instance,&lbfgs_param);

	return gamma;
}

float64_t CStochasticGBMachine::lbfgs_evaluate(void *obj, const float64_t *parameters, float64_t *gradient, const int dim,
												const float64_t step)
{
	require(obj,"object cannot be NULL");
	CDynamicObjectArray* objects=static_cast<CDynamicObjectArray*>(obj);
	require((objects->get_num_elements()==2) || (objects->get_num_elements()==4),"Number of elements in obj array"
	" ({}) does not match expectations(2 or 4)",objects->get_num_elements());

	if (objects->get_num_elements()==2)
	{
		// extract labels
		CSGObject* element=objects->get_element(0);
		require(element,"0 index element of objects is NULL");
		CDenseLabels* lab=dynamic_cast<CDenseLabels*>(element);
		SGVector<float64_t> labels=lab->get_labels();

		// extract loss function
		element=objects->get_element(1);
		require(element,"1 index element of objects is NULL");
		CLossFunction* lossf=dynamic_cast<CLossFunction*>(element);

		*gradient=0;
		float64_t ret=0;
		for (int32_t i=0;i<labels.vlen;i++)
		{
			*gradient+=lossf->first_derivative((*parameters),labels[i]);
			ret+=lossf->loss((*parameters),labels[i]);
		}

		SG_UNREF(lab);
		SG_UNREF(lossf);
		return ret;
	}

	// extract labels
	CSGObject* element=objects->get_element(0);
	require(element,"0 index element of objects is NULL");
	CDenseLabels* lab=dynamic_cast<CDenseLabels*>(element);
	SGVector<float64_t> labels=lab->get_labels();

	// extract f
	element=objects->get_element(1);
	require(element,"1 index element of objects is NULL");
	CDenseLabels* func=dynamic_cast<CDenseLabels*>(element);
	SGVector<float64_t> f=func->get_labels();

	// extract hm
	element=objects->get_element(2);
	require(element,"2 index element of objects is NULL");
	CDenseLabels* delta=dynamic_cast<CDenseLabels*>(element);
	SGVector<float64_t> hm=delta->get_labels();

	// extract loss function
	element=objects->get_element(3);
	require(element,"3 index element of objects is NULL");
	CLossFunction* lossf=dynamic_cast<CLossFunction*>(element);

	*gradient=0;
	float64_t ret=0;
	for (int32_t i=0;i<labels.vlen;i++)
	{
		*gradient+=lossf->first_derivative((*parameters)*hm[i]+f[i],labels[i]);
		ret+=lossf->loss((*parameters)*hm[i]+f[i],labels[i]);
	}

	SG_UNREF(lab);
	SG_UNREF(delta);
	SG_UNREF(func);
	SG_UNREF(lossf)
	return ret;
}

void CStochasticGBMachine::init()
{
	m_machine=NULL;
	m_loss=NULL;
	m_num_iter=0;
	m_subset_frac=0;
	m_learning_rate=0;

	m_weak_learners=new CDynamicObjectArray();
	SG_REF(m_weak_learners);

	m_gamma=new CDynamicArray<float64_t>();
	SG_REF(m_gamma);

	SG_ADD((CSGObject**)&m_machine,"m_machine","machine");
	SG_ADD((CSGObject**)&m_loss,"m_loss","loss function");
	SG_ADD(&m_num_iter,"m_num_iter","number of iterations");
	SG_ADD(&m_subset_frac,"m_subset_frac","subset fraction");
	SG_ADD(&m_learning_rate,"m_learning_rate","learning rate");
	SG_ADD((CSGObject**)&m_weak_learners,"m_weak_learners","array of weak learners");
	SG_ADD((CSGObject**)&m_gamma,"m_gamma","array of learner weights");
}
