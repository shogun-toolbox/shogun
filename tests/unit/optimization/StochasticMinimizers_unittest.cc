/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wu Lin
 */
#include <gtest/gtest.h>

#include "StochasticMinimizers_unittest.h"

#include <shogun/optimization/L2Penalty.h>
#include <shogun/optimization/SGDMinimizer.h>
#include <shogun/optimization/GradientDescendUpdater.h>
#include <shogun/optimization/ConstLearningRate.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/optimization/SVRGMinimizer.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/Map.h>
#include <shogun/optimization/StandardMomentumCorrection.h>
#include <shogun/optimization/AdaDeltaUpdater.h>
#include <shogun/optimization/AdamUpdater.h>
#include <shogun/optimization/NesterovMomentumCorrection.h>
#include <shogun/optimization/RmsPropUpdater.h>
#include <shogun/optimization/AdaptMomentumCorrection.h>
#include <shogun/optimization/L1PenaltyForTG.h>
#include <shogun/optimization/InverseScalingLearningRate.h>
#include <shogun/optimization/ElasticNetPenalty.h>
#include <shogun/optimization/SMIDASMinimizer.h>
#include <shogun/optimization/PNormMappingFunction.h>
using namespace shogun;
using namespace Eigen;

int CRegressionExample::get_sample_size()
{
	return m_y.vlen;
}

CRegressionExample::CRegressionExample()
{
	init();
}
void CRegressionExample::init()
{
	m_w=SGVector<float64_t>();
	m_y=SGVector<float64_t>();
	m_x=SGMatrix<float64_t>();
	SG_ADD(&m_w, "r_w", "r_w",MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_x, "r_x", "r_x",MS_NOT_AVAILABLE);
	SG_ADD(&m_y, "r_y", "r_y",MS_NOT_AVAILABLE);
}

float64_t CRegressionExample::get_cost()
{
	REQUIRE(m_x.num_rows==m_y.vlen,"samples size (%d %d) must match\n",m_x.num_rows,m_y.vlen);
	REQUIRE(m_x.num_cols==m_w.vlen,"features size (%d %d) must match\n",m_x.num_cols,m_w.vlen);

	Map<MatrixXd> e_x(m_x.matrix, m_x.num_rows, m_x.num_cols);
	Map<VectorXd> e_w(m_w.vector, m_w.vlen);
	Map<VectorXd> e_y(m_y.vector, m_y.vlen);

	return 0.5*(e_y-(e_x*e_w)).array().pow(2.0).sum();
}

SGVector<float64_t> CRegressionExample::get_gradient(TParameter * param)
{
	REQUIRE(param, "param not set\n");
	REQUIRE(!strcmp(param->m_name, "r_w"), "Can't compute derivative wrt %s.%s parameter\n",
		get_name(), param->m_name);
	SGVector<float64_t> res(m_w.vlen);
	res.set_const(0.0);

	Map<VectorXd> e_res(res.vector,res.vlen);
	Map<MatrixXd> e_x(m_x.matrix, m_x.num_rows, m_x.num_cols);
	Map<VectorXd> e_w(m_w.vector, m_w.vlen);
	Map<VectorXd> e_y(m_y.vector, m_y.vlen);

	e_res=e_x.transpose()*(e_x*e_w-e_y);
	return res;
}

SGVector<float64_t> CRegressionExample::get_variable(TParameter * param)
{
	REQUIRE(param, "param not set\n");

	REQUIRE(!strcmp(param->m_name, "r_w"), "Can't compute derivative wrt %s.%s parameter\n",
		get_name(), param->m_name);
	return m_w;
}

SGVector<float64_t> CRegressionExample::get_gradient(TParameter * param, index_t idx)
{
	REQUIRE(param, "param not set\n");
	REQUIRE(!strcmp(param->m_name, "r_w"), "Can't compute derivative wrt %s.%s parameter\n",
		get_name(), param->m_name);
	REQUIRE(idx>=0 && idx<m_y.vlen,"out of bound\n");
	SGVector<float64_t> res(m_w.vlen);
	res.set_const(0.0);

	Map<VectorXd> e_res(res.vector,res.vlen);
	Map<MatrixXd> e_x(m_x.matrix, m_x.num_rows, m_x.num_cols);
	Map<VectorXd> e_w(m_w.vector, m_w.vlen);
	Map<VectorXd> e_y(m_y.vector, m_y.vlen);

	e_res=(e_x.row(idx)*e_w-e_y.row(idx))[0]*e_x.row(idx).transpose();
	return res;
}

RegressionForTestCostFunction::RegressionForTestCostFunction()
	:FirstOrderSAGCostFunction()
{
	init();
}

void RegressionForTestCostFunction::init()
{
	m_obj=NULL;
	m_idx=0;
}

RegressionForTestCostFunction::~RegressionForTestCostFunction()
{
	SG_UNREF(m_obj);
}

void RegressionForTestCostFunction::set_target(CRegressionExample *obj)
{
	if(m_obj!=obj)
	{
		SG_REF(obj);
		SG_UNREF(m_obj);
		m_obj=obj;
	}
}

float64_t RegressionForTestCostFunction::get_cost()
{
	REQUIRE(m_obj,"target must set\n");
	return m_obj->get_cost();
}

int32_t RegressionForTestCostFunction::get_sample_size()
{
	REQUIRE(m_obj,"target must set\n");
	return m_obj->get_sample_size();
}

SGVector<float64_t> RegressionForTestCostFunction::get_average_gradient()
{
	REQUIRE(m_obj,"object not set\n");
	CMap<TParameter*, CSGObject*>* parameters=new CMap<TParameter*, CSGObject*>();
	m_obj->build_gradient_parameter_dictionary(parameters);

	index_t num_gradients=parameters->get_num_elements();
	SGVector<float64_t> grad;
	for(index_t i=0; i<num_gradients; i++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameters->get_node_ptr(i);
		if(node && node->data==m_obj)
		{
			grad=m_obj->get_gradient(node->key);
			for(index_t idx=0; idx<grad.vlen; idx++)
				grad[idx]/=get_sample_size();
		}
	}

	SG_UNREF(parameters);
	return grad;
}


void RegressionForTestCostFunction::begin_sample()
{
	REQUIRE(m_obj,"target must set\n");
	m_idx=-1;
}

bool RegressionForTestCostFunction::next_sample()
{
	REQUIRE(m_obj,"target must set\n");
	m_idx++;
	return m_idx<m_obj->get_sample_size();
}

SGVector<float64_t> RegressionForTestCostFunction::obtain_variable_reference()
{
	REQUIRE(m_obj,"object not set\n");
	CMap<TParameter*, CSGObject*>* parameters=new CMap<TParameter*, CSGObject*>();
	m_obj->build_gradient_parameter_dictionary(parameters);
	index_t num_variables=parameters->get_num_elements();
	SGVector<float64_t> variable;
	for(index_t idx=0; idx<num_variables; idx++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameters->get_node_ptr(idx);
		if(node && node->data==m_obj)
			variable=m_obj->get_variable(node->key);
	}

	SG_UNREF(parameters);
	return variable;
}

SGVector<float64_t> RegressionForTestCostFunction::get_gradient()
{
	REQUIRE(m_obj,"object not set\n");
	CMap<TParameter*, CSGObject*>* parameters=new CMap<TParameter*, CSGObject*>();
	m_obj->build_gradient_parameter_dictionary(parameters);

	index_t num_gradients=parameters->get_num_elements();
	SGVector<float64_t> grad;

	for(index_t idx=0; idx<num_gradients; idx++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameters->get_node_ptr(idx);
		if(node && node->data==m_obj)
			grad=m_obj->get_gradient(node->key, m_idx);
	}

	SG_UNREF(parameters);
	return grad;
}

void ClassificationForTestCostFunction::set_sample_sequences(SGVector<int32_t> index, index_t num_sequences)
{
	REQUIRE(index.vlen>0,"");
	REQUIRE(num_sequences>0,"");
	REQUIRE(index.vlen%num_sequences==0,"");
	m_sample_sequences=index;
	m_num_sequences=num_sequences;
}
void ClassificationForTestCostFunction::begin_sample()
{
	is_begin=true;
}

bool ClassificationForTestCostFunction::next_sample()
{
	if (m_call_times%m_num_sequences==0)
	{
		if(!is_begin)
			return false;
		else
			is_begin=false;
	}
	ASSERT(m_call_times<m_sample_sequences.vlen);
	m_sample_idx=m_sample_sequences[m_call_times];
	m_call_times++;
	return true;
}

int32_t ClassificationForTestCostFunction::get_sample_size()
{
	return m_labels.vlen;
}

SGVector<float64_t> ClassificationForTestCostFunction::get_gradient()
{
	SGVector<float64_t> result(m_weight.vlen);
	Map<VectorXd> e_r(result.vector,result.vlen);

	Map<VectorXd> e_w(m_weight.vector,m_weight.vlen);
	Map<MatrixXd> e_x(m_features.matrix, m_features.num_rows, m_features.num_cols);
	float64_t tmp=e_w.dot(e_x.col(m_sample_idx));
	tmp=exp(tmp*m_labels[m_sample_idx]);
	float64_t w=m_labels[m_sample_idx]*tmp / (1.0+tmp);
	e_r=w*e_x.col(m_sample_idx);
	return result;
}

SGVector<float64_t> ClassificationForTestCostFunction::get_average_gradient()
{
	SGVector<float64_t> result(m_weight.vlen);
	result.set_const(0.0);
	Map<VectorXd> e_r(result.vector,result.vlen);

	Map<VectorXd> e_w(m_weight.vector,m_weight.vlen);
	Map<MatrixXd> e_x(m_features.matrix, m_features.num_rows, m_features.num_cols);
	for(index_t idx=0; idx<m_labels.vlen; idx++)
	{
		float64_t w = 1.0-(1.0/(exp(m_labels[idx]*e_w.dot(e_x.col(idx)))+1.0));
		w= w*m_labels[idx]/m_labels.vlen;
		e_r+=w*e_x.col(idx);
	}
	return result;
}

SGVector<float64_t> ClassificationForTestCostFunction::obtain_variable_reference()
{
	return m_weight;
}

ClassificationForTestCostFunction::ClassificationForTestCostFunction()
{
	init();
}

void ClassificationForTestCostFunction::init()
{
	m_sample_idx=0;
	m_call_times=0;
	m_labels=SGVector<float64_t>();
	m_features=SGMatrix<float64_t>();
	m_weight=SGVector<float64_t>();
}

void ClassificationForTestCostFunction::set_data(SGMatrix<float64_t> features, SGVector<float64_t> labels)
{
	REQUIRE(labels.vlen==features.num_cols,"");
	m_labels=labels;
	m_features=features;
	m_weight=SGVector<float64_t>(features.num_rows);
	m_weight.set_const(0.0);
}

float64_t ClassificationForTestCostFunction::get_cost()
{
	REQUIRE(m_labels.vlen>0,"");
	Map<VectorXd> e_w(m_weight.vector,m_weight.vlen);
	Map<MatrixXd> e_x(m_features.matrix, m_features.num_rows, m_features.num_cols);
	float64_t cost=0.0;
	for(index_t idx=0; idx<m_labels.vlen; idx++)
	{
		cost+=log(exp(m_labels[idx]*e_w.dot(e_x.col(idx)))+1.0);
	}
	return cost;
}


void ClassificationForTestCostFunction2::begin_sample()
{
	m_sample_idx=-1;
}

bool ClassificationForTestCostFunction2::next_sample()
{
	m_sample_idx++;
	if(m_sample_idx>=m_labels.vlen)
		return false;
	return true;
}

struct ClassificationFixture
{
	ClassificationFixture(){init();}
	SGVector<float64_t> y;
	SGMatrix<float64_t> x;
	void init();
};

void ClassificationFixture::init()
{
	y=SGVector<float64_t>(20);
	x=SGMatrix<float64_t>(2,20);
	x(0,0)=-0.731271511775; x(1,0)=0.694867473874;
	x(0,1)=0.527549237953; x(1,1)=-0.489861948521;
	x(0,2)=-0.00912982581612; x(1,2)=-0.101017870423;
	x(0,3)=0.303185945446; x(1,3)=0.577446702271;
	x(0,4)=-0.812280826452; x(1,4)=-0.943305046956;
	x(0,5)=0.67153020784; x(1,5)=-0.13446586419;
	x(0,6)=0.524560164916; x(1,6)=-0.995787893298;
	x(0,7)=-0.10922561189; x(1,7)=0.443080064682;
	x(0,8)=-0.542475557459; x(1,8)=0.890541391108;
	x(0,9)=0.802854915223; x(1,9)=-0.938820033933;
	x(0,10)=-0.949108278013; x(1,10)=0.082824945587;
	x(0,11)=0.878298325557; x(1,11)=-0.237591524624;
	x(0,12)=-0.566801205739; x(1,12)=-0.155766848835;
	x(0,13)=-0.94191842485; x(1,13)=-0.556616667454;
	x(0,14)=-0.124224812699; x(1,14)=-0.0083755172363;
	x(0,15)=-0.533831099485; x(1,15)=-0.538266916918;
	x(0,16)=-0.420436770819; x(1,16)=-0.957020589468;
	x(0,17)=0.675155951325; x(1,17)=0.112908645305;
	x(0,18)=0.284588725865; x(1,18)=-0.628187468211;
	x(0,19)=0.985086824352; x(1,19)=0.719893057591;
	y[0]=1;
	y[1]=-1;
	y[2]=-1;
	y[3]=1;
	y[4]=1;
	y[5]=-1;
	y[6]=-1;
	y[7]=1;
	y[8]=1;
	y[9]=-1;
	y[10]=1;
	y[11]=-1;
	y[12]=1;
	y[13]=1;
	y[14]=1;
	y[15]=1;
	y[16]=-1;
	y[17]=-1;
	y[18]=-1;
	y[19]=-1;
}


struct RegressionFixture
{
	RegressionFixture() {init();}
	SGVector<float64_t> y;
	SGMatrix<float64_t> x;
	void init();
};

void RegressionFixture::init()
{
	//the data is simulated from y=0.3*x1-1.5*x2+2.0*x3 with the Gaussian noise(mean=0,variance=1.0)
	//where the ground truth w is [0.3,-1.5,2.0]
	//there are 10 samples
	y=SGVector<float64_t> (10);
	x=SGMatrix<float64_t> (10,3);
	x(0,0)=3.18934210549; x(0,1)=-6.36734839959; x(0,2)=3.87646568343;
	x(1,0)=9.71965286623; x(1,1)=-6.49372199537; x(1,2)=7.85930917808;
	x(2,0)=0.590611116182; x(2,1)=-9.78182994856; x(2,2)=8.12538297323;
	x(3,0)=-0.0883632752317; x(3,1)=-7.40468796501; x(3,2)=-3.07220055411;
	x(4,0)=5.18231755738; x(4,1)=4.95152973815; x(4,2)=-9.9870338276;
	x(5,0)=4.26708114291; x(5,1)=7.10165654603; x(5,2)=4.9253650409;
	x(6,0)=8.18854988953; x(6,1)=2.12501081402; x(6,2)=0.879753850301;
	x(7,0)=5.5827306913; x(7,1)=8.77507993975; x(7,2)=9.94739194247;
	x(8,0)=6.51074058756; x(8,1)=9.82537500991; x(8,2)=-8.74331697256;
	x(9,0)=-7.26254338011; x(9,1)=-9.73370985632; x(9,2)=8.32055931886;
	y[0]=17.826341;
	y[1]=28.947688;
	y[2]=32.482436;
	y[3]=5.475718;
	y[4]=-26.082733;
	y[5]=0.645608;
	y[6]=1.794406;
	y[7]=9.251004;
	y[8]=-31.176166;
	y[9]=30.801085;
}

TEST(SGDMinimizer,test1)
{
	SGVector<float64_t> w(3);
	//set init value of w to be estimated
	w.set_const(0.0);

	RegressionFixture data;
	CRegressionExample* aa=new CRegressionExample();

	aa->set_x(data.x);
	aa->set_y(data.y);
	aa->set_init_w(w);
	RegressionForTestCostFunction *fun=new RegressionForTestCostFunction();
	fun->set_target(aa);

	SGDMinimizer* opt=new SGDMinimizer(fun);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.01);

	GradientDescendUpdater* updater=new GradientDescendUpdater();
	opt->set_gradient_updater(updater);
	opt->set_learning_rate(rate);

	int32_t num_passes=20;
	opt->set_number_passes(num_passes);

	float64_t cost=opt->minimize()/data.y.vlen;

	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//note that the cost function used in svrg from regression is
	//0.5 \times \sum_i{(y_i-x_i'*w)^2}/n, where n is the samples size
	//In this implementation, the cost function is 0.5 \times \sum_i{(y_i-x_i'*w)^2}
	//cost=0.491198269864709
	EXPECT_NEAR(cost,0.491198269864709, 1e-10);

	delete opt;
}

TEST(SGDMinimizer,test2)
{
	SGVector<float64_t> w(3);
	//set init value of w to be estimated
	w.set_const(0.0);

	RegressionFixture data;
	CRegressionExample* aa=new CRegressionExample();
	aa->set_x(data.x);
	aa->set_y(data.y);
	aa->set_init_w(w);
	RegressionForTestCostFunction *fun=new RegressionForTestCostFunction();
	fun->set_target(aa);

	SGDMinimizer opt(fun);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.01);

	GradientDescendUpdater* updater=new GradientDescendUpdater();
	opt.set_gradient_updater(updater);
	opt.set_learning_rate(rate);

	int32_t num_passes=20;
	opt.set_number_passes(num_passes);
	opt.minimize();

	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//w=
	//0.425694
	//-1.550272
	//2.131146
	EXPECT_NEAR(w[0],0.425694,1e-6);
	EXPECT_NEAR(w[1],-1.550272,1e-6);
	EXPECT_NEAR(w[2],2.131146,1e-6);
}

TEST(SGDMinimizer,test3)
{
	SGVector<float64_t> w(3);
	//set init value of w to be estimated
	w.set_const(0.0);

	RegressionFixture data;
	CRegressionExample* aa=new CRegressionExample();
	aa->set_x(data.x);
	aa->set_y(data.y);
	aa->set_init_w(w);
	RegressionForTestCostFunction *fun=new RegressionForTestCostFunction();
	fun->set_target(aa);

	SGDMinimizer* opt=new SGDMinimizer(fun);
	opt->set_penalty_weight(1.0);
	L2Penalty* penalty_type=new L2Penalty();
	opt->set_penalty_type(penalty_type);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.0001);

	GradientDescendUpdater* updater=new GradientDescendUpdater();
	MomentumCorrection* momentum_correction=new StandardMomentumCorrection();
	momentum_correction->set_correction_weight(0.9);
	updater->set_descend_correction(momentum_correction);

	opt->set_learning_rate(rate);
	opt->set_gradient_updater(updater);

	int32_t num_passes=20;
	opt->set_number_passes(num_passes);

	opt->minimize();
	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//w=
	//0.289383
	//-1.513633
	//2.045611
	EXPECT_NEAR(w[0],0.289383,1e-6);
	EXPECT_NEAR(w[1],-1.513633,1e-6);
	EXPECT_NEAR(w[2],2.045611,1e-6);

	delete opt;
}

TEST(SGDMinimizer,test4)
{
	SGVector<float64_t> w(3);
	//set init value of w to be estimated
	w.set_const(0.0);

	RegressionFixture data;
	CRegressionExample* aa=new CRegressionExample();
	aa->set_x(data.x);
	aa->set_y(data.y);
	aa->set_init_w(w);
	RegressionForTestCostFunction *fun=new RegressionForTestCostFunction();
	fun->set_target(aa);

	SGDMinimizer* opt=new SGDMinimizer(fun);
	opt->set_penalty_weight(1.0);
	L2Penalty* penalty_type=new L2Penalty();
	opt->set_penalty_type(penalty_type);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.0001);

	GradientDescendUpdater* updater=new GradientDescendUpdater();
	MomentumCorrection* momentum_correction=new StandardMomentumCorrection();
	momentum_correction->set_correction_weight(0.9);
	updater->set_descend_correction(momentum_correction);
	opt->set_gradient_updater(updater);
	opt->set_learning_rate(rate);

	int32_t num_passes=20;
	opt->set_number_passes(num_passes);

	float64_t cost=opt->minimize();
	cost=(cost-aa->get_cost())+aa->get_cost()/data.y.vlen;

	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//note that the L2 penalized cost function used in svrg from regression is
	//0.5 \times \sum_i{(y_i-x_i'*w)^2}/n + 0.5*w'*w, where n is the samples size
	//In this implementation, the cost function is 0.5 \times \sum_i{(y_i-x_i'*w)^2}+0.5*w'*w
	//cost=3.48852246851037
	EXPECT_NEAR(cost,3.48852246851037, 1e-10);

	delete opt;
}

TEST(SGDMinimizer,test5)
{
	SGVector<float64_t> w(3);
	//set init value of w to be estimated
	w.set_const(0.0);

	RegressionFixture data;
	CRegressionExample* aa=new CRegressionExample();
	aa->set_x(data.x);
	aa->set_y(data.y);
	aa->set_init_w(w);
	RegressionForTestCostFunction *fun=new RegressionForTestCostFunction();
	fun->set_target(aa);

	SGDMinimizer* opt=new SGDMinimizer(fun);
	opt->set_penalty_weight(1.0);
	L2Penalty* penalty_type=new L2Penalty();
	opt->set_penalty_type(penalty_type);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.001);

	GradientDescendUpdater* updater=new GradientDescendUpdater();
	MomentumCorrection* momentum_correction=new StandardMomentumCorrection();
	momentum_correction->set_correction_weight(0.9);
	updater->set_descend_correction(momentum_correction);

	opt->set_learning_rate(rate);
	opt->set_gradient_updater(updater);
	opt->set_number_passes(20);

	float64_t cost=opt->minimize();
	cost=(cost-aa->get_cost())+aa->get_cost()/10;
	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//note that the L2 penalized cost function used in svrg from regression is
	//0.5 \times \sum_i{(y_i-x_i'*w)^2}/n + 0.5*w'*w, where n is the samples size
	//In this implementation, the cost function is 0.5 \times \sum_i{(y_i-x_i'*w)^2}+0.5*w'*w
	//cost is return by going through the data 20 times
	//cost=8.54011254349676
	EXPECT_NEAR(cost,8.54011254349676, 1e-10);
	delete opt;
}

TEST(SVRGMinimizer,test1)
{
	SGVector<float64_t> w(3);
	//set init value of w to be estimated
	w.set_const(0.0);

	RegressionFixture data;
	CRegressionExample* aa=new CRegressionExample();
	aa->set_x(data.x);
	aa->set_y(data.y);
	aa->set_init_w(w);
	RegressionForTestCostFunction *fun=new RegressionForTestCostFunction();
	fun->set_target(aa);

	SVRGMinimizer* opt=new SVRGMinimizer(fun);
	//add L2 penalty
	L2Penalty* penalty_type=new L2Penalty();
	opt->set_penalty_type(penalty_type);
	//add penalty weight
	opt->set_penalty_weight(1.0);

	//uss const learning rate
	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.001);

	//using momentum method
	GradientDescendUpdater* updater=new GradientDescendUpdater();
	MomentumCorrection* momentum_correction=new StandardMomentumCorrection();
	//momentum=0.9
	momentum_correction->set_correction_weight(0.9);
	updater->set_descend_correction(momentum_correction);

	opt->set_learning_rate(rate);
	opt->set_gradient_updater(updater);
	opt->set_number_passes(10);
	opt->set_sgd_number_passes(2);
	opt->set_average_update_interval(2);
	opt->minimize();

	//the result is from the svrg software using svrg method with momentum
	//http://riejohnson.com/svrg_download.html
	//w=
	//0.5180193923
	//-1.0194184559
	//1.6913053544
	EXPECT_NEAR(w[0],0.5180193923,1e-8);
	EXPECT_NEAR(w[1],-1.0194184559,1e-8);
	EXPECT_NEAR(w[2],1.6913053544,1e-8);

	delete opt;
}

TEST(SVRGMinimizer,test2)
{
	//We fix the sample sequences
	//Note that we a pass of going through a data set can be called a sample sequence
	//We generalize the definition of "sample sequence". A sample sequence can be a subset of the data set.
	SGVector<int32_t> seq(25);

	//there are 5 sample sequences, each of which has 5 data points
	seq[0]=10;
	seq[1]=18;
	seq[2]=15;
	seq[3]=5;
	seq[4]=4;

	seq[5]=13;
	seq[6]=3;
	seq[7]=7;
	seq[8]=12;
	seq[9]=8;

	seq[10]=0;
	seq[11]=17;
	seq[12]=17;
	seq[13]=6;
	seq[14]=11;

	seq[15]=19;
	seq[16]=16;
	seq[17]=1;
	seq[18]=5;
	seq[19]=5;

	seq[20]=16;
	seq[21]=12;
	seq[22]=2;
	seq[23]=0;
	seq[24]=18;

	ClassificationFixture data;
	ClassificationForTestCostFunction* bb=new ClassificationForTestCostFunction();
	bb->set_data(data.x, data.y);
	//there are 5 sample sequences
	bb->set_sample_sequences(seq, 5);

	SVRGMinimizer* opt=new SVRGMinimizer(bb);
	opt->set_penalty_weight(1.0/data.y.vlen);
	L2Penalty* penalty_type=new L2Penalty();
	opt->set_penalty_type(penalty_type);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(1.758619054751211);
	GradientDescendUpdater* updater=new GradientDescendUpdater();
	opt->set_gradient_updater(updater);
	//since there are 5 samples sequences, we set the number of passes is 5
	opt->set_number_passes(5);
	opt->set_learning_rate(rate);
	opt->set_sgd_number_passes(0);
	opt->set_average_update_interval(1);

	//result from the Semi Stochastic Gradient Descent 1.0 (S2GD) package
	//http://mloss.org/software/view/556/
	//w[0]=1.840662
	//w[1]=-0.855683
	//cost=0.334054
	opt->minimize();
	float64_t cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.334054,1e-6);

	SGVector<float64_t> w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],1.840662,1e-6);
	EXPECT_NEAR(w[1],-0.855683,1e-6);

	delete opt;
}

TEST(AdaDeltaUpdater, test1)
{
	ClassificationFixture data;
	ClassificationForTestCostFunction2* bb=new ClassificationForTestCostFunction2();
	bb->set_data(data.x, data.y);

	SGDMinimizer* opt=new SGDMinimizer(bb);
	AdaDeltaUpdater* updater=new AdaDeltaUpdater();
	updater->set_learning_rate(1.8);
	updater->set_epsilon(1e-6);
	updater->set_decay_factor(0.95);
	MomentumCorrection* momentum_correction=new NesterovMomentumCorrection();
	momentum_correction->set_correction_weight(0.99);
	updater->set_descend_correction(momentum_correction);

	opt->set_gradient_updater(updater);
	opt->set_number_passes(1);

	//The reference result is from https://github.com/BRML/climin
	//w=
	//9.322789137268 -5.041678838144
	//cost=
	//0.0999328675763
	opt->minimize();
	float64_t cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.0999328675763,1e-10);
	SGVector<float64_t> w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],9.322789137268,1e-10);
	EXPECT_NEAR(w[1],-5.041678838144,1e-10);

	opt->set_number_passes(1);
	opt->minimize();
	//The reference result is from https://github.com/BRML/climin
	//w=
	//37.005870439442 -10.868474774524
	//cost=
	//0.52542975268
	cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.52542975268,1e-10);
	w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],37.005870439442,1e-10);
	EXPECT_NEAR(w[1],-10.868474774524,1e-10);

	delete opt;
}

TEST(AdamUpdater, test1)
{
	ClassificationFixture data;
	ClassificationForTestCostFunction2* bb=new ClassificationForTestCostFunction2();
	bb->set_data(data.x, data.y);

	SGDMinimizer* opt2=new SGDMinimizer(bb);
	AdamUpdater* updater2=new AdamUpdater(0.1, 1e-8, 0.9, 0.999);

	opt2->set_gradient_updater(updater2);
	opt2->set_number_passes(1);
	opt2->minimize();
	//The reference result is from tensorflow
	//w=
	//1.49674  -0.88859
	//cost=
	//0.368617
	float64_t cost=bb->get_cost()/(float64_t)bb->get_sample_size();
	EXPECT_NEAR(cost,0.368617,1e-5);
	SGVector<float64_t> w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],1.49674,1e-5);
	EXPECT_NEAR(w[1],-0.88859,1e-5);

	delete opt2;
}


TEST(AdaptMomentumCorrection, test1)
{
	ClassificationFixture data;
	ClassificationForTestCostFunction2* bb=new ClassificationForTestCostFunction2();
	bb->set_data(data.x, data.y);

	SGDMinimizer* opt=new SGDMinimizer(bb);
	RmsPropUpdater* updater=new RmsPropUpdater();
	updater->set_learning_rate(1.0);
	updater->set_epsilon(0.0);
	updater->RmsPropUpdater::set_decay_factor(0.9);
	MomentumCorrection* momentum_correction=new NesterovMomentumCorrection();
	momentum_correction->set_correction_weight(0.5);
	AdaptMomentumCorrection* correction=new AdaptMomentumCorrection();
	correction->set_momentum_correction(momentum_correction);
	correction->set_init_descend_rate(0.9);
	correction->set_adapt_rate(0.1, 0.05, 2.0);
	updater->set_descend_correction(correction);
	opt->set_gradient_updater(updater);
	opt->set_number_passes(1);

	//The reference result is from https://github.com/BRML/climin
	//w=
	//13.014770067941 -6.735459931220
	//cost=
	//0.0874683992352
	opt->minimize();
	float64_t cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.0874683992352,1e-10);
	SGVector<float64_t> w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],13.014770067941,1e-10);
	EXPECT_NEAR(w[1],-6.735459931220,1e-10);


	opt->set_number_passes(1);
	opt->minimize();
	//The reference result is from https://github.com/BRML/climin
	//w=
	//13.898448669614 -11.852199405401
	//cost=
	//0.0750873577032
	cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.0750873577032,1e-10);
	w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],13.898448669614,1e-10);
	EXPECT_NEAR(w[1],-11.852199405401,1e-10);

	delete opt;
}

TEST(L1PenaltyForTG, test1)
{
	ClassificationFixture data;
	ClassificationForTestCostFunction2* bb=new ClassificationForTestCostFunction2();
	bb->set_data(data.x, data.y);
	SGDMinimizer* opt=new SGDMinimizer(bb);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.01);
	GradientDescendUpdater* updater=new GradientDescendUpdater();
	opt->set_gradient_updater(updater);

	opt->set_penalty_weight(0.01);
	L1Penalty* penalty_type=new L1PenaltyForTG();

	penalty_type->set_rounding_epsilon(0);
	opt->set_penalty_type(penalty_type);
	opt->set_number_passes(1);
	opt->set_learning_rate(rate);

	opt->minimize();
	//the loss in the reference program is log(1+exp(-y*w*x))
	//However, the loss in our implementation is log(1+exp(y*w*x))
	//
	//result from sklearn.linear_model.SGDClassifier (v.0.16.1)
	//w=
	//-0.047225925298
	//0.018566844801
	//
	//cost=
	//0.679635661120
	float64_t cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.679635661120,1e-10);
	SGVector<float64_t> w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],0.047225925298,1e-10);
	EXPECT_NEAR(w[1],-0.018566844801,1e-10);

	delete opt;
}

TEST(L1PenaltyForTG, test2)
{
	ClassificationFixture data;
	ClassificationForTestCostFunction2* bb=new ClassificationForTestCostFunction2();
	bb->set_data(data.x, data.y);
	SGDMinimizer* opt=new SGDMinimizer(bb);

	InverseScalingLearningRate* rate= new InverseScalingLearningRate();
	rate->set_initial_learning_rate(0.1);
	rate->set_exponent(0.6);
	rate->set_slope(1.0);
	rate->set_intercept(0.0);

	GradientDescendUpdater* updater=new GradientDescendUpdater();
	opt->set_gradient_updater(updater);

	opt->set_penalty_weight(0.01);
	L1Penalty* penalty_type=new L1PenaltyForTG();
	penalty_type->set_rounding_epsilon(0);

	opt->set_penalty_type(penalty_type);
	opt->set_number_passes(2);
	opt->set_learning_rate(rate);
	opt->minimize();

	//the loss in the reference program is log(1+exp(-y*w*x))
	//However, the loss in our implementation is log(1+exp(y*w*x))
	//
	//result from sklearn.linear_model.SGDClassifier (v.0.16.1)
	//w=
	//-0.203765743995
	//0.109255680811
	//cost=
	//0.633950836133
	//
	float64_t cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.633950836133,1e-10);
	SGVector<float64_t> w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],0.203765743995,1e-10);
	EXPECT_NEAR(w[1],-0.109255680811,1e-10);

	delete opt;
}

TEST(ElasticNetPenalty, test1)
{
	ClassificationFixture data;
	ClassificationForTestCostFunction2* bb=new ClassificationForTestCostFunction2();
	bb->set_data(data.x, data.y);
	SGDMinimizer* opt=new SGDMinimizer(bb);

	InverseScalingLearningRate* rate= new InverseScalingLearningRate();
	rate->set_initial_learning_rate(0.1);
	rate->set_exponent(0.6);
	rate->set_slope(1.0);
	rate->set_intercept(0.0);

	GradientDescendUpdater* updater=new GradientDescendUpdater();
	opt->set_gradient_updater(updater);

	opt->set_penalty_weight(0.01);
	ElasticNetPenalty* penalty_type=new ElasticNetPenalty();
	penalty_type->set_l1_ratio(0.7);

	opt->set_penalty_type(penalty_type);
	opt->set_number_passes(2);
	opt->set_learning_rate(rate);
	opt->minimize();

	//the loss in the reference program is log(1+exp(-y*w*x))
	//However, the loss in our implementation is log(1+exp(y*w*x))
	//
	//result from sklearn.linear_model.SGDClassifier (v.0.16.1)
	//
	//w[0]=-0.206101230857 w[1]=0.111680271395
	//total loss=0.633194272101
	float64_t cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.633194272101,1e-10);
	SGVector<float64_t> w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],0.206101230857,1e-10);
	EXPECT_NEAR(w[1],-0.111680271395,1e-10);

	delete opt;
}

TEST(SMIDASMinimizer, test1)
{
	ClassificationFixture data;
	ClassificationForTestCostFunction2* bb=new ClassificationForTestCostFunction2();
	bb->set_data(data.x, data.y);
	SMIDASMinimizer* opt=new SMIDASMinimizer(bb);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.01);
	GradientDescendUpdater* updater=new GradientDescendUpdater();
	opt->set_gradient_updater(updater);
	PNormMappingFunction* mapping=new PNormMappingFunction();
	mapping->set_norm(2.0);
	opt->set_mapping_function(mapping);

	opt->set_penalty_weight(0.01);
	L1Penalty* penalty_type=new L1Penalty();
	penalty_type->set_rounding_epsilon(1e-8);
	opt->set_penalty_type(penalty_type);
	opt->set_number_passes(2);
	opt->set_learning_rate(rate);
	opt->minimize();

	//the loss in the reference program is log(1+exp(-y*w*x))
	//However, the loss in our implementation is log(1+exp(y*w*x))
	//
	//reference result from http://mloss.org/software/view/208/
	//
	//w=
	//-0.0934993
	//0.0367823
	//
	//cost=
	//0.666646
	float64_t cost=bb->get_cost()/bb->get_sample_size();
	EXPECT_NEAR(cost,0.666646,1e-6);
	SGVector<float64_t> w=bb->obtain_variable_reference();
	EXPECT_NEAR(w[0],0.0934993,1e-6);
	EXPECT_NEAR(w[1],-0.0367823,1e-6);

	delete opt;
}
