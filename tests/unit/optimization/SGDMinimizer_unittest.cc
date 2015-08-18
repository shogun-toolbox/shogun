/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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
 *
 */

#include "SGDMinimizer_unittest.h"
#ifdef HAVE_EIGEN3
#include <shogun/optimization/L2Penalty.h>
#include <shogun/optimization/FirstOrderStochasticCostFunction.h>
#include <shogun/optimization/SGDMinimizer.h>
#include <shogun/optimization/GradientDescendUpdater.h>
#include <shogun/optimization/ConstLearningRate.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/base/Parameter.h>
#include <gtest/gtest.h>
#include <shogun/lib/Map.h>
#include <shogun/optimization/StandardMomentumCorrection.h>
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

SGDTestStochasticCostFunction::SGDTestStochasticCostFunction()
	:FirstOrderStochasticCostFunction()
{
	init();
}

void SGDTestStochasticCostFunction::init()
{
	m_obj=NULL;
	m_idx=0;
}

SGDTestStochasticCostFunction::~SGDTestStochasticCostFunction()
{
	SG_UNREF(m_obj);
}

void SGDTestStochasticCostFunction::set_target(CRegressionExample *obj)
{
	if(m_obj!=obj)
	{
		SG_REF(obj);
		SG_UNREF(m_obj);
		m_obj=obj;
	}
}

float64_t SGDTestStochasticCostFunction::get_cost()
{
	REQUIRE(m_obj,"target must set\n");
	return m_obj->get_cost();
}

void SGDTestStochasticCostFunction::begin_sample()
{
	REQUIRE(m_obj,"target must set\n");
	m_idx=-1;
}

bool SGDTestStochasticCostFunction::next_sample()
{
	REQUIRE(m_obj,"target must set\n");
	m_idx++;
	return m_idx<m_obj->get_sample_size();
}

SGVector<float64_t> SGDTestStochasticCostFunction::obtain_variable_reference()
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

SGVector<float64_t> SGDTestStochasticCostFunction::get_gradient()
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

TEST(SGDMinimizer,test1)
{
	SGVector<float64_t> w(3);
	SGVector<float64_t> y(10);
	SGMatrix<float64_t> x(10,3);
	//the data is simulated from y=0.3*x1-1.5*x2+2.0*x3 with the Gaussian noise(mean=0,variance=1.0)
	//where the ground truth w is [0.3,-1.5,2.0]
	//there are 10 samples

	//set init value of w to be estimated
	w.set_const(0.0);

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

	CRegressionExample* aa=new CRegressionExample();

	aa->set_x(x);
	aa->set_y(y);
	aa->set_init_w(w);
	SGDTestStochasticCostFunction *fun=new SGDTestStochasticCostFunction();
	fun->set_target(aa);

	SGDMinimizer* opt=new SGDMinimizer(fun);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.01);

	GradientDescendUpdater* updater=new GradientDescendUpdater(rate);
	opt->set_gradient_updater(updater);

	int32_t num_passes=20;
	opt->set_number_passes(num_passes);

	float64_t cost=opt->minimize()/y.vlen;

	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//note that the cost function used in svrg from regression is
	//0.5 \times \sum_i{(y_i-x_i'*w)^2}/n, where n is the samples size
	//In this implementation, the cost function is 0.5 \times \sum_i{(y_i-x_i'*w)^2}
	//cost=0.491198269864709
	EXPECT_NEAR(cost,0.491198269864709, 1e-10);
		
	delete opt;
	delete fun;
	delete rate;
	delete updater;
}

TEST(SGDMinimizer,test2)
{
	SGVector<float64_t> w(3);
	SGVector<float64_t> y(10);
	SGMatrix<float64_t> x(10,3);
	//set init value of w to be estimated
	w.set_const(0.0);

	//the data is simulated from y=0.3*x1-1.5*x2+2.0*x3 with the Gaussian noise(mean=0,variance=1.0)
	//where the ground truth w is [0.3,-1.5,2.0]
	//there are 10 samples
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

	CRegressionExample* aa=new CRegressionExample();

	aa->set_x(x);
	aa->set_y(y);
	aa->set_init_w(w);
	SGDTestStochasticCostFunction *fun=new SGDTestStochasticCostFunction();
	fun->set_target(aa);

	SGDMinimizer opt(fun);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.01);

	GradientDescendUpdater* updater=new GradientDescendUpdater(rate);
	opt.set_gradient_updater(updater);

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

	delete fun;
	delete rate;
	delete updater;
}

TEST(SGDMinimizer,test3)
{
	SGVector<float64_t> w(3);
	SGVector<float64_t> y(10);
	SGMatrix<float64_t> x(10,3);
	w.set_const(0.0);
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

	CRegressionExample* aa=new CRegressionExample();

	aa->set_x(x);
	aa->set_y(y);
	aa->set_init_w(w);
	SGDTestStochasticCostFunction *fun=new SGDTestStochasticCostFunction();
	fun->set_target(aa);

	SGDMinimizer* opt=new SGDMinimizer(fun);
	opt->set_penalty_weight(1.0);
	L2Penalty* penalty_type=new L2Penalty();
	opt->set_penalty_type(penalty_type);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.0001);

	GradientDescendUpdater* updater=new GradientDescendUpdater(rate);
	MomentumCorrection* momentum_correction=new StandardMomentumCorrection();
	momentum_correction->set_correction_weight(0.9);
	updater->set_descend_correction(momentum_correction);

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
	delete fun;
	delete rate;
	delete penalty_type;
	delete updater;
	delete momentum_correction;
}

TEST(SGDMinimizer,test4)
{
	SGVector<float64_t> w(3);
	SGVector<float64_t> y(10);
	SGMatrix<float64_t> x(10,3);
	w.set_const(0.0);
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

	CRegressionExample* aa=new CRegressionExample();

	aa->set_x(x);
	aa->set_y(y);
	aa->set_init_w(w);
	SGDTestStochasticCostFunction *fun=new SGDTestStochasticCostFunction();
	fun->set_target(aa);

	SGDMinimizer* opt=new SGDMinimizer(fun);
	opt->set_penalty_weight(1.0);
	L2Penalty* penalty_type=new L2Penalty();
	opt->set_penalty_type(penalty_type);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.0001);

	GradientDescendUpdater* updater=new GradientDescendUpdater(rate);
	MomentumCorrection* momentum_correction=new StandardMomentumCorrection();
	momentum_correction->set_correction_weight(0.9);
	updater->set_descend_correction(momentum_correction);
	opt->set_gradient_updater(updater);

	int32_t num_passes=20;
	opt->set_number_passes(num_passes);

	float64_t cost=opt->minimize();
	cost=(cost-aa->get_cost())+aa->get_cost()/y.vlen;

	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//note that the L2 penalized cost function used in svrg from regression is
	//0.5 \times \sum_i{(y_i-x_i'*w)^2}/n + 0.5*w'*w, where n is the samples size
	//In this implementation, the cost function is 0.5 \times \sum_i{(y_i-x_i'*w)^2}+0.5*w'*w
	//cost=3.48852246851037
	EXPECT_NEAR(cost,3.48852246851037, 1e-10);

	delete opt;
	delete fun;
	delete rate;
	delete penalty_type;
	delete updater;
	delete momentum_correction;
}

TEST(SGDMinimizer,test5)
{
	SGVector<float64_t> w(3);
	SGVector<float64_t> y(10);
	SGMatrix<float64_t> x(10,3);
	w.set_const(0.0);
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

	CRegressionExample* aa=new CRegressionExample();

	aa->set_x(x);
	aa->set_y(y);
	aa->set_init_w(w);
	SGDTestStochasticCostFunction *fun=new SGDTestStochasticCostFunction();
	fun->set_target(aa);

	SGDMinimizer* opt=new SGDMinimizer(fun);
	opt->set_penalty_weight(1.0);
	L2Penalty* penalty_type=new L2Penalty();
	opt->set_penalty_type(penalty_type);

	ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.001);

	GradientDescendUpdater* updater=new GradientDescendUpdater(rate);
	MomentumCorrection* momentum_correction=new StandardMomentumCorrection();
	momentum_correction->set_correction_weight(0.9);
	updater->set_descend_correction(momentum_correction);

	opt->set_gradient_updater(updater);
	opt->set_number_passes(5);

	float64_t cost=opt->minimize();
	CMinimizerContext* context=opt->save_to_context();
	delete opt;
	delete updater;
	delete momentum_correction;
	delete rate;

	SGDMinimizer* opt2=new SGDMinimizer(fun);
	opt2->set_penalty_weight(1.0);
	opt2->set_penalty_type(penalty_type);

	ConstLearningRate* rate2=new ConstLearningRate();
	rate2->set_const_learning_rate(0.001);
	GradientDescendUpdater* updater2=new GradientDescendUpdater(rate2);
	MomentumCorrection* momentum_correction2=new StandardMomentumCorrection();
	momentum_correction2->set_correction_weight(0.9);
	updater2->set_descend_correction(momentum_correction2);
	opt2->set_gradient_updater(updater2);

	opt2->load_from_context(context);
	opt2->set_number_passes(15);
	delete context;
	cost=opt2->minimize();
	cost=(cost-aa->get_cost())+aa->get_cost()/10;
	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//note that the L2 penalized cost function used in svrg from regression is
	//0.5 \times \sum_i{(y_i-x_i'*w)^2}/n + 0.5*w'*w, where n is the samples size
	//In this implementation, the cost function is 0.5 \times \sum_i{(y_i-x_i'*w)^2}+0.5*w'*w
	//cost is return by going through the data 20 times
	//cost=8.54011254349676
	EXPECT_NEAR(cost,8.54011254349676, 1e-10);
	delete rate2;
	delete updater2;
	delete momentum_correction2;
	delete fun;
	delete penalty_type;
	delete opt2;
}

#endif /* HAVE_EIGEN3 */
