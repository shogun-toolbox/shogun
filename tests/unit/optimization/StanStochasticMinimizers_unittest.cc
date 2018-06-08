/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Wu Lin
 */

#include <shogun/optimization/StanFirstOrderSAGCostFunction.h>
#include <gtest/gtest.h>
#include "StanStochasticMinimizers_unittest.h"
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
using namespace stan::math;
using std::function;

StanVector generateParameters()
{
	StanVector w(3, 1);
	var x1(0), x2(0), x3(0);
	w(0, 0) = x1;
	w(1, 0) = x2;
  w(2, 0) = x3;
	return w;
}

function<var(StanVector*, float64_t)> cost_for_ith_datapoints(
    SGMatrix<float64_t>& X, SGMatrix<float64_t>& y)
{
	auto f_i = [X, y](StanVector* w, int32_t idx) {
		var wx_y = (*w)(0, 0) * X(0, idx) + (*w)(1, 0) * X(1, idx) + (*w)(2,0)*X(2,idx) - y(0, idx);
		var res = wx_y * wx_y;
		res /= 2;
		return res;
	};
	return f_i;
}

function<var(StanVector*)> get_tot_cost()
{
	auto cost = [](StanVector* v) {
		var total_cost = v->sum();
		return total_cost;
	};
	return cost;
}

struct RegressionData
{
	RegressionData() {init();}
	SGMatrix<float64_t> y;
	SGMatrix<float64_t> x;
	void init();
};

void RegressionData::init()
{
  // The data is referenced from Wu Lin Unit tests
	//the data is simulated from y=0.3*x1-1.5*x2+2.0*x3 with the Gaussian noise(mean=0,variance=1.0)
	//where the ground truth w is [0.3,-1.5,2.0]
	//there are 10 samples
	y=SGMatrix<float64_t> (1,10);
	x=SGMatrix<float64_t> (3,10);
	x(0,0)=3.18934210549; x(1,0)=-6.36734839959; x(2,0)=3.87646568343;
	x(0,1)=9.71965286623; x(1,1)=-6.49372199537; x(2,1)=7.85930917808;
	x(0,2)=0.590611116182; x(1,2)=-9.78182994856; x(2,2)=8.12538297323;
	x(0,3)=-0.0883632752317; x(1,3)=-7.40468796501; x(2,3)=-3.07220055411;
	x(0,4)=5.18231755738; x(1,4)=4.95152973815; x(2,4)=-9.9870338276;
	x(0,5)=4.26708114291; x(1,5)=7.10165654603; x(2,5)=4.9253650409;
	x(0,6)=8.18854988953; x(1,6)=2.12501081402; x(2,6)=0.879753850301;
	x(0,7)=5.5827306913; x(1,7)=8.77507993975; x(2,7)=9.94739194247;
	x(0,8)=6.51074058756; x(1,8)=9.82537500991; x(2,8)=-8.74331697256;
	x(0,9)=-7.26254338011; x(1,9)=-9.73370985632; x(2,9)=8.32055931886;
	y(0,0)=17.826341;
	y(0,1)=28.947688;
	y(0,2)=32.482436;
	y(0,3)=5.475718;
	y(0,4)=-26.082733;
	y(0,5)=0.645608;
	y(0,6)=1.794406;
	y(0,7)=9.251004;
	y(0,8)=-31.176166;
	y(0,9)=30.801085;
}

TEST(StanSGDMinimizer,test1)
{
  int32_t n = 10;

  RegressionData data;
  auto X = data.x;
  auto y = data.y;
  auto w = generateParameters();

  auto f_i = cost_for_ith_datapoints(X, y);

  Matrix<function<var(StanVector*, float64_t)>, Dynamic, 1> cost_for_ith_point =
      Matrix<function<var(StanVector*, float64_t)>, Dynamic, 1>::Constant(n, 1, f_i);

  auto total_cost = get_tot_cost();

  SquareErrorTestCostFunction* fun = new SquareErrorTestCostFunction(X,y,&w, &cost_for_ith_point, &total_cost);

  SGDMinimizer* opt=new SGDMinimizer(fun);

  ConstLearningRate* rate=new ConstLearningRate();
	rate->set_const_learning_rate(0.01);

	GradientDescendUpdater* updater=new GradientDescendUpdater();
	opt->set_gradient_updater(updater);
	opt->set_learning_rate(rate);

	int32_t num_passes=20;
	opt->set_number_passes(num_passes);

	float64_t cost=opt->minimize()/n;

  // The data is referenced from Wu Lin Unit tests
	//the result is from the svrg software using plain stochastic gradient descent
	//http://riejohnson.com/svrg_download.html
	//note that the cost function used in svrg from regression is
	//0.5 \times \sum_i{(y_i-x_i'*w)^2}/n, where n is the samples size
	//In this implementation, the cost function is 0.5 \times \sum_i{(y_i-x_i'*w)^2}
	//cost=0.491198269864709
	EXPECT_NEAR(cost,0.491198269864709, 1e-10);

  delete opt;
}
