/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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


// Eigen3 is required for working with this example
#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/base/init.h>
#include <shogun/machine/gp/LogitPiecewiseBoundLikelihood.h>
#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/optimization/lbfgs/lbfgs.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;


SGMatrix<float64_t> init_piecewise_bound()
{
	index_t row = 20;
	index_t col= 5;
	SGMatrix<float64_t> bound(row, col);
	//load('llp.mat'); 
	bound(0,0)=0.000188712193000;
	bound(1,0)=0.028090310300000;
	bound(2,0)=0.110211757000000;
	bound(3,0)=0.232736440000000;
	bound(4,0)=0.372524706000000;
	bound(5,0)=0.504567936000000;
	bound(6,0)=0.606280283000000;
	bound(7,0)=0.666125432000000;
	bound(8,0)=0.689334264000000;
	bound(9,0)=0.693147181000000;
	bound(10,0)=0.693147181000000;
	bound(11,0)=0.689334264000000;
	bound(12,0)=0.666125432000000;
	bound(13,0)=0.606280283000000;
	bound(14,0)=0.504567936000000;
	bound(15,0)=0.372524706000000;
	bound(16,0)=0.232736440000000;
	bound(17,0)=0.110211757000000;
	bound(18,0)=0.028090310400000;
	bound(19,0)=0.000188712000000;

	bound(0,1)=0;
	bound(1,1)=0.006648614600000;
	bound(2,1)=0.034432684600000;
	bound(3,1)=0.088701969900000;
	bound(4,1)=0.168024214000000;
	bound(5,1)=0.264032863000000;
	bound(6,1)=0.360755794000000;
	bound(7,1)=0.439094482000000;
	bound(8,1)=0.485091758000000;
	bound(9,1)=0.499419205000000;
	bound(10,1)=0.500580795000000;
	bound(11,1)=0.514908242000000;
	bound(12,1)=0.560905518000000;
	bound(13,1)=0.639244206000000;
	bound(14,1)=0.735967137000000;
	bound(15,1)=0.831975786000000;
	bound(16,1)=0.911298030000000;
	bound(17,1)=0.965567315000000;
	bound(18,1)=0.993351385000000;
	bound(19,1)=1.000000000000000;

	bound(0,2)=0;
	bound(1,2)=0.000397791059000;
	bound(2,2)=0.002753100850000;
	bound(3,2)=0.008770186980000;
	bound(4,2)=0.020034759300000;
	bound(5,2)=0.037511596000000;
	bound(6,2)=0.060543032900000;
	bound(7,2)=0.086256780600000;
	bound(8,2)=0.109213531000000;
	bound(9,2)=0.123026104000000;
	bound(10,2)=0.123026104000000;
	bound(11,2)=0.109213531000000;
	bound(12,2)=0.086256780600000;
	bound(13,2)=0.060543032900000;
	bound(14,2)=0.037511596000000;
	bound(15,2)=0.020034759300000;
	bound(16,2)=0.008770186980000;
	bound(17,2)=0.002753100850000;
	bound(18,2)=0.000397791059000;
	bound(19,2)=0;

	bound(0,3)=-CMath::INFTY;
	bound(1,3)=-8.575194939999999;
	bound(2,3)=-5.933689180000000;
	bound(3,3)=-4.525933600000000;
	bound(4,3)=-3.528107790000000;
	bound(5,3)=-2.751548540000000;
	bound(6,3)=-2.097898790000000;
	bound(7,3)=-1.519690830000000;
	bound(8,3)=-0.989533382000000;
	bound(9,3)=-0.491473077000000;
	bound(10,3)=0;
	bound(11,3)=0.491473077000000;
	bound(12,3)=0.989533382000000;
	bound(13,3)=1.519690830000000;
	bound(14,3)=2.097898790000000;
	bound(15,3)=2.751548540000000;
	bound(16,3)=3.528107790000000;
	bound(17,3)=4.525933600000000;
	bound(18,3)=5.933689180000000;
	bound(19,3)=8.575194939999999;


	bound(0,4)=-8.575194939999999;
	bound(1,4)=-5.933689180000000;
	bound(2,4)=-4.525933600000000;
	bound(3,4)=-3.528107790000000;
	bound(4,4)=-2.751548540000000;
	bound(5,4)=-2.097898790000000;
	bound(6,4)=-1.519690830000000;
	bound(7,4)=-0.989533382000000;
	bound(8,4)=-0.491473077000000;
	bound(9,4)=0;
	bound(10,4)=0.491473077000000;
	bound(11,4)=0.989533382000000;
	bound(12,4)=1.519690830000000;
	bound(13,4)=2.097898790000000;
	bound(14,4)=2.751548540000000;
	bound(15,4)=3.528107790000000;
	bound(16,4)=4.525933600000000;
	bound(17,4)=5.933689180000000;
	bound(18,4)=8.575194939999999;
	bound(19,4)= CMath::INFTY;
	return bound;
}

SGMatrix<float64_t> create_feature(index_t num_sample, index_t num_dim)
{
	ASSERT(num_sample % 2 == 0);
	SGMatrix<float64_t> X(num_sample,num_dim);
	
	/*
	//X = [5*rand(N/2,D); -5*rand(N/2,D)]; 
	//The following code is used to generate synthetic data
	for(index_t i = 0; i < num_sample; i++)
	{
		for(index_t j = 0; j < num_dim; j++)
		{
			if (i < num_sample/2)
				X(i, j) = CMath::random(0,1)*5.0;
			else
				X(i, j) = CMath::random(0,1)*-5.0;
		}
	}
	*/

	//The following pre-init value is used to verify the correctness
	X(0,0)=2.085110023512870;
	X(0,1)=2.095972572016474;
	X(0,2)=4.003722843377683;
	X(0,3)=0.491734169165250;
	X(0,4)=4.944305444532474;
	X(1,0)=3.601622467210790;
	X(1,1)=3.426097501983798;
	X(1,2)=4.841307878596988;
	X(1,3)=2.105538125025261;
	X(1,4)=3.740828271899197;
	X(2,0)=0.000571874086724;
	X(2,1)=1.022261248657587;
	X(2,2)=1.567120890796214;
	X(2,3)=4.789447650752510;
	X(2,4)=1.402219960322026;
	X(3,0)=1.511662863159199;
	X(3,1)=4.390587181954727;
	X(3,2)=3.461613078346570;
	X(3,3)=2.665826424865085;
	X(3,4)=3.946396642257443;
	X(4,0)=0.733779454085565;
	X(4,1)=0.136937965989631;
	X(4,2)=4.381945761480192;
	X(4,3)=3.459385569752367;
	X(4,4)=0.516130032888210;
	X(5,0)=0.461692973843989;
	X(5,1)=3.352337550892011;
	X(5,2)=4.473033317519237;
	X(5,3)=1.577578155030315;
	X(5,4)=2.239467630879526;
	X(6,0)=0.931301056888355;
	X(6,1)=2.086524011835635;
	X(6,2)=0.425221056848890;
	X(6,3)=3.432504638407918;
	X(6,4)=4.542977515465478;
	X(7,0)=1.727803635215239;
	X(7,1)=2.793449142228758;
	X(7,2)=0.195273916164412;
	X(7,3)=4.173128359486864;
	X(7,4)=1.468070741868397;
	X(8,0)=1.983837371153350;
	X(8,1)=0.701934692976169;
	X(8,2)=0.849152097822845;
	X(8,3)=0.091441386720959;
	X(8,4)=1.438876692931744;
	X(9,0)=2.694083670016785;
	X(9,1)=0.990507445424394;
	X(9,2)=4.390712517147065;
	X(9,3)=3.750721574724837;
	X(9,4)=0.650142860591388;
	X(10,0)=-0.096834789351485;
	X(10,1)=-0.511672144139129;
	X(10,2)=-4.517009576439418;
	X(10,3)=-4.416530456029049;
	X(10,4)=-0.573729864766876;
	X(11,0)=-3.394177664699455;
	X(11,1)=-2.070279939097841;
	X(11,2)=-0.687373520731188;
	X(11,3)=-3.118361035278045;
	X(11,4)=-4.747446293535356;
	X(12,0)=-1.058140580000295;
	X(12,1)=-3.472000788638725;
	X(12,2)=-0.696381736253793;
	X(12,3)=-3.754712170136686;
	X(12,4)=-2.249560667399702;
	X(13,0)=-1.327733296861131;
	X(13,1)=-2.070896347634513;
	X(13,2)=-4.036956443547619;
	X(13,3)=-1.744491709889213;
	X(13,4)=-2.891948071935659;
	X(14,0)=-2.457865796401691;
	X(14,1)=-0.249767294730436;
	X(14,2)=-1.988384184927668;
	X(14,3)=-1.349639458825130;
	X(14,4)=-2.040684013806406;
	X(15,0)=-0.266812725585402;
	X(15,1)=-2.679482029577558;
	X(15,2)=-0.826770985584664;
	X(15,3)=-4.479431090980334;
	X(15,4)=-1.185134901215139;
	X(16,0)=-2.870588027460065;
	X(16,1)=-3.318973226098944;
	X(16,2)=-4.637542901980169;
	X(16,3)=-2.140455949356475;
	X(16,4)=-4.516897602811269;
	X(17,0)=-0.733642874529051;
	X(17,1)=-2.574445560291543;
	X(17,2)=-1.738829298727533;
	X(17,3)=-4.824200235741928;
	X(17,4)=-2.868397433361429;
	X(18,0)=-2.946527684516421;
	X(18,1)=-4.722973779954066;
	X(18,2)=-3.754060515680778;
	X(18,3)=-3.317207489092240;
	X(18,4)=-0.014351635155795;
	X(19,0)=-3.498791800104656;
	X(19,1)=-2.932775202509965;
	X(19,2)=-3.629989926752257;
	X(19,3)=-3.108478601045609;
	X(19,4)=-3.085724568103619;
	return X;
}


SGVector<float64_t> create_label(SGVector<float64_t> mu, SGMatrix<float64_t> sigma)
{
	ASSERT(sigma.num_rows == mu.vlen && sigma.num_rows == sigma.num_cols);
	index_t num_sample = sigma.num_rows;
	SGVector<float64_t> y(num_sample);

	/*
	//The following code is used to generate synthetic data
	Eigen::Map<Eigen::MatrixXd> eigen_sigma(sigma.matrix, sigma.num_rows, sigma.num_cols);

	//y = mvnrnd(mu, Sigma, 1);
	CProbabilityDistribution * dist =new CGaussianDistribution(mu, sigma);
	y = dist->sample();
	//y = (y(:)>0);
	//Note that Shogun uses -1 and 1 as labels
	for( index_t i = 0; i < y.vlen; ++i)
	{
		if (y[i] > 0)
			y[i] = 1;
		else
			y[i] = -1;
	}
	SG_UNREF(dist);
	*/

	//The following pre-init value is used to verify the correctness
	//Note that Shogun uses -1 and 1 as labels
	y[0]=1;
	y[1]=1;
	y[2]=-1;
	y[3]=1;
	y[4]=-1;
	y[5]=1;
	y[6]=1;
	y[7]=1;
	y[8]=1;
	y[9]=1;
	y[10]=-1;
	y[11]=-1;
	y[12]=-1;
	y[13]=-1;
	y[14]=-1;
	y[15]=-1;
	y[16]=-1;
	y[17]=-1;
	y[18]=-1;
	y[19]=-1;
	return y;
}


//implementing an example based on the Matlab code,example.m
class CProblem
{
public:
	CProblem(SGMatrix<float64_t> bound, SGVector<float64_t> mu,
		SGMatrix<float64_t> data, SGVector<float64_t> label,
		SGMatrix<float64_t> sigma);
	~CProblem();
	void find_optima();
private:
	void init_computation();
	float64_t get_function_value();
	void get_mu_gradient(Eigen::Map<Eigen::VectorXd> * eigen_gradient);
	void inti_lbfgs_parameters();
	static float64_t evaluate(void *obj, const float64_t *alpha,
		float64_t *gradient, const int dim, const float64_t step);
	CLogitPiecewiseBoundLikelihood m_lik;
	SGMatrix<float64_t> m_feature;
	SGVector<float64_t> m_label;
	lbfgs_parameter_t m_lbfgs_param;
	SGMatrix<float64_t> m_omega;
	SGVector<float64_t> m_variables;
	SGVector<float64_t> m_mu;
	SGVector<float64_t> m_g;

};

CProblem::CProblem(SGMatrix<float64_t> bound, SGVector<float64_t> mu, SGMatrix<float64_t> data,  SGVector<float64_t> label, SGMatrix<float64_t> sigma)
{
	REQUIRE(data.num_rows == label.vlen,
		"The size of features and labels should be the same\n");
	REQUIRE((sigma.num_rows == sigma.num_cols) && (sigma.num_rows == data.num_rows),
		"Mismatch the dimension of sigma\n");
	m_label = label;
	m_lik.set_bound(bound);
	m_feature = data;
	m_omega = SGMatrix<float64_t>(m_feature.num_rows, m_feature.num_rows);
	m_mu = mu;
	m_g = SGVector<float64_t>(m_feature.num_rows);

	Eigen::Map<Eigen::MatrixXd> eigen_sigma(sigma.matrix, sigma.num_rows, sigma.num_cols);

	Eigen::Map<Eigen::MatrixXd> eigen_data(m_feature.matrix, m_feature.num_rows, m_feature.num_cols);
	Eigen::Map<Eigen::MatrixXd> eigen_omega(m_omega.matrix, m_omega.num_rows, m_omega.num_cols);
	//Sigma = X*X' + eye(N); % linear kernel
	Eigen::FullPivLU<Eigen::MatrixXd>lu (eigen_sigma);
	ASSERT(lu.isInvertible());
	//Omega = inv(Sigma);
	eigen_omega = lu.inverse();
	m_variables = SGVector<float64_t>(m_feature.num_rows*2);
}

CProblem::~CProblem()
{
}
void CProblem::inti_lbfgs_parameters()
{
	m_lbfgs_param.m = 100;
	m_lbfgs_param.max_linesearch = 1000;
	m_lbfgs_param.linesearch = LBFGS_LINESEARCH_DEFAULT;
	m_lbfgs_param.max_iterations = 1000;
	m_lbfgs_param.delta = 1e-15;
	m_lbfgs_param.past = 0;
	m_lbfgs_param.epsilon = 1e-15;
	m_lbfgs_param.min_step = 1e-20;
	m_lbfgs_param.max_step = 1e+20;
	m_lbfgs_param.ftol = 1e-4;
	m_lbfgs_param.wolfe = 0.9;
	m_lbfgs_param.gtol = 0.9;
	m_lbfgs_param.xtol = 1e-16;
	m_lbfgs_param.orthantwise_c = 0;
	m_lbfgs_param.orthantwise_start = 0;
	m_lbfgs_param.orthantwise_end = 1;
}

void CProblem::find_optima()
{
	//init
	Eigen::Map<Eigen::VectorXd> eigen_m0(m_variables.vector, m_feature.num_rows);
	Eigen::Map<Eigen::VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	//m0 = mu; % initial value
	eigen_m0 = eigen_mu;
	//currently we assume s2 is fixed
	//v = ones(N,1); % fix v to 1
	Eigen::Map<Eigen::VectorXd> eigen_s2(m_variables.vector+m_feature.num_rows, m_feature.num_rows);
	eigen_s2.fill(1);
	inti_lbfgs_parameters();

	float64_t f_value = 0.0;
	void * obj_prt = static_cast<void *>(this);
	int ret = lbfgs(m_variables.vlen, m_variables.vector, &f_value,
		CProblem::evaluate, NULL, obj_prt, &m_lbfgs_param);

	SG_SPRINT("lbfgs status =%d\n",ret);
	SG_SPRINT("min function =%.10f\n",f_value);
	SG_SPRINT("opt mu =\n");
	for(index_t i = 0; i < m_feature.num_rows; ++i)
	{
		SG_SPRINT("mu[%d] =%.10f\n", i+1, eigen_m0(i));
	}
}


float64_t CProblem::evaluate(void *obj, const float64_t *variable,
	float64_t *gradient, const int dim, const float64_t step)
{
	CProblem * obj_prt = static_cast<CProblem *>(obj);
	obj_prt->init_computation();

	float64_t value = obj_prt->get_function_value();
	//currently we assume variance is fixed
	Eigen::Map<Eigen::VectorXd> eigen_gradient(gradient, dim/2);
	obj_prt->get_mu_gradient(&eigen_gradient);
	return value;
}


void CProblem::init_computation()
{
	CBinaryLabels lab(m_label);
	Eigen::Map<Eigen::VectorXd> eigen_variables(m_variables.vector,m_variables.vlen);
	SGVector<float64_t> m(m_feature.num_rows);
	Eigen::Map<Eigen::VectorXd> eigen_m(m.vector, m.vlen);
	SGVector<float64_t> v(m_feature.num_rows);
	Eigen::Map<Eigen::VectorXd> eigen_v(v.vector, v.vlen);
	eigen_m = eigen_variables.head(m_feature.num_rows);
	eigen_v = eigen_variables.tail(m_feature.num_rows);
	m_lik.set_distribution(m, v, &lab);

	Eigen::Map<Eigen::VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Eigen::Map<Eigen::VectorXd> eigen_g(m_g.vector, m_g.vlen);
	Eigen::Map<Eigen::MatrixXd> eigen_omega(m_omega.matrix, m_omega.num_rows, m_omega.num_cols);
	//e = m-mu;
	//g = Omega*e;
	eigen_g = eigen_omega*(eigen_m - eigen_mu);
}

float64_t CProblem::get_function_value()
{
	SGVector<float64_t> fi = m_lik.get_variational_expection(); 

	Eigen::Map<Eigen::VectorXd> eigen_m(m_variables.vector, m_feature.num_rows);
	Eigen::Map<Eigen::VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Eigen::Map<Eigen::VectorXd> eigen_g(m_g.vector, m_g.vlen);

	//f = -e'*g/2 + sum(fi);
	Eigen::VectorXd ff = -0.5*((eigen_m-eigen_mu).transpose()*eigen_g);
	ASSERT(ff.size() == 1);
	float64_t f = ff(0) + SGVector<float64_t>::sum(fi);
	//f = -f;
	return -f;
}
void CProblem::get_mu_gradient(Eigen::Map<Eigen::VectorXd> * eigen_gradient)
{
	TParameter* mu_param=m_lik.m_gradient_parameters->get_parameter("mu");
	SGVector<float64_t> gmi = m_lik.get_variational_first_derivative(mu_param);

	Eigen::Map<Eigen::VectorXd> eigen_g(m_g.vector, m_g.vlen);
	Eigen::Map<Eigen::VectorXd> eigen_gmi(gmi.vector, gmi.vlen);
	//g = -g + gmi; 
	//g = -g;
	*eigen_gradient = eigen_g - eigen_gmi; 
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	//N = 20; % number of data examples
	index_t num_sample = 20;
	//D = 5; % feature dimensionality
	index_t num_dim = 5;


	SGMatrix<float64_t> data = create_feature(num_sample, num_dim);
	SGMatrix<float64_t> bound = init_piecewise_bound();
	SGVector<float64_t> mu(num_sample);
	Eigen::Map<Eigen::VectorXd> eigen_mu(mu.vector, mu.vlen);
	//mu = zeros(N,1); % zero mean
	eigen_mu.fill(0);

	SGMatrix<float64_t> sigma(num_sample, num_sample);
	Eigen::Map<Eigen::MatrixXd> eigen_data(data.matrix, data.num_rows, data.num_cols);
	Eigen::Map<Eigen::MatrixXd> eigen_sigma(sigma.matrix, sigma.num_rows, sigma.num_cols);
	//Sigma = X*X' + eye(N);
	eigen_sigma = eigen_data * (eigen_data.transpose()) + Eigen::MatrixXd::Identity(num_sample, num_sample);

	SGVector<float64_t> y = create_label(mu, sigma);

	CProblem *ok = new CProblem(bound, mu, data, y, sigma);
	ok->find_optima();
	delete ok;


	exit_shogun();
	return 0;
}
#else
int main(int argc, char **argv)
{
  return 0;
}
#endif /* HAVE_EIGEN3 */
