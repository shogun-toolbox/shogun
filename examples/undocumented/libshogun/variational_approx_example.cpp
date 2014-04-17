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
 *
 * Code adapted from 
 * https://github.com/emtiyaz/VariationalApproxExample
 * and the reference paper is
 * Marlin, Benjamin M., Mohammad Emtiyaz Khan, and Kevin P. Murphy.
 * "Piecewise Bounds for Estimating Bernoulli-Logistic Latent Gaussian Models." ICML. 2011.
 *
 * This code specifically adapted from example.m and simpleVariational.m
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
#include <shogun/io/CSVFile.h>
#include <cstdio>

using namespace shogun;

/** @brief This variational_approx_example program
 * demonstrates the usage of Shogun's L-BFGS API and the LogitPiecewiseBoundLikelihood class.
 * First, we generate synthetic data (labels) from a Gaussian distribution with logistic link function.
 * Then, we use the KL method (Nickisch, Hannes, and Carl Edward Rasmussen,
 * "Approximations for Binary Gaussian Process Classification." JMLR (2008)) to find model parameters.
 * In this example, we use a Gaussian distribution with identity covariance matrix
 * to approximate the posterior distribution by directly minimizing the KL divergence
 * between these distributions via the L-BFGS technique.
 * Note that we use the LogitPiecewiseBoundLikelihood class to compute gradient and function value
 * during the L-BFGS iteration.
 */

//init the variational Piecewise bound 
SGMatrix<float64_t> init_piecewise_bound(const char * fname)
{
	SGMatrix<float64_t> bound;
	CCSVFile* bound_file = new CCSVFile(fname);
	bound_file->set_delimiter('\t');
	bound.load(bound_file);
	SG_UNREF(bound_file);
	return bound;
}

//The following pre-init value is used to verify the correctness
//The following code will be removed.
SGVector<float64_t> load_m_from_matlab(const char * fname)
{
	SGVector<float64_t> m_from_matlab;
	CCSVFile* m_file = new CCSVFile(fname);
	m_file->set_delimiter('\t');
	m_from_matlab.load(m_file);
	SG_UNREF(m_file);
	return m_from_matlab;
}

//The following pre-init value is used to verify the correctness
//The following code will be removed.
float64_t load_loglik_from_matlab(const char * fname)
{
	SGVector<float64_t> f_from_matlab;
	CCSVFile* f_file = new CCSVFile(fname);
	f_file->set_delimiter('\t');
	f_from_matlab.load(f_file);
	SG_UNREF(f_file);
	REQUIRE(f_from_matlab.vlen == 1, "logLik is a scalar");
	return f_from_matlab[0];
}

//Randomly generating the input feature (X)
SGMatrix<float64_t> create_feature(const char *fname, index_t num_sample,
	index_t num_dim)
{

	REQUIRE(num_sample % 2 == 0, "For this example we assume the num_sample is even");
	/*
	//X = [5*rand(N/2,D); -5*rand(N/2,D)]; 
	//The following code is used to generate synthetic data
	SGMatrix<float64_t> X(num_sample,num_dim);
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
	//The following code will be removed.
	SGMatrix<float64_t> X;
	CCSVFile* X_file = new CCSVFile(fname);
	X_file->set_delimiter('\t');
	X.load(X_file);
	SG_UNREF(X_file);
	return X;
}

//Randomly generating the observated labels (y) followed by Guassian distribution (synthetic data)
SGVector<float64_t> create_label(const char * fname, SGVector<float64_t> mu,
	SGMatrix<float64_t> sigma)
{

	REQUIRE(sigma.num_rows == sigma.num_cols, "Sigma should be a covariance (square) matrix");
	REQUIRE(sigma.num_rows == mu.vlen, "Sigma and mu should have the same dimensionality");

	/*
	//The following code is used to generate synthetic data
	index_t num_sample = sigma.num_rows;
	SGVector<float64_t> y(num_sample);
	Eigen::Map<Eigen::MatrixXd> eigen_sigma(sigma.matrix, sigma.num_rows, sigma.num_cols);

	//y = mvnrnd(mu, Sigma, 1);
	CProbabilityDistribution * dist = new CGaussianDistribution(mu, sigma);
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
	//The following code will be removed.
	//Note that Shogun uses -1 and 1 as labels
	SGVector<float64_t> y;
	CCSVFile* y_file = new CCSVFile(fname);
	y_file->set_delimiter('\t');
	y.load(y_file);
	SG_UNREF(y_file);

	for(index_t i = 0; i < y.vlen; i++)
	{
		if (y[i] > 0)
			y[i] = 1;
		else
			y[i] = -1;
	}

	REQUIRE(y.vlen == mu.vlen,
		"The labels loaded from the file should have the same dimensionality of mu");
	return y;
}

//The following struct is used to pass information when using the build-in L-BFGS component
struct Shared 
{
	CLogitPiecewiseBoundLikelihood *lik;
	SGVector<float64_t> y;
	SGVector<float64_t> mu;
	lbfgs_parameter_t lbfgs_param;
	SGVector<float64_t> m0;
	SGVector<float64_t> v;
	SGMatrix<float64_t> sigma;
	SGMatrix<float64_t> data;
	SGMatrix<float64_t> bound;
	Eigen::LDLT<Eigen::MatrixXd> ldlt;
};

//Init the parameters used for L-BFGS
lbfgs_parameter_t inti_lbfgs_parameters()
{
	lbfgs_parameter_t tmp;
	tmp.m = 100;
	tmp.max_linesearch = 1000;
	tmp.linesearch = LBFGS_LINESEARCH_DEFAULT;
	tmp.max_iterations = 1000;
	tmp.delta = 1e-15;
	tmp.past = 0;
	tmp.epsilon = 1e-15;
	tmp.min_step = 1e-20;
	tmp.max_step = 1e+20;
	tmp.ftol = 1e-4;
	tmp.wolfe = 0.9;
	tmp.gtol = 0.9;
	tmp.xtol = 1e-16;
	tmp.orthantwise_c = 0;
	tmp.orthantwise_start = 0;
	tmp.orthantwise_end = 1;
	return tmp;
}

//This function is similar to the Matlab code, simpleVariational.m
float64_t evaluate(void *obj, const float64_t *variable, float64_t *gradient,
	const int dim, const float64_t step)
{
	Shared * obj_prt = static_cast<Shared *>(obj);

	CBinaryLabels lab(obj_prt->y);
	obj_prt->lik->set_distribution(obj_prt->m0, obj_prt->v, &lab);
	Eigen::Map<Eigen::VectorXd> eigen_mu(obj_prt->mu.vector, obj_prt->mu.vlen);
	Eigen::Map<Eigen::VectorXd> eigen_m(obj_prt->m0.vector, obj_prt->m0.vlen);

	//[fi, gmi, gvi] = ElogLik('bernLogit', y, m, v, bound); get fi at here
	SGVector<float64_t> fi = obj_prt->lik->get_variational_expection(); 

	TParameter* mu_param = obj_prt->lik->m_gradient_parameters->get_parameter("mu");
	//[fi, gmi, gvi] = ElogLik('bernLogit', y, m, v, bound); get gmi at here
	SGVector<float64_t> gmi =
		obj_prt->lik->get_variational_first_derivative(mu_param);

	SGVector<float64_t> g(dim);
	Eigen::Map<Eigen::VectorXd> eigen_g(g.vector, g.vlen);

	//e = m-mu;
	//g = Omega*e;
	eigen_g = obj_prt->ldlt.solve(eigen_m - eigen_mu);

	//f = -e'*g/2 + sum(fi);
	Eigen::VectorXd ff = -0.5*((eigen_m-eigen_mu).transpose()*eigen_g);
	ASSERT(ff.size() == 1);
	float64_t f = ff(0) + SGVector<float64_t>::sum(fi);

	Eigen::Map<Eigen::VectorXd> eigen_gradient(gradient, dim);
	//get the gradient based on the current variable
	Eigen::Map<Eigen::VectorXd> eigen_gmi(gmi.vector, gmi.vlen);
	//g = -g + gmi; 
	//g = -g;
	eigen_gradient = eigen_g - eigen_gmi; 

	//f = -f;
	return -f;
}

void run(const char * x_file, const char * y_file, const char * bound_file,
	const char * m_file, const char * loglik_file)
{
	//N = 20; % number of data examples
	index_t num_sample = 20;
	//D = 5; % feature dimensionality
	index_t num_dim = 5;

	Shared obj;

	//X = [5*rand(N/2,D); -5*rand(N/2,D)]; 
	obj.data = create_feature(x_file, num_sample, num_dim);

	//if we read from file
	num_sample = obj.data.num_rows;
	num_dim = obj.data.num_cols;

	SG_SPRINT("num_samples:%d num_dimensions:%d\n", num_sample, num_dim);

	//Sigma = X*X' + eye(N); % linear kernel
	obj.sigma = SGMatrix<float64_t> (num_sample, num_sample);
	Eigen::Map<Eigen::MatrixXd> eigen_data(obj.data.matrix, obj.data.num_rows,
		obj.data.num_cols);
	Eigen::Map<Eigen::MatrixXd> eigen_sigma(obj.sigma.matrix,
		obj.sigma.num_rows, obj.sigma.num_cols);
	//Sigma = X*X' + eye(N);
	eigen_sigma = eigen_data * (eigen_data.transpose()) +
		Eigen::MatrixXd::Identity(num_sample, num_sample);

	//mu = zeros(N,1); % zero mean
	obj.mu = SGVector<float64_t> (num_sample);
	Eigen::Map<Eigen::VectorXd> eigen_mu(obj.mu.vector, obj.mu.vlen);
	//mu = zeros(N,1); % zero mean
	eigen_mu.fill(0);

	//y = mvnrnd(mu, Sigma, 1);
	//y = (y(:)>0);
	obj.y = create_label(y_file, obj.mu, obj.sigma);

	//% optimizers options
	//optMinFunc = struct('Display', 1,...
	//'Method', 'lbfgs',...
	//'DerivativeCheck', 'off',...
	//'LS', 2,...
	//'MaxIter', 1000,...
	//'MaxFunEvals', 1000,...
	//'TolFun', 1e-4,......
	//'TolX', 1e-4);
	obj.lbfgs_param = inti_lbfgs_parameters();

	//load('llp.mat'); 
	obj.bound = init_piecewise_bound(bound_file);
	obj.lik = new CLogitPiecewiseBoundLikelihood();
	obj.lik->set_bound(obj.bound);

	//m0 = mu; % initial value all zero
	obj.m0 = SGVector<float64_t> (num_sample);
	obj.v = SGVector<float64_t> (num_sample);
	Eigen::Map<Eigen::VectorXd> eigen_m0(obj.m0.vector, obj.m0.vlen);
	//m0 = mu; % initial value
	eigen_m0 = eigen_mu;

	//v = ones(N,1); % fix v to 1
	Eigen::Map<Eigen::VectorXd> eigen_v(obj.v, num_sample);
	eigen_v.fill(1);

	//Omega = inv(Sigma);
	obj.ldlt.compute(eigen_sigma);
	//sigma is positive definitive
	ASSERT(obj.ldlt.isPositive());

	float64_t logLik = 0.0;
	//[m, logLik] = minFunc(@simpleVariational, m0, optMinFunc, y, X, mu, Omega, v, bound);
	int ret = lbfgs(obj.m0.vlen, obj.m0.vector, &logLik,
		evaluate, NULL, &obj, &obj.lbfgs_param);

	SGVector<float64_t> m_from_matlab = load_m_from_matlab(m_file);
	float64_t logLik_from_matlab = load_loglik_from_matlab(loglik_file);
	ASSERT(m_from_matlab.vlen == num_sample);

	SG_SPRINT("lbfgs status =%d\n",ret);
	SG_SPRINT("logLik from Shogun =%.10f from Matlab =%.10f\n", logLik, logLik_from_matlab);
	SG_SPRINT("opt m =\n");

	for(index_t i = 0; i < obj.m0.vlen; ++i)
	{
		float64_t relative_diff;

		if (m_from_matlab[i] != 0.0)
			relative_diff = CMath::abs(obj.m0[i]/m_from_matlab[i] - 1);
		else
			relative_diff = CMath::abs(obj.m0[i]);

		SG_SPRINT("m[%d] from Shogun =%.10f from Matlab = %.10f relative_diff = %.10f\n", i+1,
			obj.m0[i], m_from_matlab[i], relative_diff);
	}

	SG_UNREF(obj.lik);
}

void test_datasets()
{
	const index_t buff_size = 1024;
	const char * data_path = "../data/toy/variational";

	char bound_path_buffer[buff_size];
	char x_path_buffer[buff_size];
	char y_path_buffer[buff_size];
	char m_path_buffer[buff_size];
	char loglik_path_buffer[buff_size];

	snprintf(bound_path_buffer, buff_size, "%s/bounds", data_path);

	FILE* pfile = fopen(bound_path_buffer, "r");

	if (pfile == NULL)
	{
		SG_SPRINT("Unable to open file: %s\n", bound_path_buffer);
		return;
	}

	fclose(pfile);
	
	for (index_t i = 1; i <= 6; i++)
	{
		snprintf(x_path_buffer, buff_size, "%s/X_dataset%d", data_path, i);
		snprintf(y_path_buffer, buff_size, "%s/y_dataset%d", data_path, i);
		snprintf(m_path_buffer, buff_size, "%s/m_dataset%d", data_path, i);
		snprintf(loglik_path_buffer, buff_size, "%s/logLik_dataset%d", data_path, i);
		SG_SPRINT("\nDataset %d\n", i);
		run(x_path_buffer, y_path_buffer, bound_path_buffer, m_path_buffer, loglik_path_buffer);
	}

}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
	test_datasets();
	exit_shogun();
	return 0;
}
#endif /* HAVE_EIGEN3 */
