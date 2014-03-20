/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 pl8787
 */

#include <shogun/lib/config.h>

// temporally disabled, since API was changed
#ifdef HAVE_EIGEN3

#include <iostream>
#include <shogun/base/init.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;
using namespace Eigen;

#define ZEROTH_MOMENT 1
#define GRAD_MEAN 2
#define GRAD_VAR 4
#define HM 8

// Type of Prob Name
typedef enum _probName
{
	poisson,
	bernLogit
}ProbName;

// Type of Bsxfun Operation
typedef enum _bsxfunOp
{
	plus,
	minus,
	times
}BsxfunOp;

// Debug information
void debug(const char * msg)
{
	std::cout<<msg<<std::endl;
}

float64_t _normpdf(float64_t x)
{
	if(x == CMath::INFTY || x == -CMath::INFTY)
		return 0.0;
	else
		return CMath::exp(-0.5*(x*x + CMath::log(2.0*CMath::PI)));
}

template<typename M1>
MatrixXd normpdf(const MatrixBase<M1> &x)
{
	return x.unaryExpr(std::ptr_fun(_normpdf));
}

float64_t _normcdf(float64_t x)
{
	return 0.5*CStatistics::error_function(x/CMath::sqrt(2.0));
}

template<typename M1>
MatrixXd normcdf(const MatrixBase<M1> &x)
{
	return x.unaryExpr(std::ptr_fun(_normcdf));
}

template<typename M1, typename M2>
MatrixXd bsxfun(BsxfunOp op,const MatrixBase<M1> &x,const MatrixBase<M2> &y)
{
	ASSERT((x.rows()==y.rows() || x.rows()==1 || y.rows()==1) && (x.cols()==y.cols() || x.cols()==1 || y.cols()==1));

	MatrixXd xx;
	MatrixXd yy;
	if(x.rows()==1)
		xx = x.replicate(y.rows(), 1);
	else if(x.cols()==1)
		xx = x.replicate(1, y.cols());
	else
		xx = x;
	if(y.rows()==1)
		yy = y.replicate(x.rows(), 1);
	else if(y.cols()==1)
		yy = y.replicate(1, x.cols());
	else
		yy = y;

	switch(op)
	{
	case plus:
		return xx+yy;
	case minus:
		return xx-yy;
	case times:
		return (xx.array() * yy.array()).matrix();
	default:
		return xx;
	}
}

template<typename M1, typename M2, typename M3>
bool Ellp(
		index_t mode,
		MatrixBase<M1> &m,
		MatrixBase<M1> &v,
		MatrixBase<M2> &bound,
		MatrixBase<M3> &f,
		MatrixBase<M1> &gm,
		MatrixBase<M1> &gv)
{
// compute piecewise bound to E(log(1+exp(x))) where x~N(m,v)
// Here, m and v can be vectors
// bound need to be a matrix, can be obtained by loading llp.mat
// ind is 3x1 vector specifying which outputs to compute
// Example:
// [f2, gm2, gv2] = funObj_pw_new(m, v, bound, [1 1 1]);
// see the appendix
// http://www.cs.ubc.ca/~emtiyaz/papers/truncatedGaussianMoments.pdf
// for detailed expressions
//
// Written by Emtiyaz, CS, UBC
// Modifiied on May 26, 2012


	for(index_t i=0; i<v.cols(); i++)
	{
		if(v(i)<0)
		{
			std::cout<<"Normal variance must be strictly positive"<<std::endl;
			return false;
		}
	}
	// get piecewise bound parameters
	// (a,b,c) are parameters for quadratic pieces and (l,h) are lower and upper limit of each piece
	debug("c = bound(1,:)';");
	MatrixXd c = bound.row(0);
	debug("b = bound(2,:)';");
	MatrixXd b = bound.row(1);
	debug("a = bound(3,:)';");
	MatrixXd a = bound.row(2);
	debug("l = bound(4,:)';");
	MatrixXd l = bound.row(3);
	debug("h = bound(5,:)';");
	MatrixXd h = bound.row(4);

	// compute pdf and cdfs
	debug("zl = bsxfun(@times, bsxfun(@minus,l,m), 1./sqrt(v))");
	MatrixXd v_sq_inv = v.array().sqrt().inverse().matrix();
	MatrixXd zl = bsxfun(times, bsxfun(minus,l,m), v_sq_inv);

	debug("zh = bsxfun(@times, bsxfun(@minus,h,m), 1./sqrt(v))");
	MatrixXd zh = bsxfun(times, bsxfun(minus,h,m), v_sq_inv);

	debug("pl = bsxfun(@times, normpdf(zl), 1./sqrt(v));"); //normal pdf
	MatrixXd pl = bsxfun(times, normpdf(zl), v_sq_inv);

	debug("ph = bsxfun(@times, normpdf(zh), 1./sqrt(v));"); //normal pdf
	MatrixXd ph = bsxfun(times, normpdf(zh), v_sq_inv);

	debug("cl = 0.5*erf(zl/sqrt(2));"); //normal cdf -const
	MatrixXd cl = normcdf(zl);

	debug("ch = 0.5*erf(zh/sqrt(2));"); //normal cdf -cosnt
	MatrixXd ch = normcdf(zh);

	// zero out the inf and -inf in l and h
	l(0,0) = 0;
	h(0,bound.cols()-1) = 0;

	f.fill(0.0);
	gm.fill(0.0);
	gv.fill(0.0);

	// compute function value
	if(mode & ZEROTH_MOMENT)
	{
		//Compute truncated zeroth moment
		debug("ex0 = ch-cl;");
		MatrixXd ex0 = ch - cl;
		//Compute truncated first moment
		//ex1= v.*(pl-ph) + m.*(ch-cl);
		debug("ex1= bsxfun(@times, v, (pl-ph)) + bsxfun(@times, m,(ch-cl));");
		MatrixXd ex1 = bsxfun(times, v, pl-ph) + bsxfun(times, m, ch-cl);
		//Compute truncated second moment
		//ex2=  v.*((l+m).*pl - (h+m).*ph) + (v+m.^2).*(ch - cl);
		debug("ex2 = bsxfun(@times, v, (bsxfun(@plus, l, m)).*pl... \
		- (bsxfun(@plus, h, m)).*ph) \
		+ bsxfun(@times, (v+m.^2), (ch-cl));");
		MatrixXd ex2 = bsxfun(times, v, bsxfun(plus, l, m)).cwiseProduct(pl) \
					   - bsxfun(plus, h, m).cwiseProduct(ph) \
					   + bsxfun(times, v+m.array().square().matrix(), ch-cl);
		// compute f
		//fr = a.*ex2 + b.*ex1 + c.*ex0;
		debug("fr = bsxfun(@times, a, ex2) + bsxfun(@times, b, ex1) + bsxfun(@times, c, ex0);");
		MatrixXd fr = bsxfun(times, a, ex2) + bsxfun(times, b, ex1) + bsxfun(times, c, ex0);
		debug("f = sum(fr,1)';");
		f = fr.rowwise().sum();
	}

	//Compute Gradient wrt to mean
	if(mode & GRAD_MEAN)
	{
		//gm = a.*( (l.^2+2*v).*pl - (h.^2+2*v).*ph) + a.*2.*m.*(ch-cl);
		debug("gm = bsxfun(@times, a, bsxfun(@plus, l.^2, 2*v).*pl \
		  - bsxfun(@plus, h.^2, 2*v).*ph)... \
		  + 2*bsxfun(@times, a, m).*(ch - cl);");
		MatrixXd gmr = bsxfun(times, a, bsxfun(plus, l.array().square().matrix(), 2*v).cwiseProduct(pl) \
				- bsxfun(plus, h.array().square().matrix(), 2*v).cwiseProduct(ph)) \
				+ 2*bsxfun(times, a, m).cwiseProduct(ch-cl);
		//gm = gm + b.*(l.*pl-h.*ph) + b.*(ch-cl);
		debug("gm = gm + bsxfun(@times, b, bsxfun(@times, l, pl) - bsxfun(@times, h, ph))... \
		       + bsxfun(@times, b, ch-cl);");
		gmr = gmr + bsxfun(times, b, bsxfun(times, l, pl) - bsxfun(times, h, ph)) \
				+ bsxfun(times, b, ch-cl);
		//gm = gm + c.*(pl-ph);
		debug("gm = gm + bsxfun(@times, c, pl-ph);");
		gmr = gmr + bsxfun(times, c, pl-ph);
		debug("gm = sum(gm,1)';");
		gm = gmr.rowwise().sum();
	}

	//Compute Gradient wrt to variance
	if(mode & GRAD_VAR)
	{
		debug("t1 = bsxfun(@plus, 2*bsxfun(@times, v, l), l.^3) - bsxfun(@times, l.^2, m);");
		MatrixXd t1 = bsxfun(plus, 2 * bsxfun(times, v, l), l.array().pow(3).matrix()) \
					  - bsxfun(times, l.array().square().matrix(), m);

		debug("t2 = bsxfun(@plus, 2*bsxfun(@times, v, h), h.^3) - bsxfun(@times, h.^2, m);");
		MatrixXd t2 = bsxfun(plus, 2 * bsxfun(times, v, h), h.array().pow(3).matrix()) \
					  - bsxfun(times, h.array().square().matrix(), m);

		MatrixXd v_inv = v.array().inverse().matrix();

		debug("gv = bsxfun(@times, a/2, 1./v).*(t1.*pl - t2.*ph) + bsxfun(@times, a, ch-cl);");
		MatrixXd gvr = bsxfun(times, a/2, v_inv).cwiseProduct(t1.cwiseProduct(pl) - t2.cwiseProduct(ph)) \
					   + bsxfun(times, a, ch-cl);

		debug("gv = gv + bsxfun(@times, b/2, 1./v).*...\
		    ((bsxfun(@plus, l.^2, v) - bsxfun(@times, l, m)).*pl ...\
		    - (bsxfun(@plus, h.^2, v) - bsxfun(@times, h, m)).*ph);");
		gvr = gvr + bsxfun(times, b/2, v_inv).cwiseProduct( \
				    (bsxfun(plus, l.array().square().matrix(), v) - bsxfun(times, l, m)).cwiseProduct(pl) \
				    - (bsxfun(plus, h.array().square().matrix(), v) - bsxfun(times, h, m)).cwiseProduct(ph) \
			   );

		debug("gv = gv + bsxfun(@times, c/2, 1./v).*...\
		    ((bsxfun(@minus,l,m)).*pl - (bsxfun(@minus,h,m)).*ph);");
		gvr = gvr + bsxfun(times, c/2, v_inv).cwiseProduct( \
				  bsxfun(minus, l, m).cwiseProduct(pl) - bsxfun(minus, h, m).cwiseProduct(ph) \
				);

		//gv = a/2./v.*( (2*v*l + l.^3 -l.^2*m).*pl - (2*v*h + h.^3 -h.^2*m).*ph) +a.*(ch-cl);
		//gv = gv + b/2./v.*( (l.^2+v-l*m).*pl - (h.^2+v-h*m).*ph);
		//gv = gv + c/2./v.*((l-m).*pl-(h-m).*ph);
		debug("gv = sum(gv,1)';");
		gv = gvr.rowwise().sum();
	}

	if(mode & HM)
	{
		MatrixXd v_inv = v.array().inverse().matrix();
		debug("hm = bsxfun(@times, a, bsxfun(@times, 1./v, \
			  bsxfun(@plus, l.^3, -l.^2*m + 2*l*v).*pl - bsxfun(@plus, h.^3, -h.^2*m + 2*h*v).*ph) ...");
		//  + 2.*(ch - cl));
		MatrixXd hmr = bsxfun(times, a, \
				         bsxfun(times, v_inv, \
				           bsxfun(plus, l.array().pow(3).matrix(), \
				        		   -m*l.array().square().matrix() + 2*v*l).cwiseProduct(pl) \
				           - bsxfun(plus, h.array().pow(3).matrix(), \
				        		   -m*h.array().square().matrix() + 2*v*h).cwiseProduct(ph) \
				         ) - 2*(ch-cl)
				       );

		debug("hm = hm + bsxfun(@times, b, bsxfun(@times, 1./v, bsxfun(@minus, l.^2, l*m).*pl \
				- bsxfun(@minus, h.^2, h*m).*ph) + (pl - ph));");
		hmr = hmr + bsxfun(times, b, \
					  bsxfun(times, v_inv, \
					    bsxfun(minus, l.array().square().matrix(), m*l).cwiseProduct(pl) \
					    - bsxfun(minus, h.array().square().matrix(), m*h).cwiseProduct(ph) \
					  ) + pl - ph
					);

		debug("hm = hm + bsxfun(@times, c, 1./v).* \
				((bsxfun(@minus,l,m)).*pl - (bsxfun(@minus,h,m)).*ph);");
		hmr = hmr + bsxfun(times, c, v_inv).cwiseProduct( \
				      bsxfun(minus, l, m).cwiseProduct(pl) - bsxfun(minus, h, m).cwiseProduct(ph) \
					);

		debug("hm = sum(hm,1)'");
		MatrixXd hm = hmr.rowwise().sum();
	}
	return true;
}


bool ElogLik(
		ProbName name,
		SGVector<float64_t> &y,
		SGVector<float64_t> &m,
		SGVector<float64_t> &v,
		SGMatrix<float64_t> &bound,
		SGVector<float64_t> &f,
		SGVector<float64_t> &gm,
		SGVector<float64_t> &gv)
{
/*
This function computes E( log p(y|x) ) where
expectation is wrt p(x) = N(x|m,v) with mean m and variance v.
params are optional parameters required for approximation

Written by Emtiyaz, EPFL, pl8787
Modified on March 8, 2014
*/
	//Eigenlize all parameter
	//y = y(:); m = m(:); v = v(:);
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	Map<VectorXd> eigen_m(m.vector, m.vlen);
	Map<VectorXd> eigen_v(v.vector, v.vlen);
	Map<MatrixXd> eigen_bound(bound.matrix, bound.num_rows, bound.num_cols);

	//Output value
	Map<VectorXd> eigen_f(f.vector, f.vlen);
	Map<VectorXd> eigen_gm(gm.vector, gm.vlen);
	Map<VectorXd> eigen_gv(gv.vector, gv.vlen);

	//Temp value
	VectorXd eigen_t;

	//Return val
	bool rtn = true;

	switch(name)
	{
	case poisson:
	//log p(y|x) = y*x - e^x where y is non-negative integer
		//t = exp(m+v/2);
		eigen_t = (eigen_m+eigen_v * 0.5).array().exp().matrix();
		//f = y.*m - t;
		eigen_f = (eigen_y.array() * eigen_m.array()).matrix() - eigen_t;
		//gm = y - t;
		eigen_gm = eigen_y - eigen_t;
		//gv = -t/2;
		eigen_gv = -eigen_t * 0.5;
		break;
	case bernLogit:
	//log p(y|x) = y*x - log(1+exp(x) where y is 0 or 1
	//Based on "Piecewise Bounds for Estimating ...
	//Bernoulli-Logistic Latent Gaussian Models", ICML 2011
	    //llp_bound = params; % approx to log(1+exp(x))
	    //[t, gm, gv] = Ellp(m, v, llp_bound, [1 1 1]);
		rtn = Ellp(ZEROTH_MOMENT | GRAD_MEAN | GRAD_VAR | HM, eigen_m, eigen_v, \
				eigen_bound, eigen_t, eigen_gm, eigen_gv);
		if(!rtn)
			return false;
	    //f = y.*m - t;
		eigen_f = (eigen_y.array() * eigen_m.array()).matrix() - eigen_t;
	    //gm = y - gm;
		eigen_gm = eigen_y - eigen_gm;
	    //gv = -gv;
		eigen_gv = -eigen_gv;
		break;
	default:
		std::cout<<"no such name"<<std::endl;
		return false;
	}
	return true;
}

int main()
{
	index_t row = 5;
	index_t col = 20;
	SGMatrix<float64_t> bound(row, col);
	bound(0,0)=0.000188712193000;
	bound(0,1)=0.028090310300000;
	bound(0,2)=0.110211757000000;
	bound(0,3)=0.232736440000000;
	bound(0,4)=0.372524706000000;
	bound(0,5)=0.504567936000000;
	bound(0,6)=0.606280283000000;
	bound(0,7)=0.666125432000000;
	bound(0,8)=0.689334264000000;
	bound(0,9)=0.693147181000000;
	bound(0,10)=0.693147181000000;
	bound(0,11)=0.689334264000000;
	bound(0,12)=0.666125432000000;
	bound(0,13)=0.606280283000000;
	bound(0,14)=0.504567936000000;
	bound(0,15)=0.372524706000000;
	bound(0,16)=0.232736440000000;
	bound(0,17)=0.110211757000000;
	bound(0,18)=0.028090310400000;
	bound(0,19)=0.000188712000000;

	bound(1,0)=0;
	bound(1,1)=0.006648614600000;
	bound(1,2)=0.034432684600000;
	bound(1,3)=0.088701969900000;
	bound(1,4)=0.168024214000000;
	bound(1,5)=0.264032863000000;
	bound(1,6)=0.360755794000000;
	bound(1,7)=0.439094482000000;
	bound(1,8)=0.485091758000000;
	bound(1,9)=0.499419205000000;
	bound(1,10)=0.500580795000000;
	bound(1,11)=0.514908242000000;
	bound(1,12)=0.560905518000000;
	bound(1,13)=0.639244206000000;
	bound(1,14)=0.735967137000000;
	bound(1,15)=0.831975786000000;
	bound(1,16)=0.911298030000000;
	bound(1,17)=0.965567315000000;
	bound(1,18)=0.993351385000000;
	bound(1,19)=1.000000000000000;

	bound(2,0)=0;
	bound(2,1)=0.000397791059000;
	bound(2,2)=0.002753100850000;
	bound(2,3)=0.008770186980000;
	bound(2,4)=0.020034759300000;
	bound(2,5)=0.037511596000000;
	bound(2,6)=0.060543032900000;
	bound(2,7)=0.086256780600000;
	bound(2,8)=0.109213531000000;
	bound(2,9)=0.123026104000000;
	bound(2,10)=0.123026104000000;
	bound(2,11)=0.109213531000000;
	bound(2,12)=0.086256780600000;
	bound(2,13)=0.060543032900000;
	bound(2,14)=0.037511596000000;
	bound(2,15)=0.020034759300000;
	bound(2,16)=0.008770186980000;
	bound(2,17)=0.002753100850000;
	bound(2,18)=0.000397791059000;
	bound(2,19)=0;

	bound(3,0)=-CMath::INFTY;
	bound(3,1)=-8.575194939999999;
	bound(3,2)=-5.933689180000000;
	bound(3,3)=-4.525933600000000;
	bound(3,4)=-3.528107790000000;
	bound(3,5)=-2.751548540000000;
	bound(3,6)=-2.097898790000000;
	bound(3,7)=-1.519690830000000;
	bound(3,8)=-0.989533382000000;
	bound(3,9)=-0.491473077000000;
	bound(3,10)=0;
	bound(3,11)=0.491473077000000;
	bound(3,12)=0.989533382000000;
	bound(3,13)=1.519690830000000;
	bound(3,14)=2.097898790000000;
	bound(3,15)=2.751548540000000;
	bound(3,16)=3.528107790000000;
	bound(3,17)=4.525933600000000;
	bound(3,18)=5.933689180000000;
	bound(3,19)=8.575194939999999;


	bound(4,0)=-8.575194939999999;
	bound(4,1)=-5.933689180000000;
	bound(4,2)=-4.525933600000000;
	bound(4,3)=-3.528107790000000;
	bound(4,4)=-2.751548540000000;
	bound(4,5)=-2.097898790000000;
	bound(4,6)=-1.519690830000000;
	bound(4,7)=-0.989533382000000;
	bound(4,8)=-0.491473077000000;
	bound(4,9)=0;
	bound(4,10)=0.491473077000000;
	bound(4,11)=0.989533382000000;
	bound(4,12)=1.519690830000000;
	bound(4,13)=2.097898790000000;
	bound(4,14)=2.751548540000000;
	bound(4,15)=3.528107790000000;
	bound(4,16)=4.525933600000000;
	bound(4,17)=5.933689180000000;
	bound(4,18)=8.575194939999999;
	bound(4,19)= CMath::INFTY;

	const index_t dim = 2;
	SGVector<float64_t> y(dim);
	SGVector<float64_t> m(dim);
	SGVector<float64_t> v(dim);
	SGVector<float64_t> f(dim);
	SGVector<float64_t> gm(dim);
	SGVector<float64_t> gv(dim);

	y[0] = 1;
	y[1] = 1;
	m[0] = 0.5;
	m[1] = 10;
	v[0] = 1;
	v[1] = 2;

	ElogLik(poisson, y, m, v, bound, f, gm, gv);

	for(index_t i = 0; i < dim; i++)
	{
		printf("f[%d]=%.10f gm[%d]=%.10f gv[%d]=%.10f\n", i, f[i], i, gm[i], i, gv[i]);
	}

	ElogLik(bernLogit, y, m, v, bound, f, gm, gv);

	for(index_t i = 0; i < dim; i++)
	{
		printf("f[%d]=%.10f gm[%d]=%.10f gv[%d]=%.10f\n", i, f[i], i, gm[i], i, gv[i]);
	}
	return 0;
}
#endif
