/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser, Saurabh Goyal, Chiyuan Zhang, 
 *          Viktor Gal
 */

#include <shogun/classifier/mkl/MKLMulticlassGradient.h>
#include <shogun/mathematics/Math.h>
#include <vector>
#include <cassert>

using namespace shogun;

MKLMulticlassGradient::MKLMulticlassGradient()
{
	numkernels = 0;
	pnorm=2;

}
MKLMulticlassGradient::~MKLMulticlassGradient()
{

}

MKLMulticlassGradient MKLMulticlassGradient::operator=(MKLMulticlassGradient & gl)
{
	numkernels=gl.numkernels;
	pnorm=gl.pnorm;
	return (*this);

}
MKLMulticlassGradient::MKLMulticlassGradient(MKLMulticlassGradient & gl)
{
	numkernels=gl.numkernels;
	pnorm=gl.pnorm;

}

void MKLMulticlassGradient::setup(const int32_t numkernels2)
{
	numkernels=numkernels2;
	if (numkernels<=1)
	{
		error("void glpkwrapper::setup(const int32_tnumkernels): input "
				"numkernels out of bounds: {}\n",numkernels);
	}

}

void MKLMulticlassGradient::set_mkl_norm(float64_t norm)
{
	pnorm=norm;
	if(pnorm<1 )
      error("MKLMulticlassGradient::set_mkl_norm(float64_t norm) : parameter pnorm<1");
}


void MKLMulticlassGradient::addconstraint(const ::std::vector<float64_t> & normw2,
		const float64_t sumofpositivealphas)
{
	normsofsubkernels.push_back(normw2);
	sumsofalphas.push_back(sumofpositivealphas);
}

void MKLMulticlassGradient::genbetas( ::std::vector<float64_t> & weights ,const ::std::vector<float64_t> & gammas)
{

	assert((int32_t)gammas.size()+1==numkernels);

	double pi4=3.151265358979238/2;

	weights.resize(numkernels);


	// numkernels-dimensional polar transform
	weights[0]=1;

	for(int32_t i=0; i< numkernels-1 ;++i)
	{
		for(int32_t k=0; k< i+1 ;++k)
		{
			weights[k]*=cos( std::min(std::max(0.0,gammas[i]),pi4) );
		}
		weights[i+1]=sin( std::min(std::max(0.0,gammas[i]),pi4) );
	}

	// pnorm- manifold adjustment
	if(pnorm!=2.0)
	{
		for(int32_t i=0; i< numkernels ;++i)
			weights[i]=pow(weights[i],2.0/pnorm);
	}
}

void MKLMulticlassGradient::gengammagradient( ::std::vector<float64_t> & gammagradient ,const ::std::vector<float64_t> & gammas,const int32_t dim)
{

	assert((int32_t)gammas.size()+1==numkernels);

	double pi4=3.151265358979238/2;

	gammagradient.resize(numkernels);
	std::fill(gammagradient.begin(),gammagradient.end(),0.0);

	// numkernels-dimensional polar transform
	gammagradient[0]=1;

	for(int32_t i=0; i< numkernels-1 ;++i)
	{
		if(i!=dim)
		{
			for(int32_t k=0; k< std::min(i+1,dim+2) ;++k)
			{
				gammagradient[k]*=pow( cos( std::min(std::max(0.0,gammas[i]),pi4) ), 2.0/pnorm) ;
			}
			if(i<dim)
				gammagradient[i+1]=pow( sin( std::min(std::max(0.0,gammas[i]),pi4) ),2.0/pnorm);
		}
		else if(i==dim)
		{
			// i==dim, higher dims are 0
			for(int32_t k=0; k< i+1 ;++k)
			{
				gammagradient[k]*= pow( cos( std::min(std::max(0.0,gammas[i]),pi4) ), 2.0/pnorm-1.0)*(-1)*sin( std::min(std::max(0.0,gammas[i]),pi4) );
			}
			gammagradient[i+1]=pow( sin( std::min(std::max(0.0,gammas[i]),pi4) ),2.0/pnorm-1)*cos( std::min(std::max(0.0,gammas[i]),pi4) );
		}
	}
}

float64_t MKLMulticlassGradient::objectives(const ::std::vector<float64_t> & weights, const int32_t index)
{
	assert(index>=0);
	assert(index < (int32_t) sumsofalphas.size());
	assert(index < (int32_t) normsofsubkernels.size());


	float64_t obj= -sumsofalphas[index];
	for(int32_t i=0; i< numkernels ;++i)
	{
		obj+=0.5*normsofsubkernels[index][i]*weights[i];
	}
	return(obj);
}


void MKLMulticlassGradient::linesearch(std::vector<float64_t> & finalbeta,const std::vector<float64_t> & oldweights)
{

	float64_t pi4=3.151265358979238/2;

	float64_t fingrad=1e-7;
	int32_t maxhalfiter=20;
	int32_t totaliters=6;
	float64_t maxrelobjdiff=1e-6;

	std::vector<float64_t> finalgamma,curgamma;

	curgamma.resize(numkernels-1);
	if(oldweights.empty())
	{
	std::fill(curgamma.begin(),curgamma.end(),pi4/2);
	}
	else
	{
	// curgamma from init: arcsin on highest dim ^p/2 !!! and divided everthing by its cos
		std::vector<float64_t> tmpbeta(numkernels);
		for(int32_t i=numkernels-1; i>= 0 ;--i)
		{
         tmpbeta[i]=pow(oldweights[i],pnorm/2);
		}

		for(int32_t i=numkernels-1; i>= 1 ;--i)
		{
			curgamma[i-1]=asin(tmpbeta[i]);

			if(i<numkernels-1)
			{
				if( cos(curgamma[i])<=0)
				{
               io::info("linesearch(...): at i {} cos(curgamma[i-1])<=0 {}",i, cos(curgamma[i-1]));
					//curgamma[i-1]=pi4/2;
				}
			}

			for(int32_t k=numkernels-2; k>= 1 ;--k) // k==0 not necessary
			{
				if(cos(curgamma[i-1])>0)
				{
					tmpbeta[k]/=cos(curgamma[i-1]);
					if(tmpbeta[k]>1)
					{
                  io::info("linesearch(...): at k {} tmpbeta[k]>1 {}",k, tmpbeta[k]);
					}
					tmpbeta[k]=std::min(1.0,std::max(0.0, tmpbeta[k]));
				}
			}
		}
	}

				for(size_t i=0;i<curgamma.size();++i)
	{
		io::info("linesearch(...): curgamma[i] {}",curgamma[i]);
	}


	bool finished=false;
	int32_t longiters=0;
	while(!finished)
	{
		++longiters;
		std::vector<float64_t> curbeta;
		genbetas( curbeta ,curgamma);
		//find smallest objective
		int32_t minind=0;
		float64_t minval=objectives(curbeta,  minind);
		io::info("linesearch(...): objectives at i {}",minval);
		for(int32_t i=1; i< (int32_t)sumsofalphas.size() ;++i)
		{
			float64_t tmpval=objectives(curbeta, i);
		io::info("linesearch(...): objectives at i {}",tmpval);
			if(tmpval<minval)
			{
				minval=tmpval;
				minind=i;
			}
		}
		float64_t lobj=minval;
		//compute gradient for smallest objective
		std::vector<float64_t> curgrad;
		for(int32_t i=0; i< numkernels-1 ;++i)
		{
			::std::vector<float64_t> gammagradient;
			gengammagradient(  gammagradient ,curgamma,i);
			curgrad.push_back(objectives(gammagradient, minind));
		}
		//find boundary hit point (check for each dim) to [0, pi/4]
		std::vector<float64_t> maxalphas(numkernels-1,0);
		float64_t maxgrad=0;
		for(int32_t i=0; i< numkernels-1 ;++i)
		{
			maxgrad=std::max(maxgrad,fabs(curgrad[i]) );
			if(curgrad[i]<0)
			{
				maxalphas[i]=(0-curgamma[i])/curgrad[i];
			}
			else if(curgrad[i]>0)
			{
				maxalphas[i]=(pi4-curgamma[i])/curgrad[i];
			}
			else
			{
				maxalphas[i]=1024*1024;
			}
		}

		float64_t maxalpha=maxalphas[0];
		for(int32_t i=1; i< numkernels-1 ;++i)
		{
			maxalpha=std::min(maxalpha,maxalphas[i]);
		}

		if((maxalpha>1024*1023)|| (maxgrad<fingrad))
		{
			//curgrad[i] approx 0 for all i terminate
			finished=true;
			finalgamma=curgamma;
		}
		else // of if(maxalpha>1024*1023)
		{
		//linesearch: shrink until min of all objective increases compared to starting point, then check left half and right halve until finish
		// curgamma + al*curgrad ,aximizes al in [0, maxal]
			float64_t leftalpha=0, rightalpha=maxalpha, midalpha=(leftalpha+rightalpha)/2;

			std::vector<float64_t> tmpgamma=curgamma, tmpbeta;
			for(int32_t i=1; i< numkernels-1 ;++i)
			{
				tmpgamma[i]=tmpgamma[i]+rightalpha*curgrad[i];
			}
			genbetas( tmpbeta ,tmpgamma);
			float64_t curobj=objectives(tmpbeta, 0);
			for(int32_t i=1; i< (int32_t)sumsofalphas.size() ;++i)
			{
				curobj=std::min(curobj,objectives(tmpbeta, i));
			}

			int curhalfiter=0;
			while((curobj < minval)&&(curhalfiter<maxhalfiter)&&(fabs(curobj/minval-1 ) > maxrelobjdiff ))
			{
				rightalpha=midalpha;
				midalpha=(leftalpha+rightalpha)/2;
				++curhalfiter;
				tmpgamma=curgamma;
				for(int32_t i=1; i< numkernels-1 ;++i)
				{
					tmpgamma[i]=tmpgamma[i]+rightalpha*curgrad[i];
				}
				genbetas( tmpbeta ,tmpgamma);
				curobj=objectives(tmpbeta, 0);
				for(int32_t i=1; i< (int32_t)sumsofalphas.size() ;++i)
				{
					curobj=std::min(curobj,objectives(tmpbeta, i));
				}
			}

			float64_t robj=curobj;
         float64_t tmpobj=std::max(lobj,robj);
			do
			{

				tmpobj=std::max(lobj,robj);

				tmpgamma=curgamma;
				for(int32_t i=1; i< numkernels-1 ;++i)
				{
					tmpgamma[i]=tmpgamma[i]+midalpha*curgrad[i];
				}
				genbetas( tmpbeta ,tmpgamma);
				curobj=objectives(tmpbeta, 0);
				for(int32_t i=1; i< (int32_t)sumsofalphas.size() ;++i)
				{
					curobj=std::min(curobj,objectives(tmpbeta, i));
				}

				if(lobj>robj)
				{
					rightalpha=midalpha;
					robj=curobj;
				}
				else
				{
					leftalpha=midalpha;
					lobj=curobj;
				}
				midalpha=(leftalpha+rightalpha)/2;

			}
			while(  fabs(curobj/tmpobj-1 ) > maxrelobjdiff  );
			finalgamma=tmpgamma;
			curgamma=tmpgamma;
		} // else // of if(maxalpha>1024*1023)

		if(longiters>= totaliters)
		{
			finished=true;
		}
	}
	genbetas(finalbeta,finalgamma);
	float64_t nor=0;
	for(int32_t i=0; i< numkernels ;++i)
	{
		nor+=pow(finalbeta[i],pnorm);
	}
	if(nor>0)
	{
		nor=pow(nor,1.0/pnorm);
		for(int32_t i=0; i< numkernels ;++i)
		{
			finalbeta[i]/=nor;
		}
	}
}


void MKLMulticlassGradient::linesearch2(std::vector<float64_t> & finalbeta,const std::vector<float64_t> & oldweights)
{

const float64_t epsRegul = 0.01;

int32_t num_kernels=(int)oldweights.size();
int32_t nofKernelsGood=num_kernels;

finalbeta=oldweights;

	for( int32_t p=0; p<num_kernels; ++p )
	{
		//io::print("MKL-direct:  sumw[{:3d}] = {:e}  ( oldbeta = {:e} )\n", p, sumw[p], old_beta[p] );
		if(  oldweights[p] >= 0.0 )
		{
			finalbeta[p] = normsofsubkernels.back()[p] * oldweights[p]*oldweights[p] / pnorm;
			finalbeta[p] = Math::pow( finalbeta[p], 1.0 / (pnorm+1.0) );
		}
		else
		{
			finalbeta[p] = 0.0;
			--nofKernelsGood;
		}
		ASSERT( finalbeta[p] >= 0 )
	}

	// --- normalize
	float64_t Z = 0.0;
	for( int32_t p=0; p<num_kernels; ++p )
		Z += Math::pow( finalbeta[p], pnorm );

	Z = Math::pow( Z, -1.0/pnorm );
	ASSERT( Z >= 0 )
	for( int32_t p=0; p<num_kernels; ++p )
		finalbeta[p] *= Z;

	// --- regularize & renormalize
	float64_t preR = 0.0;
	for( int32_t p=0; p<num_kernels; ++p )
		preR += Math::pow( oldweights[p] - finalbeta[p], 2.0 );

	const float64_t R = std::sqrt(preR / pnorm) * epsRegul;
	if( !( R >= 0 ) )
	{
		io::print("MKL-direct: p = {:.3f}\n", pnorm );
		io::print("MKL-direct: nofKernelsGood = {}\n", nofKernelsGood );
		io::print("MKL-direct: Z = {:e}\n", Z );
		io::print("MKL-direct: eps = {:e}\n", epsRegul );
		for( int32_t p=0; p<num_kernels; ++p )
		{
			const float64_t t = Math::pow( oldweights[p] - finalbeta[p], 2.0 );
			io::print("MKL-direct: t[{:3d}] = {:e}  ( diff = {:e} = {:e} - {:e} )\n", p, t, oldweights[p]-finalbeta[p], oldweights[p], finalbeta[p] );
		}
		io::print("MKL-direct: preR = {:e}\n", preR );
		io::print("MKL-direct: preR/p = {:e}\n", preR/pnorm );
		io::print("MKL-direct: sqrt(preR/p) = {:e}\n", std::sqrt(preR / pnorm));
		io::print("MKL-direct: R = {:e}\n", R );
		error("Assertion R >= 0 failed!" );
	}

	Z = 0.0;
	for( int32_t p=0; p<num_kernels; ++p )
	{
		finalbeta[p] += R;
		Z += Math::pow( finalbeta[p], pnorm );
		ASSERT( finalbeta[p] >= 0 )
	}
	Z = Math::pow( Z, -1.0/pnorm );
	ASSERT( Z >= 0 )
	for( int32_t p=0; p<num_kernels; ++p )
	{
		finalbeta[p] *= Z;
		ASSERT( finalbeta[p] >= 0.0 )
		if( finalbeta[p] > 1.0 )
			finalbeta[p] = 1.0;
	}
}

void MKLMulticlassGradient::computeweights(std::vector<float64_t> & weights2)
{
	if(pnorm<1 )
		error("MKLMulticlassGradient::computeweights(std::vector<float64_t> & weights2) : parameter pnorm<1");

	SG_DEBUG("MKLMulticlassGradient::computeweights(...): pnorm {}",pnorm)

	std::vector<float64_t> initw(weights2);
	linesearch2(weights2,initw);

	io::info("MKLMulticlassGradient::computeweights(...): newweights ");
	for(size_t i=0;i<weights2.size();++i)
	{
		io::info(" {}",weights2[i]);
	}
	io::info(" ");

	/*
	   int maxnumlinesrch=15;
	   float64_t maxdiff=1e-6;

	   bool finished =false;
	   int numiter=0;
	   do
	   {
	   ++numiter;
	   std::vector<float64_t> initw(weights2);
	   linesearch(weights2,initw);
	   float64_t norm=0;
	   if(!initw.empty())
	   {
	   for(size_t i=0;i<weights2.size();++i)
	   {
	   norm+=(weights2[i]-initw[i])*(weights2[i]-initw[i]);
	   }
	   norm=sqrt(norm);
	   }
	   else
	   {
	   norm=maxdiff+1;
	   }

	   if((norm < maxdiff) ||(numiter>=maxnumlinesrch ))
	   {
	   finished=true;
	   }
	// for(size_t i=0;i<weights2.size();++i)
	// {
	//    io::info("MKLMulticlassGradient::computeweights(...): oldweights {}",initw[i]);
	//	}
	io::info("MKLMulticlassGradient::computeweights(...): newweights at iter {} normdiff {}",numiter,norm);
	for(size_t i=0;i<weights2.size();++i)
	{
	io::info(" {}",weights2[i]);
	}
	io::info(" ");
	}
	while(false==finished);
	*/

}
