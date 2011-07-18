/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/classifier/mkl/MKLMultiClassGradient.h>

using namespace shogun;

MKLMultiClassGradient::MKLMultiClassGradient()
{
	numkernels = 0;
	pnorm=2;

}
MKLMultiClassGradient::~MKLMultiClassGradient()
{

}

MKLMultiClassGradient MKLMultiClassGradient::operator=(MKLMultiClassGradient & gl)
{
	numkernels=gl.numkernels;
	pnorm=gl.pnorm;
	return (*this);

}
MKLMultiClassGradient::MKLMultiClassGradient(MKLMultiClassGradient & gl)
{
	numkernels=gl.numkernels;
	pnorm=gl.pnorm;

}

void MKLMultiClassGradient::setup(const int32_t numkernels2)
{
	numkernels=numkernels2;
	if (numkernels<=1)
	{
		SG_ERROR("void glpkwrapper::setup(const int32_tnumkernels): input "
				"numkernels out of bounds: %d\n",numkernels);
	}


}

void MKLMultiClassGradient::set_mkl_norm(float64_t norm)
{
	pnorm=norm;
	if(pnorm<1 )
		SG_ERROR("MKLMultiClassGradient::set_mkl_norm(float64_t norm) : parameter pnorm<1");
}


void MKLMultiClassGradient::addconstraint(const ::std::vector<float64_t> & normw2,
		const float64_t sumofpositivealphas)
{
	normsofsubkernels.push_back(normw2);
	sumsofalphas.push_back(sumofpositivealphas);
}

void MKLMultiClassGradient::genbetas( ::std::vector<float64_t> & weights ,const ::std::vector<float64_t> & gammas)
{
	
	assert((int32_t)gammas.size()+1==numkernels);

	double pi4=3.151265358979238/4;

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

void MKLMultiClassGradient::gengammagradient( ::std::vector<float64_t> & gammagradient ,const ::std::vector<float64_t> & gammas,const int32_t dim)
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



float64_t MKLMultiClassGradient::objectives(const ::std::vector<float64_t> & weights, const int32_t index)
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


void MKLMultiClassGradient::linesearch(std::vector<float64_t> & finalbeta,const std::vector<float64_t> & oldweights)
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
			for(int32_t k=numkernels-2; k>= 1 ;--k) // k==0 not necessary
			{
				if(cos(curgamma[i-1])>0)
					tmpbeta[k]/=cos(curgamma[i-1]);
			}
		}
		
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
		for(int32_t i=1; i< (int32_t)sumsofalphas.size() ;++i)
		{
			float64_t tmpval=objectives(curbeta, i);
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


void MKLMultiClassGradient::computeweights(std::vector<float64_t> & weights2)
{
	if(pnorm<1 )
		SG_ERROR("MKLMultiClassGradient::computeweights(std::vector<float64_t> & weights2) : parameter pnorm<1");

	SG_SDEBUG("MKLMultiClassGradient::computeweights(...): pnorm %f\n",pnorm);

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
	}
	while(false==finished);


	
	
}
