/*
*
*    MultiBoost - Multi-purpose boosting package
*
*    Copyright (C) 2010   AppStat group
*                         Laboratoire de l'Accelerateur Lineaire
*                         Universite Paris-Sud, 11, CNRS
*
*    This file is part of the MultiBoost library
*
*    This library is free software; you can redistribute it
*    and/or modify it under the terms of the GNU General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    General Public License for more details.
*
*    You should have received a copy of the GNU General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
*
*    Contact: Balazs Kegl (balazs.kegl@gmail.com)
*             Norman Casagrande (nova77@gmail.com)
*             Robert Busa-Fekete (busarobi@gmail.com)
*
*    For more information and up-to-date version, please visit
*
*                       http://www.multiboost.org/
*
*/



#ifndef _EXP3LS_LS_H
#define _EXP3LS_LS_H

#include <list> 
#include <set>
#include <functional>
//#include <pair>
#include <vector>
#include <math.h> //for pow
#include "classifier/boosting/BanditsLS/GenericBanditAlgorithmLS.h"
#include "classifier/boosting/Utils/Utils.h"
#include <cstdlib>
#include <iostream>

using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

/*
The implementation is based on this article:
@Article{ACFS02,
  author =       {Auer, P. and Cesa-Bianchi, N. and Freund, Y. and Schapire, R.E.},
  title =        {The non-stochastic multi-armed bandit problem},
  journal =      {SIAM Journal on Computing},
  year =         {2002},
  volume =       {32},
  number =   {1},
  pages =        {48--77}
}

The _X corresponds the w vector in the paper.

*/


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template< typename BaseType=double,typename KeyType=int>
class Exp3LS : public GenericBanditAlgorithmLS< BaseType, KeyType>
{
/*
using GenericBanditAlgorithmLS< BaseType, KeyType>::getArmNumber;
using GenericBanditAlgorithmLS< BaseType, KeyType>::incIter;
using GenericBanditAlgorithmLS< BaseType, KeyType>::setInitializedFlagToTrue;
using GenericBanditAlgorithmLS< BaseType, KeyType>::_X;
using GenericBanditAlgorithmLS< BaseType, KeyType>::_T;
*/
//some type definition because of the templates
protected:
	typedef pair<BaseType,KeyType> 							pBaseKey;
	typedef vector<pBaseKey> 								vecBaseKey;
    typedef typename vector<pBaseKey>::iterator             itVectorPairBaseKey;
	typedef map<KeyType,BaseType>							mapKeyBase;
	typedef typename map<KeyType,BaseType>::iterator		itMapKeyBase;
	typedef typename set<KeyType>::iterator					itSetKey;

protected:
	BaseType							_gamma;
	map<KeyType,BaseType>				_p;
	double								_Xsum;
	double								_expsum;	
	vecBaseKey							_cumSum;
public:
	Exp3LS(void) : GenericBanditAlgorithmLS<BaseType,KeyType>(),  _Xsum(0.0), _expsum(1.0)
	{
		_gamma	= static_cast<BaseType>(0.05);
	}

	virtual ~Exp3LS(void) 
	{
	}

	//----------------------------------------------------------------
	//----------------------------------------------------------------
	// getters and setters 
	//----------------------------------------------------------------
	//----------------------------------------------------------------
	BaseType getGamma() { return _gamma; }
	void setGamma( BaseType gamma ) { _gamma = gamma; }

	virtual void receiveReward( KeyType key, BaseType reward );

	virtual void initialize( map<KeyType,BaseType>& vals );

	virtual KeyType getNextAction( KeyType defaultValue );

protected:
	virtual void updateithValue( KeyType key );	
	inline virtual BaseType getPValue( KeyType key );
	/*
	double calculateXsum( void ) 
	{ 
		double sum = 0.0;
		for( itMapKeyBase it = _X.begin(); it != _X.end(); it++ ) sum += it->second;
		return sum;
	}
	*/
	
};


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
KeyType Exp3LS<BaseType,KeyType>::getNextAction( KeyType defaultValue )
{
	KeyType retVal=defaultValue;
	double r = rand() / (double) RAND_MAX;
	if ( (! _cumSum.empty() ) && ( r < _cumSum.back().first ) )
	{
		for( itVectorPairBaseKey it = _cumSum.begin(); it != _cumSum.end(); it++ )
		{
			if ( r < it->first  ) {
				retVal = it->second;			
				break;
			}
		}		
	}

	return retVal;
}

//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
void Exp3LS<BaseType,KeyType>::initialize( map<KeyType,BaseType>& vals )
{
	_p.clear();
	//according to the paper cited in the header, we set this parameter sqrt( (KlnK)/(e-1)g ) based on the Corollary 3.2
	//_alpha = pow( 4*_numOfArms*log((double)_numOfArms) * ( 1.0 / 100000 ), 1.0/3.0);
	_gamma = pow( (this->getArmNumber()*log((double)this->getArmNumber()))/( 2.0 * 100000 ) , 1.0/3.0);
	//if ( _gamma > 1.0 ) _gamma = 1.0;
	_Xsum = static_cast<double>( 0.0 );
	
	_expsum = this->getArmNumber() * exp( 1.0 / this->getArmNumber() );

	for( itMapKeyBase it = vals.begin(); it != vals.end(); it++ ) 
	{
		receiveReward( it->first, it->second );
	}
	
	this->setInitializedFlagToTrue();
}

//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
void Exp3LS<BaseType,KeyType>::updateithValue( KeyType key )
{
	itMapKeyBase it;
	_cumSum.resize( this->_X.size()+1 );
	//_cumSumKeys.resize( _p.size()+1 );
	
	int i=0;
	_cumSum[i].first = static_cast<BaseType>(0.0);

	BaseType pTemp;

	for( it=this->_X.begin(), i=0; it != this->_X.end(); it++, i++ )
	{
		pTemp = getPValue( key );
		_cumSum[i+1].first = _cumSum[i].first + pTemp;
		_cumSum[i+1].second = it->first;
	}
	//renormalize the probabilities


	//because of the numerical inaccuracy we set the last element of the cummulative sum to one if all arms have already pulled at least once
	if ( (_cumSum.size() - 1) == this->getArmNumber()) _cumSum[ _cumSum.size()- 1].first = static_cast<BaseType>(1.0);
}

//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
void Exp3LS<BaseType,KeyType>::receiveReward( KeyType key, BaseType reward )
{
	BaseType xhat;
	BaseType prevValue;

	// calculate the p value
	xhat = getPValue( key );
	
	//_X[ key ] += static_cast<BaseType>( ( _gamma * (xhat  / _numOfArms) ) ); 
	if ( this->_X.find( key ) == this->_X.end() )
	{
		prevValue = 0.0;
		this->_T[ key ]=1;
		this->_X[ key ] = static_cast<BaseType>( ( _gamma / this->getArmNumber() ) * ( reward / xhat ) );
	} else {
		prevValue = this->_X[key];
		this->_T[ key ]++;
		this->_X[ key ] += static_cast<BaseType>( ( _gamma / this->getArmNumber() ) * ( reward / xhat ) );
	}
	//update the sum of the w values
	BaseType deltaw = this->_X[ key ] - prevValue;
	_Xsum += deltaw;
	
	this->incIter();
	//update the probabilities of the arms
	updateithValue( key );		
}

//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
BaseType Exp3LS<BaseType,KeyType>::getPValue( KeyType key )
{
	if ( this->_X.find( key ) == this->_X.end() ) return static_cast<BaseType>( 1 / (double)this->getArmNumber());
	BaseType pValue=0.0;
	if ( _Xsum > 0.0 )
	{
		
		double tmpSum = ( this->getArmNumber() - this->_X.size() ) * (1.0 / this->getArmNumber());
		tmpSum += _Xsum;
		//double expSum = ( _numOfArms - _X.size() ) * exp(1.0 / _numOfArms);
		//for( map<KeyType,BaseType>::iterator it = _X.begin(); it != _X.end(); it++ ) expSum += exp( it->second / tmpSum );


		//pValue = static_cast<float>( ( 1 - _gamma ) * ( ( exp( _X[key] / tmpSum ) )/ _expSum ) + ( _gamma / _numOfArms ) );
		pValue = static_cast<BaseType>( ( 1 - _gamma ) * ( ( ( this->_X[key] / tmpSum ) ) ) + ( _gamma / this->getArmNumber() ) );
	} else {
		pValue = static_cast<BaseType>( 1 / (double)this->getArmNumber());
	}
	return pValue;
}

//----------------------------------------------------------------
//----------------------------------------------------------------


} // end of namespace shogun

#endif
