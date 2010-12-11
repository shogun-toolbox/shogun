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



#ifndef _EXP3GLS_LS_H
#define _EXP3GLS_LS_H

#include <list> 
#include <set>
#include <functional>
#include <math.h> //for pow
#include "GenericBanditAlgorithmLS.h"
#include "Exp3LS.h"
#include "classifier/boosting/Utils/Utils.h"


using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/*
The implementation is based on this article:
@InProceedings{KoSz05,
  author =       {Kocsis, L.  and Szepesv\'{a}ri, Cs.},
  title =        {Reduced-{V}ariance {P}ayoff {E}stimation in {A}dversarial {B}andit {P}roblems},
  booktitle =    {Proceedings of the ECML-2005 Workshop on Reinforcement Learning in Non-Stationary Environments},
  pages =        {},
  year =         {2005}
}

The _X corresponds the w vector in the paper.

*/


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template< typename BaseType=double,typename KeyType=int>
class Exp3GLS : public Exp3LS< BaseType, KeyType>
{
/*
using GenericBanditAlgorithmLS< BaseType, KeyType>::getArmNumber;
using GenericBanditAlgorithmLS< BaseType, KeyType>::incIter;
using GenericBanditAlgorithmLS< BaseType, KeyType>::setInitializedFlagToTrue;
using GenericBanditAlgorithmLS< BaseType, KeyType>::_X;
using GenericBanditAlgorithmLS< BaseType, KeyType>::_T;
using GenericBanditAlgorithmLS< BaseType, KeyType>::getIterNum;
*/
using Exp3LS< BaseType, KeyType>::_gamma;
using Exp3LS< BaseType, KeyType>::_expsum;
using Exp3LS< BaseType, KeyType>::_Xsum;
using Exp3LS< BaseType, KeyType>::initialize;
//some type definition because of the templates
protected:
	typedef pair<BaseType,KeyType> 							pBaseKey;
	typedef vector<pBaseKey> 								vecBaseKey;
    typedef typename vector<pBaseKey>::iterator             itVectorPairBaseKey;
	typedef map<KeyType,BaseType>							mapKeyBase;
	typedef typename map<KeyType,BaseType>::iterator		itMapKeyBase;
	typedef typename set<KeyType>::iterator					itSetKey;
	typedef set<KeyType>									setKey;
	typedef map< KeyType, setKey>							mapKeySetKey;

	typedef pair<KeyType,KeyType>							pKeys;
	typedef map<pKeys,BaseType>								mapKeyPairsBase;

protected:
	mapKeyPairsBase					_sideInformation;
	mapKeySetKey					_consecutiveKeyStat; //for the efficient impelemntation we have to store the consecutive keys
	KeyType							_previousAction;
	BaseType						_eta;
public:
	Exp3GLS(void): _consecutiveKeyStat()
	{
		Exp3LS<BaseType,KeyType>();
		_gamma	= static_cast<BaseType>(0.05);
	}
	
	virtual ~Exp3GLS(void) 
	{
	}

	//----------------------------------------------------------------
	//----------------------------------------------------------------
	// getters and setters 
	//----------------------------------------------------------------
	//----------------------------------------------------------------
	KeyType getPreviousAction() { return this->_previousAction; }
	void setPreviousAction( KeyType key ) { _previousAction = key; }

	virtual void receiveReward( KeyType key, BaseType reward );
	virtual void initialize( map<KeyType,BaseType>& vals );
};


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////



//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
void Exp3GLS<BaseType,KeyType>::initialize( map<KeyType,BaseType>& vals )
{
	Exp3LS<BaseType,KeyType>::initialize( vals );

	_gamma = static_cast<BaseType>(0.1);
	_eta = 0.05;
	_sideInformation.clear();
	
	this->setInitializedFlagToTrue();
}


//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
void Exp3GLS<BaseType,KeyType>::receiveReward( KeyType key, BaseType reward )
{
	//bool newKey = false;
	if ( this->_X.find( key ) == this->_X.end() ) // we haven't pulled this arm yet
	{
		//newKey = true;
		this->_T[ key ] = 0;
		this->_X[ key ] = static_cast<BaseType>(0.0);					
	} 
	
	
	if ( this->getIterNum() == 0 )
	{
		_previousAction = key;
	}

	if ( _consecutiveKeyStat.find( _previousAction ) == _consecutiveKeyStat.end() )
	{
		set<KeyType> tmpSet;
		tmpSet.clear();
		_consecutiveKeyStat[ _previousAction ] = tmpSet;
	}
	_consecutiveKeyStat[ _previousAction ].insert( key );


	pKeys tmpPair( key, _previousAction );
	if ( _sideInformation.find( tmpPair ) == _sideInformation.end() )
		_sideInformation[tmpPair] = 1;
	else
		_sideInformation[tmpPair]++;
	
	BaseType rBelow = _sideInformation[tmpPair];
	BaseType rAbove;
	for( itSetKey it = _consecutiveKeyStat[_previousAction].begin(); it != _consecutiveKeyStat[_previousAction].end(); it++ )
	{
		KeyType tmpKey = *it;
		
		tmpPair.first = tmpKey;
		rAbove = _sideInformation[tmpPair];

		// calculate the p value
		BaseType prevValue = this->_X[ tmpKey ];

		//update the w values
		this->_T[ tmpKey ]++;
		this->_X[ tmpKey ] += static_cast<BaseType>( ( rAbove / rBelow ) *  ( _eta * reward ) ); 
		
		//update the sum of the w values
		BaseType deltaw = this->_X[ tmpKey ] - prevValue;
		_Xsum += deltaw;
		
		if ( this->_X.find( tmpKey ) == this->_X.end() ) 
		{
			_expsum -= exp( 1.0 / this->getArmNumber() );
			_expsum += exp( this->_X[ tmpKey ] );
		} else {
			_expsum -= exp( prevValue );
			_expsum += exp( this->_X[ tmpKey ] );
		}
	}

	this->incIter();
	//update the probabilities of the arms
	this->updateithValue( key );		
}


//----------------------------------------------------------------
//----------------------------------------------------------------

} // end of namespace MultiBoost
#endif
