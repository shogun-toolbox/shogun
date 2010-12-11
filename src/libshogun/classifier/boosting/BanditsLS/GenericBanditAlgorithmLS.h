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


#ifndef __GENERIC_BANDIT_ALGORITHM_LS_H
#define __GENERIC_BANDIT_ALGORITHM_LS_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include "classifier/boosting/Utils/Utils.h"
#include "classifier/boosting/Utils/StreamTokenizer.h"
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

/**
* An abstract class for a generic bandit algorithms.
* \see UCBK
 *\see Exp3
* \date 09/10/2009
*/

template< typename BaseType=double,typename KeyType=int>
class GenericBanditAlgorithmLS
{
//some type definition because of the templates
protected:
	typedef pair<BaseType,KeyType> 							pBaseKey;
	typedef vector<pBaseKey> 								vecBaseKey;
    typedef typename vector<pBaseKey>::iterator             itVectorPairBaseKey;
	typedef map<KeyType,BaseType>							mapKeyBase;
	typedef typename map<KeyType,BaseType>::iterator		itMapKeyBase;
	typedef typename set<KeyType>::iterator					itSetKey;

protected:
	int							_numOfArms;			// numer of the arms
	int							_numOfIter;			//number of the single stump learner have been called
	map< KeyType, BaseType >	_T;					// the number of an arm has been selected 
	map< KeyType, BaseType >	_X;					// the sum of reward for the features
	bool						_isInitialized;     // flag noting whter the object is initialized or not;

	ofstream					_rewardFile;
public:
	GenericBanditAlgorithmLS(void) : _numOfArms( -1 ), _numOfIter( 0 ), _isInitialized( false ) {}
	
	//initialize X and T vector
	void setArmNumber( int numOfArms )
	{
		//we can set the number of arm only once
		if ( _numOfArms < 0 ) 
		{		
			_numOfArms = numOfArms;
			_T.clear();
			_X.clear();
		}
	}

	virtual int getArmNumber() { return _numOfArms; }

	
	//receive reward of only one arm
	virtual void receiveReward( KeyType key, BaseType reward )
	{		
		if ( _T.find( key ) == _T.end() ) {
			_T[ key ] = static_cast<BaseType>(0);
			_X[ key ] = static_cast<BaseType>(0.0);
		}

		_T[ key ]++;
		_X[ key ] += reward;
		incIter();
		updateithValue( key );		
	}

	virtual void incIter() { _numOfIter++; }
	virtual int getIterNum() { return _numOfIter; }

	
	//abstract functions
	virtual void getKBestAction( const int k, vector<KeyType>& bestArms, KeyType defaultValue )
	{
		set<KeyType> s;
		KeyType nextAct;

		for( int i=0; i<k; i++ )
		{
			nextAct = getNextAction( defaultValue );
			if ( nextAct == defaultValue ) {
				//we should denote that we need to choose an element uniformly from the unexplored arms
				bestArms.push_back( nextAct );
			} else	{
				s.insert( nextAct );
			}
		}
		
		bestArms.clear();
		
		for( itSetKey itSet = s.begin(); itSet != s.end(); itSet++ )
		{
			bestArms.push_back( *itSet );
		}	
	}

	virtual KeyType getNextAction( KeyType defaultValue ) = 0;

	// some getter and setter
	void setInitializedFlagToTrue() { _isInitialized = true; }
	bool isInitialized( void ) { return _isInitialized; }

	//initizlaize the init values of the arms 
	virtual void initialize( map<KeyType,BaseType>& vals )
	{
		//for serialization
		setInitializedFlagToTrue();
	}

protected:
	virtual void updateithValue( KeyType key ) = 0;		
};

} // end of namespace shogun

#endif

