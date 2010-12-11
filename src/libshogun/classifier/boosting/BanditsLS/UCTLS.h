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


/**
* \file SingleStumpLearner.h A single threshold decision stump learner. 
*/

#ifndef __UCT_H
#define __UCT_H

#include "GenericBanditAlgorithmLS.h"
#include "classifier/boosting/Utils/UCTutils.h"

using namespace std;


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {
	
//////////////////////////////////////////////////////////////////////////

template< typename BaseType=double,typename KeyType=string>
class UCT : public GenericBanditAlgorithmLS< BaseType, KeyType>
{
protected:
	InnerNodeUCTSparse	_root;
	locale										_loc; 

public:

	virtual void receiveReward( KeyType key, BaseType reward );
	virtual void initialize( map<KeyType,BaseType>& vals );
	virtual KeyType getNextAction( KeyType defaultValue );

	virtual void setDepth( int d ) { InnerNodeUCTSparse::setDepth( d ); }
	virtual void setOrder( int o ) { InnerNodeUCTSparse::setBranchOrder( o ); }

protected:
	virtual void updateithValue( KeyType key ){}		

};

//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
void UCT<BaseType,KeyType>::initialize( map<KeyType,BaseType>& vals )
{	
	//initialize the root
	_loc = locale(locale(), new nor_utils::white_spaces(":"));
	_root.setChildrenNum();
	this->setInitializedFlagToTrue();
}

//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
void UCT<BaseType,KeyType>::receiveReward( KeyType key, BaseType reward )
{
	int d = InnerNodeUCTSparse::getDepth();
	stringstream ss( static_cast<string>(key) );
	ss.imbue( _loc );
	vector<int> tmpArms( d );
	for( int i=0; i < d; i++ ) 
	{
		int tmpInt; 
		ss >> tmpInt;
		tmpArms[i] = tmpInt;
	}
	_root.updateInnerNodes( static_cast<double>(reward), tmpArms );
}

//----------------------------------------------------------------
//----------------------------------------------------------------
template< typename BaseType,typename KeyType>
KeyType UCT<BaseType,KeyType>::getNextAction( KeyType defaultValue )
{
	vector<int> tmpVec(0);
	_root.getBestTrajectory( tmpVec );
	string key = nor_utils::int2string( tmpVec[0] );
	for( int i=1; i < tmpVec.size(); i++ )
	{
		key.append( ":" );
		key.append( nor_utils::int2string( tmpVec[i] ) );
	}
	return (key);
}

//----------------------------------------------------------------
//----------------------------------------------------------------
} // end of namespace shogun

#endif
