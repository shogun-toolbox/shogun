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



#include "UCBK.h"
#include <limits>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


UCBK::UCBK( void ) : GenericBanditAlgorithm(), _valuesList(0)
{
}


//----------------------------------------------------------------
//----------------------------------------------------------------


void UCBK::getKBestAction( const int k, vector<int>& bestArms )
{
	bestArms.resize(k);
	int i=0; 
	for( list< pair< double,int>* >::iterator it = _valuesList.begin(); i<k; it++, i++ )
	{
		bestArms[i] = (**it).second;
	}
}
//----------------------------------------------------------------
//----------------------------------------------------------------

int UCBK::getNextAction()
{
	return _valuesList.front()->second;
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBK::initialize( vector< double >& vals )
{
	_valueRecord.resize( _numOfArms );
		
	for( int i=0; i < _numOfArms; i++ ) {
		pair< double, int >* tmpPair = new pair< double, int >(0.0,i);

		_valuesList.push_back( tmpPair );
		_valueRecord[i] = tmpPair;
	}


	//copy the initial values to X
	copy( vals.begin(), vals.end(), _X.begin() );
	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );
	
	//update the values
	for( int i = 0; i < _numOfArms; i++ )
	{
		_valueRecord[i]->first = _X[i] / (double) _T[i] + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
	}
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );

	setInitializedFlagToTrue();
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBK::updateithValue( int i )
{
	//update the value
	_valueRecord[i]->first = _X[i] / (double) _T[i] + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );
}

//----------------------------------------------------------------
//----------------------------------------------------------------


} // end namespace MultiBoost

