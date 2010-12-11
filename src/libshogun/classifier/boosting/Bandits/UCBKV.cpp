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


#include "UCBKV.h"
#include <limits>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


UCBKV::UCBKV( void ) : GenericBanditAlgorithm(), _valuesList(0), _table( 0 ), _kszi( 1.0 ), _c(1.0/3.0), _b(1)
{
}


//----------------------------------------------------------------
//----------------------------------------------------------------


void UCBKV::getKBestAction( const int k, vector<int>& bestArms )
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

int UCBKV::getNextAction()
{
	return _valuesList.front()->second;
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBKV::initialize( vector< double >& vals )
{
	int i;

	_valueRecord.resize( _numOfArms );
		
	for( i=0; i < _numOfArms; i++ ) {
		pair< double, int >* tmpPair = new pair< double, int >(0.0,i);

		_valuesList.push_back( tmpPair );
		_valueRecord[i] = tmpPair;
	}


	//copy the initial values to X
	copy( vals.begin(), vals.end(), _X.begin() );
	
	//copy the initial values into the table
	_table.resize( _numOfArms );
	for( i = 0; i < _numOfArms; i++ ) 
	{
		_table[i].resize( 1 );
		_table[i][0] = vals[i];
	}

	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );
	
	//update the values
	for( i = 0; i < _numOfArms; i++ )
	{
		//_valueRecord[i]->first = _X[i] / (double) _T[i] + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
		_valueRecord[i]->first = _X[i] / (double) _T[i] + _c * ( ( 3 * _b * _kszi *  log( (double)getIterNum() ) )/ _T[i]); //second term is zero because of the variance is eqaul to zero
	}
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );

	setInitializedFlagToTrue();
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBKV::updateithValue( int i )
{
	//update the value
	double mean = _X[i] / (double) _T[i];
	double variance = 0.0;

	for( int j = 0; j < _T[i]; j++ )
	{
		variance += (( _table[i][j] - mean )*( _table[i][j] - mean ));
	}
	variance /= _T[i];

	_valueRecord[i]->first = mean + sqrt( ( 2.0 * _kszi * variance * log( (double)getIterNum() ) ) / _T[i] ) + 
									_c * ( ( 3 * _b * _kszi *  log( (double)getIterNum() ) )/ _T[i]) ;
	
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBKV::receiveReward( int armNum, double reward )
{
	_T[ armNum ]++;
	_X[ armNum ] += reward;
	_table[armNum].push_back( reward );
	incIter();
	updateithValue( armNum );		
}

//----------------------------------------------------------------
//----------------------------------------------------------------


} // end namespace MultiBoost
