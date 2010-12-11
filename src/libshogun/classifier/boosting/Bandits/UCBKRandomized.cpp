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



#include "UCBKRandomized.h"
#include <limits>
#include <set>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


UCBKRandomized::UCBKRandomized( void ) : UCBK()
{
}


//----------------------------------------------------------------
//----------------------------------------------------------------


void UCBKRandomized::getKBestAction( const int k, vector<int>& bestArms )
{
	set<int> s;
	for( int i=0; i<k; i++ )
	{
		s.insert( getNextAction() );
	}
	bestArms.clear();
	for( set<int>::iterator it = s.begin(); it != s.end(); it++ )
	{
		bestArms.push_back( *it );
	}
}
//----------------------------------------------------------------
//----------------------------------------------------------------

int UCBKRandomized::getNextAction()
{
	vector< double > cumsum( getArmNumber()+1 );
	int i;

	cumsum[0] = 0.0;
	for( int i=0; i < getArmNumber(); i++ )
	{
		cumsum[i+1] = cumsum[i] + _valueRecord[i]->first;
	}

	for( i=0; i < getArmNumber(); i++ )
	{
		cumsum[i+1] /= cumsum[  getArmNumber() ];
	}

	double r = rand() / (double) RAND_MAX;

	for( i=0; i < getArmNumber(); i++ )
	{
		if ( (cumsum[i] <= r) && ( r<=cumsum[i+1] )  ) break;
	}

	return i;
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBKRandomized::updateithValue( int i )
{
	//update the value
	//_valueRecord[i]->first = _X[i] / (double) _T[i] + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
	_valueRecord[i]->first = _X[i] / (double) _T[i];// + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
	//_valueRecord[i]->first = exp( _valueRecord[i]->first );
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );
}
//----------------------------------------------------------------
//----------------------------------------------------------------


} // end namespace MultiBoost


