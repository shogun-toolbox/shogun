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



#include "Exp3G2.h"
#include <limits>
#include <set>
#include <math.h>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


Exp3G2::Exp3G2( void ) : Exp3G()
{
	_gamma = 0.05;
	_eta = 0.4;
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void Exp3G2::initialize( vector< double >& vals )
{
	_p.resize( _numOfArms );
	_w.resize( _numOfArms );
	_tmpW.resize( _numOfArms );

	fill( _p.begin(), _p.end(), 1.0 / _numOfArms );
	fill( _w.begin(), _w.end(), 1.0 );


	copy( vals.begin(), vals.end(), _X.begin() );
	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );
	
	
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] = _eta * _X[i];
	}

	setInitializedFlagToTrue();
}



//----------------------------------------------------------------
//----------------------------------------------------------------
void Exp3G2::receiveReward( int armNum, double reward )
{

	_T[ armNum ]++;
	// calculate the feedback value

	incIter();
	 
	//update 
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] += ( _eta * reward );
	}

	//_w[armNum] += ( _eta * reward );

	updateithValue( armNum );		
}

//----------------------------------------------------------------
//----------------------------------------------------------------
void Exp3G2::receiveReward( vector<double> reward )
{
	incIter();
	 
	//update 
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] += ( _eta * reward[i] );
	}

	//_w[armNum] += ( _eta * reward );

	updateithValue( 0 );		
}


} // end namespace MultiBoost
