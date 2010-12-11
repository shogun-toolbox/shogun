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



#ifndef _Exp3G2_H
#define _Exp3G2_H

#include <list> 
#include <functional>
#include <math.h> //for pow
#include "GenericBanditAlgorithm.h"
#include "Exp3G.h"
#include "classifier/boosting/Utils/Utils.h"

/*
The Exp3G2 Algorithm was published in: 
Kocsis and Szepesvari:
Reduced-Variance Payoff Estimation in Adversarial Bandit Problems
(it isn't available the journal where it was appeared, but it was written that it is published in the proceedings of ECML05)
*/


using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {


class Exp3G2 : public Exp3G
{
public:
	Exp3G2(void);
	virtual ~Exp3G2(void) 
	{
	}

	virtual void receiveReward( int armNum, double reward );
	virtual void receiveReward( vector<double> reward );

	virtual void initialize( vector< double >& vals );

};


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


} // end of namespace shogun

#endif


