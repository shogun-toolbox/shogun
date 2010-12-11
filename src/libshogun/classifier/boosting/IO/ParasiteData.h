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
* \file ParasiteData.h Input data which has the column sorted.
*/

#ifndef __PARASITE_DATA_H
#define __PARASITE_DATA_H

#include "InputData.h"
#include "WeakLearners/BaseLearner.h"

#include <vector>

using namespace std;

namespace MultiBoost {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
* Overloading of the InputData class to support ParasiteLearner.
* it stores a pool of baselearners already learned
* \date 24/04/2007
*/
class ParasiteData : public InputData
{
public:

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~ParasiteData() {}

   int getNumBaseLearners() const 
   { return _baseLearners.size(); }   //!< Returns the number of base learners

   BaseLearner getBaseLearner(int i) const 
   { return _baseLearners[i]; }   //!< Returns the number of base learners

   void addBaseLearner(const BaseLearner& baseLearner) 
   { _baseLearners.push_back(baseLearner); }   //!< Returns the number of base learners

protected:

   vector<BaseLearner>    _baseLearners; //!< the pool of base learners

};

} // end of namespace MultiBoost

#endif // __PARASITE_DATA_H
