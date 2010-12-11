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
* \file AddlayerData.h Input data with additional manipulation possibilities for
* the --encode option.
*/

#ifndef __ENCODE_DATA_H
#define __ENCODE_DATA_H

#include "classifier/boosting/IO/InputData.h"

#include <vector>
#include <utility> // for pair

using namespace std;

namespace shogun {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
* Overloading of the InputData class to support the --encode option.
* \date 25/04/2007
*/
class EncodeData : public InputData
{
public:

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   * \date 25/04/2007
   */
   virtual ~EncodeData() {}

   /**
   * Clear the _data field and set _numExamples to 0.
   * \date 25/04/2007
   */
   void resetData();

   /**
   * Push back an example to the _data field and increment _numExamples.
   * Be careful: weight initialization is not done in this function.
   * \date 25/04/2007
   */
   void addExample( Example example );

};

} // end of namespace shogun

#endif // __ENCODE_DATA_H
