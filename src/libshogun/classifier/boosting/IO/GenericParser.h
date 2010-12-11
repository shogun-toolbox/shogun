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
* \file GenericParser.h Defines an abstract class for parsing files
*/

#ifndef __GENERIC_PARSER_H
#define __GENERIC_PARSER_H

#include <string>
#include <vector>
#include <algorithm>

#include "classifier/boosting/Others/Example.h"
#include "NameMap.h"
#include "InputData.h"

using namespace std;

namespace shogun
{

////////////////////////////////////////////////////////////////

class GenericParser 
{ 
public:

   GenericParser(const string& fileName)
      : _fileName(fileName), _dataRep(DR_DENSE), _labelRep(LR_DENSE), _hasWeigthInit(false) {}

   virtual void		readData(vector<Example>& examples, NameMap& classMap, 
			      vector<NameMap>& enumMaps, NameMap& attributeNameMap,
			      vector<RawData::eAttributeType>& attributeTypes) = 0;

   virtual int       getNumAttributes() const = 0;
   
   const eDataRep    getDataRep()  const { return _dataRep; }
   const eLabelRep   getLabelRep() const { return _labelRep; }

   bool  hasWeightInitialized()    const { return _hasWeigthInit; }

protected:

   const string& _fileName;
   eDataRep      _dataRep;
   eLabelRep     _labelRep;

   bool          _hasWeigthInit;

};

////////////////////////////////////////////////////////////////

};

#endif // __GENERIC_PARSER_H
