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


#include "NameMap.h"

#include "classifier/boosting/Defaults.h" // for MB_DEBUG
#include <fstream> // for ifstream
#include <iostream> // for cerr

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

// ------------------------------------------------------------------------

int NameMap::addName(const string& name)
{
   // if we haven't seen yet this name, add it to the mapping structures
   if ( _mapNameToIdx.find(name) == _mapNameToIdx.end() )
   {
      _mapNameToIdx[name] = _numRegNames;
      _mapIdxToName.push_back(name);
      ++_numRegNames;
   }
   
   return _mapNameToIdx[name];
}

// ------------------------------------------------------------------------

string NameMap::getNameFromIdx(int idx) const
{
#if MB_DEBUG
   if ( idx >= _mapIdxToName.size() )
   {
      cerr << "ERROR: trying to map a name index that does not exists. The input file" << endl;
      cerr << "might not have all the names used for training. Make sure that all names" << endl;
      cerr << "(enum attributes, classes) are correctly enumerated in the *training* file." << endl;
      exit(1);
   }
#endif
   return _mapIdxToName[idx]; 
}

// ------------------------------------------------------------------------

int NameMap::getIdxFromName(const string& name) const
{ 
#if MB_DEBUG
   if ( _mapNameToIdx.find(name) == _mapNameToIdx.end() )
   {
      cerr << "ERROR: trying to map a name (" << name << ") that does not exists. The input file" << endl;
      cerr << "might not have all the names used for training. Make sure that all names" << endl;
      cerr << "(enum attributes, classes) are correctly enumerated in the *training* file." << endl;
      exit(1);
   }
#endif
   return _mapNameToIdx[name]; 
}

// ------------------------------------------------------------------------

void NameMap::clear( void ) {
	_mapIdxToName.clear();
	_mapNameToIdx.clear();
	_numRegNames = 0;
}


// ------------------------------------------------------------------------


} // end of namespace MultiBoost

