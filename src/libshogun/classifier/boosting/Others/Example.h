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
* \file Example.h Defines what a single example is.
*/
#pragma warning( disable : 4786 )



#ifndef __MB_EXAMPLE_H
#define __MB_EXAMPLE_H

#include <string>
#include <vector>
#include <algorithm>
#include <map>
//#define NOTIWEIGHT

using namespace std;

namespace MultiBoost
{

// -----------------------------------------------------------------------------

/**
* Defines the data representation
*/
enum eDataRep
{
   DR_UNKNOWN,
   DR_DENSE,
   DR_SPARSE
};

/**
* Defines the label representation.
*/
enum eLabelRep
{
   LR_UNKNOWN,
   LR_DENSE,
   LR_SPARSE
};

// -----------------------------------------------------------------------------

struct Label
{
#ifdef NOTIWEIGHT
   Label() : weight(1) {}   
#else
   Label() : weight(1), initialWeight(1) {}
#endif

   // note: for LR_MULTI_DENSE idx is the same value as the actual index in the vector
   int      idx; 
   float   weight;

#ifndef NOTIWEIGHT
   float   initialWeight;
#endif

   char   y; // +1/-1 for the moment

   bool operator== (const int idx) const
   { return this->idx == idx; }
};

// -----------------------------------------------------------------------------

/**
* Holds the    data of the single example.
* \date 11/11/2005
*/
class Example
{
public:
   /**
   * The constructor that create the object example.
   * \param name The name of the example.
   * \date 11/11/2005
   */
   Example( const string& name = "" ) 
      : _name(name) {  } 

   inline const string& getName() const { return _name; }
   inline void          setName(const string& name ) { _name = name; }

   //////////////////////////////////////////////////////////////////////////

   //inline const vector< pair<int, float> >& getWeights() const { return _weights; }
   //inline       vector< pair<int, float> >& getWeights()       { return _weights; }

   //////////////////////////////////////////////////////////////////////////
   inline void addBinaryLabel(int labelIdx)
   { 
      _labels.resize(2); // at least size 2!
      _labels[0].idx = 0;
      _labels[1].idx = 1;

      if ( labelIdx == 0 )
      {
         _labels[0].y = +1;
         _labels[1].y = -1;
      }
      else
      {
         _labels[0].y = -1;
         _labels[1].y = +1;
      }
   }

   inline void addLabels(vector<Label>& labels)
   { swap(_labels, labels); } // should be done in constant time

   inline void addValues(vector<float>& values)
   { swap(_values, values); } // should be done in constant time

   inline const vector<Label>& getLabels() const { return _labels; }
   inline       vector<Label>& getLabels()       { return _labels; }

   inline bool hasLabel(const int labelIdx) const
   { return (find(_labels.begin(), _labels.end(), labelIdx) != _labels.end()); }

   // kinda slow
   inline bool hasPositiveLabel(const int labelIdx) const
   {
      vector<Label>::const_iterator it;
      for ( it = _labels.begin(); it != _labels.end(); ++it )
      {
         if ( it->idx == labelIdx && it->y > 0 )
            return true;
      }
      return false;
   }


   //////////////////////////////////////////////////////////////////////////

   inline const vector<float>& getValues() const { return _values; }
   inline       vector<float>& getValues()       { return _values; }

   // this return the indexes of the data (if it is sparse, otherwise the
   // vector is just empty
   inline const vector<int>&    getValuesIndexes() const { return _valIdxs; }
   inline       vector<int>&    getValuesIndexes()       { return _valIdxs; }

   inline  const map<int,int>&  getValuesIndexesMap() const { return _valIdxsMap; }
   inline		 map<int,int>&  getValuesIndexesMap()       { return _valIdxsMap; }

   //////////////////////////////////////////////////////////////////////////

private:

   /**
   * The labels (indexes and +1/-1) of the example, which contains the index of the
   * label and the weight.
   * i.e.
   * Single label: where the first class is true:
   * { <0,0.223,+1>, <1,0.113,-1> } 
   * Multiple (dense) labels: where there are three labels, and the first two are true:
   * { <0,0.123,+1>, <1,0.113,+1>, <2,0.08,-1> } 
   * Multiple (sparse) labels: where there are 1000 labels, and label 344 and 552 are true:
   * { <344,0.33,+1>, <552,0.544,+1> }
   * \remark A possible optimization would be to replace the int with a short
   * and limit the number of labels to 65k
   */
   vector<Label>  _labels; //!< The labels (index) of the example.

   vector<float> _values; //!< The values of the example.

   /**
   * The (column) indexes of the values of the example. 
   * If this vector is empty the data is dense!
   * example of sparse data:
   * @data
   * {1 X, 3 Y, 4 -1}
   * (we ignore the label at column 4 for the moment)
   * _values would be  = {X, Y}
   * _valIdxs would be = {1, 3}
   * \remark This might be a heavy memory footprint for dense data, since it that case it
   * is simply not used. For each example the total memory usage (empty) is 16 bytes.
   */
   vector<int>    _valIdxs; 
   map<int,int>	  _valIdxsMap;
   string         _name; //<! The name of the example

   //void print(ostream &);
};

} // end of namespace MultiBoost

#endif // 
