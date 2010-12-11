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


#ifndef __EXAMPLE_RESULTS_H
#define __EXAMPLE_RESULTS_H

#include <vector>
#include "classifier/boosting/Others/Example.h" // for Example

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

/**
* Holds the results per example.
* This class holds all the results obtained with computeResults(), 
* which is equivalent to \f${\bf g}(x)\f$. It also offers two methods
* that allow the quick evaluation of the results.
* \date 16/11/2005
*/
class ExampleResults 
{
public:

   /**
   * The constructor. Initialize the index and the votes vector.
   * \param idx The index of the example.
   * \param numClasses The number of classes.
   * \date 16/11/2005
   */
   ExampleResults(const int idx, const int numClasses) 
      : _idx(idx), _votesVector(numClasses, 0) {}

   const int getIdx() { return _idx; }

   vector<float>& getVotesVector() { return _votesVector; }

   /**
   * Get the winner. 
   * \param rank The rank. 0 = winner. 1 = second, etc..
   * \return A pair <\f$\ell\f$, \f$g_\ell(x)\f$>, where \f$\ell\f$ is the class index.
   * \date 16/11/2005
   */
   pair<int, float> getWinner(int rank = 0);

   /**
   * Checks if the given class is the winner class.
   * Example: if the ranking is 5 2 6 3 1 4 (in class indexes):
   * \code
   * isWinner(5,0); // -> true
   * isWinner(2,0); // -> false
   * isWinner(2,1); // -> true
   * isWinner(3,3); // -> true
   * \endcode
   * \param idxRealClass The index of the actual class.
   * \param atLeastRank The maximum rank in which the class must be
   * to be considered a "winner".
   * \date 16/11/2005
   */
   bool isWinner(const Example& example, int atLeastRank = 0) const;

private:

   /**
   * Create a sorted ranking list. It uses votesVector to build a vector
   * of pairs that contains the index of the class and the value of the votes
   * (that is a vector of <\f$\ell\f$, \f$g_\ell(x)\f$>), which is sorted
   * by the second element, resulting in a ranking of the votes per class.
   * \param rankedList the vector that will be filled with the rankings.
   * \date 16/11/2005
   */
   void getRankedList( vector< pair<int, float> >& rankedList ) const;

   const int _idx; //!< The index of the example 

   /**
   * The vector with the results. Equivalent to what returned by \f${\bf g}(x)\f$.
   * \remark It is public because the methods of Classifier will access it
   * directly.
   */
   vector<float> _votesVector; 
  
   /**
   * Fake assignment operator to avoid warning.
   * \date 6/03/2006
   */
   ExampleResults& operator=( const ExampleResults& ) {return *this;}

}; // ExampleResults

} // end of namespace shogun

#endif // __EXAMPLE_RESULTS_H
