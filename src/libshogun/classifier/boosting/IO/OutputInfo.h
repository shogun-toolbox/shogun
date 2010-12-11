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
* \file OutputInfo.h Outputs the step-by-step information.
*/

#ifndef __OUTPUT_INFO_H
#define __OUTPUT_INFO_H

#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "classifier/boosting/IO/InputData.h"

using namespace std;

namespace MultiBoost {

// forward declaration to avoid an include
class BaseLearner;


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
* Format and output step-by-step information.
* With this class it is possible to output and update
* the error rates, margins and the edge.
* These function must be called at each iteration with the
* newly found weak hypothesis, but \b before the update of the
* weights.
* \warning Don't forget to begin the list of information
* printed with outputIteration(), and close it with
* a call to endLine()!
* \date 16/11/2005
*/
class OutputInfo
{
public:

   /**
   * The constructor. Create the object and open the output file.
   * \param outputInfoFile The name of the file which will be updated
   * with the data.
   * \date 14/11/2005
   */
   OutputInfo(const string& outputInfoFile);

   /**
   * Just output the iteration number.
   * \param t The iteration number.
   * \date 14/11/2005
   */
   void outputIteration(int t);
   void initialize(InputData* pData);

   /**
   * Just output the current time.
   * \param 
   * \date 14/11/2005
   */
   void outputCurrentTime( void );

   /**
   * Output the column names
   * \date 04/08/2006
   */
   void outputHeader();

   /**
   * Output the error of the given data.
   * The error is computed by holding the information on the previous
   * weak hypotheses. In AdaBoost, the discriminant function is computed with the formula
   * \f[
   * {\bf g}(x) = \sum_{t=1}^T \alpha^{(t)} {\bf h}^{(t)}(x),
   * \f]
   * we therefore update the \f${\bf g}(x)\f$ vector (for each example)
   * each time this method is called:
   * \f[
   * {\bf g} = {\bf g} + \alpha^{(t)} {\bf h}^{(t)}(x).
   * \f]
   * \remark There can be any number of data to have the gTable. Each one is
   * mapped into a map that uses the pointer of the data as key.
   * \param pData The input data.
   * \param pWeakHypothesis The current weak hypothesis.
   * \see table
   * \see _gTableMap
   * \date 16/11/2005
   */
   void outputError(InputData* pData, BaseLearner* pWeakHypothesis);


   
   void outputBalancedError(InputData* pData, BaseLearner* pWeakHypothesis);

   void outputROC(InputData* pData, BaseLearner* pWeakHypothesis);

   void outputMAE(InputData* pData);
   
   /**
   * Output the minimum margin the sum of below zero margins.
   * These two elements are useful for an analysis of the training process.
   *
   * The margins are represent the per-class weighted correct rate, that is
   * \f[
   * \rho_{i, \ell} = \sum_{t=1}^T \alpha^{(t)} h_\ell^{(t)}(x_i) y_i
   * \f]
   * The \b fist \b value that this method outputs is the minimum margin, that is
   * \f[
   * \rho_{min} = \mathop{\rm arg\, min}_{i, \ell} \rho_{i, \ell},
   * \f]
   * which is normalized by the sum of alpha
   * \f[
   * \frac{\rho_{min}}{\sum_{t=1}^T \alpha^{(t)}}.
   * \f]
   * This can give a useful measure of the size of the functional margin.
   *
   * The \b second \b value which this method outputs is simply the sum of the
   * margins below zero.
   * \param pData The input data.
   * \param pWeakHypothesis The current weak hypothesis.
   * \date 16/11/2005
   */
   void outputMargins(InputData* pData, BaseLearner* pWeakHypothesis);

   /**
   * Output the edge. It is the measure of the accuracy of the current 
   * weak hypothesis relative to random guessing, and is defined as
   * \f[
   * \gamma = \sum_{i=1}^n \sum_{\ell=1}^k w_{i, \ell}^{(t)} h_\ell^{(t)}(x_i)
   * \f]
   * \param pData The input data.
   * \param pWeakHypothesis The current weak hypothesis.
   * \date 16/11/2005
   */
   void outputEdge(InputData* pData, BaseLearner* pWeakHypothesis);

   /**
   * End of line in the file stream.
   * Call it when all the needed information has been outputted.
   * \date 16/11/2005
   */
   void endLine() { _outStream << endl; }

   void outUserData( float data )
   {
	   _outStream << '\t' << data;
   }


   /**
   * A table representing the votes for each example.
   * Example:
   * \verbatim
   Ex_1:  Class 0, Class 1, Class 2, .. , Class k
   Ex_2:  Class 0, Class 1, Class 2, .. , Class k
   ..
   Ex_n:  Class 0, Class 1, Class 2, .. , Class k \endverbatim
   * \date 16/11/2005
   */
   typedef vector< vector<float> > table;

   table& getTable( InputData* pData )
   {
       table& g = _gTableMap[pData];
	   return g;
   }

	void setTable( InputData* pData, table& tmpTable )
	{
		table& g = _gTableMap[pData];
		for( int i=0; i<g.size(); i++ )
			copy( tmpTable[i].begin(), tmpTable[i].end(), g[i].begin() ); 
	}

   table& getMargins( InputData* pData )
   {
       table& g = _margins[pData];
	   return g;
   }

protected:


   ofstream                _outStream; //!< The output stream 

   time_t				   _beginingTime;

   /**
   * Maps the data to its g(x) table.
   * It is needed to keep this information saved from iteration to
   * iteration.
   * \see table
   * \see outputError()
   * \date 16/11/2005
   */
   map<InputData*, table> _gTableMap; 

   /**
   * Maps the data to the margins table.
   * It is needed to keep this information saved from iteration to
   * iteration.
   * \see table
   * \see outputMargins()
   * \date 16/11/2005
   */
   map<InputData*, table>  _margins;

   /**
   * Maps the data to the sum of the alpha.
   * It is needed to keep this information saved from iteration to
   * iteration.
   * \see outputMargins()
   * \date 16/11/2005
   */
   map<InputData*, float> _alphaSums;

   
   double getROC( vector< pair< int, float > > data );

};

} // end of namespace MultiBoost

#endif // __OUTPUT_INFO_H
