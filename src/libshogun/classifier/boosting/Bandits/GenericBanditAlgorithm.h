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


/*
*    \file GenericBanditAlgorithm.h An abstract class for a generic
*    bandit algorithm. 
*/ 


#ifndef _GENERICBANDITALGORITHM_H
#define _GENERICBANDITALGORITHM_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include "classifier/boosting/Utils/Utils.h"
#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/Utils/StreamTokenizer.h"
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* An abstract class for a generic bandit algorithms.
* \see UCBK
 *\see Exp3
* \date 09/10/2009
*/


class GenericBanditAlgorithm
{
protected:
	int					_numOfArms;			// numer of the arms
	int					_numOfIter;			//number of the single stump learner have been called
	vector< int >				_T;				// the number of an arm has been selected 
	vector< double >			_X;				// the sum of reward for the features
	bool					_isInitialized;			// flag noting whter the object is initialized or not;

	bool					_serializationFlag;
	ofstream				_rewardFile;
public:
	/**
	* The constructor. It initilaizes the variables, like number of arms and number of iteration we will take.
	* \date 30/03/2010 
	*/
	GenericBanditAlgorithm(void) : _numOfArms( -1 ), _numOfIter( 2 ), _isInitialized( false ), _serializationFlag( false ) {}
	
	/**
	* Set the number of arm and allocate memroy for array _X which contains the sum of rewards.
	* \param numOfArms The number of arms.
	* \date 30/03/2010
	*/
	void setArmNumber( int numOfArms )
	{
		//we can set the number of arm only once
		if ( _numOfArms < 0 ) 
		{		
			_numOfArms = numOfArms;
			_T.resize( _numOfArms );
			_X.resize( _numOfArms );		
		}

		fill( _T.begin(), _T.end(), 0 );
		fill( _X.begin(), _X.end(), 0.0 );
	}
	
	/**
	* Get the number of arms
	* \return The number of arms.
	* \date 30/03/2010
	*/
	virtual int getArmNumber() { return _numOfArms; }

	
	/**
	* Receive reward and updates the statistics, like the sum of reward of an arm
	* and how many times this arm has been pulled.
	* \param armNum The index of arm has been pulled.
	* \param reward The reward produced by the "environment". 
	* \date 30/03/2010
	*/
	virtual void receiveReward( int armNum, double reward )
	{
		_T[ armNum ]++;
		_X[ armNum ] += reward;
		incIter();
		updateithValue( armNum );		
	}

	/**
	* Increase the current number of iteration number.
	* \date 30/03/2010
	*/
	virtual void incIter() { _numOfIter++; }
	
	/**
	* Gets the number of current iteration number.
	* \date 30/03/2010
	*/
	virtual int getIterNum() { return _numOfIter; }

	/**
	* Write out the current arm statistics to the standard output. Only for debugging...
	* \date 30/03/2010
	*/	
	virtual void displayArmStatistic( void )
	{
		for( int i=0; i < (int) _T.size(); i++ )
		{
			cout << i << ": " << _T[i] << " " << _X[i] << endl;
		}
	}

	/**
	* Get the K best arm. Hovewer, in the original bandit framework it is only allowed 
	* to pull one arm in an iteration, here we implement that case when we can pull many
	* arms in one iteration. 
	* \param k number of arms we will pull at the same time
	* \param bestArms integer vector containing the indices of arms we pulled  
	* \date 30/03/2010
	*/	
	virtual void getKBestAction( const int k, vector<int>& bestArms )
	{
		set<int> s;
		int i = 0;
		while ( i < _numOfArms )
		{
			s.insert( getNextAction() );
			if ( ((int)s.size())>=k) break;
			i++;
		}

		bestArms.clear();
		for( set<int>::iterator it = s.begin(); it != s.end(); it++ )
		{
			bestArms.push_back( *it );
		}	
	}

	/**
	* Get the best action.
	* \date 30/03/2010
	*/
	virtual int getNextAction() = 0;

	/**
	* Set the initialization flag to true.
	* \date 30/03/2010
	*/	
	void setInitializedFlagToTrue() { _isInitialized = true; }
	
	/**
	* Get whether the is already initialized or not.
	* \date 30/03/2010
	*/
	bool isInitialized( void ) { return _isInitialized; }

	/**
	* A simple interface to initialze the member variables. This function simply
	* sets the flag because all bandit method has an own initialization process.
	* \param vals the values for the initalization
	* \date 30/03/2010
	*/
	virtual void initialize( vector< double >& vals )
	{
		setInitializedFlagToTrue();
	}


   	/**
   	* Set the arguments of the algorithm using the standard interface
   	* of the arguments. Call this to set the arguments asked by the user.
   	* \param args The arguments defined by the user in the command line.
   	* \date 10/03/2010
	*/
	virtual void initLearningOptions(const nor_utils::Args& args) = 0;

   /**
   * Declare arguments that belongs to all bandit algorithms. 
   * \remarks This method belongs only to this base class and must not
   * be extended.
   * \remarks I cannot use the standard declareArguments method, as it
   * is called only to instantiated objects, and as this class is abstract
   * I cannot do it.
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 10/2/2006
   */
   static void declareBaseArguments(nor_utils::Args& args)
	{
		args.setGroup("Bandit Algorithm Options");
		args.declareArgument("gamma", 
			"Exploation parameter.", 
			1, "<gamma>");   
		args.declareArgument("eta", 
			"Second parameter for EXP3G, EXP3.P", 
			1, "<eta>");
	}


protected:
	/**
	* Update the statics of the arms. In the case of stochastic bandit algorithms  it updates the B-values,
	* and in the case of adversarials bandit methods it updates the probabilities over the arms. 
	* \param armNum the index of arm whose statistics will be updated.
   	* \date 10/2/2006
	*/
	virtual void updateithValue( int armNum ) = 0;		
};

} // end of namespace MultiBoost

#endif
