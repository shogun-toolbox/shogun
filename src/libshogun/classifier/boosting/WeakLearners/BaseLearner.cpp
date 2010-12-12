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


#include <cmath> // for log and exp

#include "classifier/boosting/WeakLearners/BaseLearner.h"

#include "classifier/boosting/StrongLearners/AdaBoostMHLearner.h"
//#include "StrongLearners/BrownBoostLearner.h"
//#include "StrongLearners/LogitBoostLearner.h"
#include "classifier/boosting/StrongLearners/FilterBoostLearner.h"
#include "classifier/boosting/StrongLearners/ABMHLearnerYahoo.h"

#include "classifier/boosting/Utils/Utils.h" // for is_zero

#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/IO/Serialization.h" // for the helper function "standardTag"

namespace shogun {

	// -----------------------------------------------------------------------------

	const float BaseLearner::_smallVal = 1e-7;
	int    BaseLearner::_verbose = 1;
	float BaseLearner::_smoothingVal = BaseLearner::_smallVal;

	// -----------------------------------------------------------------------------

	void BaseLearner::declareArguments(nor_utils::Args& args)
	{
	}

	// -----------------------------------------------------------------------------

	void BaseLearner::declareBaseArguments(nor_utils::Args& args)
	{
		args.declareArgument("shypname", 
			"The name of output strong hypothesis (default: "
			+ string(SHYP_NAME) + "." + string(SHYP_EXTENSION) + ").", 
			1, "<filename>");

		args.declareArgument("shypcomp", 
			"The shyp file will be compressed", 
			1, "<flag 0-1>");

		args.setGroup("Basic Algorithm Options");
		args.declareArgument("resume", 
			"Resumes a training process using the strong hypothesis file.", 
			1, "<shypFile>");   
		args.declareArgument("edgeoffset", 
			"Defines the value of the edge offset (theta) (default: no edge offset).", 
			1, "<val>");
	}

	// ------------------------------------------------------------------------------

	void BaseLearner::initLearningOptions(const nor_utils::Args& args)
	{
		if ( args.hasArgument("verbose") )
			args.getValue("verbose", 0, _verbose);

		// Set the value of theta
		if ( args.hasArgument("edgeoffset") )
			args.getValue("edgeoffset", 0, _theta);   

	}

	// -----------------------------------------------------------------------

	GenericStrongLearner* BaseLearner::createGenericStrongLearner( nor_utils::Args& args )
	{
		string sHypothesisName = "";
		if ( args.hasArgument("stronglearner") ) 
		{
			args.getValue("stronglearner", 0, sHypothesisName );
		} else
		{
			if ( _verbose > 0 ) 
			{
				cerr << "Warning: No strong learner is given. Set to default (AdaBoost)." << endl; 
			}
			sHypothesisName = "AdaBoostMH";
		}
		if ( _verbose > 0 ) 
		{
			cout << "The strong learner is " << sHypothesisName << endl; 
		}

		GenericStrongLearner* sHypothesis = NULL;

		if ( sHypothesisName.compare( "AdaBoostMH" ) == 0)
		{
			sHypothesis = new AdaBoostMHLearner();
		} else if ( sHypothesisName.compare( "BrownBoost" ) == 0 )
		{
			//sHypothesis = new BrownBoostLearner();
		} else if ( sHypothesisName.compare( "LogitBoost" ) == 0 )
		{
			//sHypothesis = new LogitBoostLearner();
		} else if ( sHypothesisName.compare( "FilterBoost" ) == 0 ) {
			sHypothesis = new FilterBoostLearner();
		} else if ( sHypothesisName.compare( "YahooBoost" ) == 0 ) {
			sHypothesis = new ABMHLearnerYahoo();
		} else {
			cout << "Unknown strong learner!!!!" << endl;
			exit( -1 );
		}
		return sHypothesis;
	}

	// -----------------------------------------------------------------------

	InputData* BaseLearner::createInputData()
	{
		return new InputData();
	}

	// -----------------------------------------------------------------------

	float BaseLearner::getAlpha(float eps_min, float eps_pls) const
	{
		return 0.5 * log( (eps_pls + _smoothingVal) / (eps_min + _smoothingVal) );
	}

	// -----------------------------------------------------------------------

	float BaseLearner::getAlpha(float eps_min, float eps_pls, 
		float theta) const
	{
		// if theta == 0
		if ( nor_utils::is_zero(theta) )
			return getAlpha( eps_min, eps_pls );

		const float eps_zero = 1 - eps_min - eps_pls;

		if (eps_min < _smallVal)
		{
			// if eps_min == 0
			return log( ( (1-theta)* eps_pls ) / (theta * eps_zero) );
		}
		else
		{
			// ln( -b + sqrt( b^2 + c) );
			const float denom = (1+theta) * eps_min;
			const float b = ((theta) * eps_zero) / (2*denom);
			const float c = ((1-theta) * eps_pls) / denom;

			return log( -b + sqrt( b * b + c ) );
		}

	}

	// -----------------------------------------------------------------------

	float BaseLearner::getEnergy(float eps_min, float eps_pls) const
	{
		return 2 * sqrt( eps_min * eps_pls ) + ( 1 - eps_min - eps_pls );
	}

	// -----------------------------------------------------------------------

	float BaseLearner::getEnergy(float eps_min, float eps_pls, 
		float alpha, float theta) const
	{
		// if theta == 0
		if ( nor_utils::is_zero(theta) )
			return getEnergy( eps_min, eps_pls );

		return exp(alpha * theta) * ( eps_min * exp(alpha) +  eps_pls * exp(-alpha)
			+  (1 - eps_min - eps_pls) );

	}

	// -----------------------------------------------------------------------

	void BaseLearner::save(ofstream& outputStream, int numTabs)
	{
		// save name
		outputStream << Serialization::standardTag("weakLearner", _name, numTabs) << endl;

		// save alpha
		outputStream << Serialization::standardTag("alpha", _alpha, numTabs) << endl;
	}

	// -----------------------------------------------------------------------

	void BaseLearner::load(nor_utils::StreamTokenizer& st)
	{
		// name should be loaded by caller so he knows what derived class to load
		// He will then call create() which copies the name

		// load alpha
		_alpha = UnSerialization::seekAndParseEnclosedValue<float>(st, "alpha");
	}

	// -----------------------------------------------------------------------

	BaseLearner* BaseLearner::copyState()
	{
		BaseLearner *pBaseLearner = subCreate();
		subCopyState(pBaseLearner);
		return pBaseLearner;
	}

	// -----------------------------------------------------------------------

	void BaseLearner::subCopyState(BaseLearner *pBaseLearner)
	{
		pBaseLearner->_theta = _theta;
		pBaseLearner->_alpha = _alpha;
		pBaseLearner->_name = _name;
		pBaseLearner->_id = _id;
		pBaseLearner->_pTrainingData = _pTrainingData;
	}

	// -----------------------------------------------------------------------

	float BaseLearner::getEdge( bool isNormalized )
	{
		float edge = 0.0;
		float sumPos = 0.0;
		float sumNeg = 0.0;

		for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) {

			vector< Label > l = _pTrainingData->getLabels( i );
			//cout << d->getRawIndex( i ) << " " << endl;

			for( vector<Label>::iterator it = l.begin(); it !=  l.end(); it++ ) {
				float cl = classify( _pTrainingData, i, it->idx );
				float tmpVal = cl * it->weight * it->y;
				if ( tmpVal >= 0.0 ) sumPos += tmpVal;
				else sumNeg -= tmpVal;
			}
		}
		//cout << endl;
		if ( isNormalized )
		{
			if ( _pTrainingData->isFiltered() ) {
				float sumEdge = sumNeg + sumPos;
				if ( ! nor_utils::is_zero( sumEdge ) ) edge = ( sumPos - sumNeg ) / sumEdge; 
			}
		} else {
			edge = sumPos - sumNeg;
		}

		return edge;
	}

  void LearnersRegs::addLearner(const string& learnerName, BaseLearner* pLearnerToRegister)
  { 
     _learners[learnerName] = pLearnerToRegister; 
     pLearnerToRegister->setName(learnerName);
  }
	// -----------------------------------------------------------------------


} // end of namespace shogun
