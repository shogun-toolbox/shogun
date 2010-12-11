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




#ifndef __BANDIT_LEARNER_H
#define __BANDIT_LEARNER_H

#include "classifier/boosting/WeakLearners/BaseLearner.h"
//#include "WeakLearners/TreeLearner.h"
#include "classifier/boosting/Utils/Args.h"
#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/Utils/UCTutils.h"
#include "classifier/boosting/BanditsLS/GenericBanditAlgorithmLS.h"
#include "classifier/boosting/BanditsLS/Exp3LS.h"
#include "classifier/boosting/BanditsLS/Exp3GLS.h"


#include <vector>
#include <fstream>
#include <string>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

	//////////////////////////////////////////////////////////////////////////
	enum BanditAlgoLS
	{
		BA_RANDOM_LS, //(not implemented)
		BA_UCBK_LS, // UCBK (implemented)
		BA_UCBKV_LS, // UCBKV (implemented)
		BA_UCBKR_LS, // UCBK randomzied (implemented)
		BA_EXP3_LS, // EXP3 (implemented)
		BA_EXP3G_LS, // EXP3G (implemented)
		BA_UCT_LS
	};

	class BanditLearner : public BaseLearner
	{
	protected:
		typedef Exp3LS<double,string> Exp3LSDoubleString;
		typedef Exp3GLS<double,string> ExpG3LSDoubleString;
	public:
		BanditLearner() : BaseLearner(), _banditAlgo( NULL ) 
		{
			_loc = locale(locale(), new nor_utils::white_spaces(":"));
		}

		/**
		* The destructor. Must be declared (virtual) for the proper destruction of 
		* the object.
		*/
		virtual ~BanditLearner() {}

		/**
		* Declare weak-learner-specific arguments.
		* These arguments will be added to the list of arguments under 
		* the group specific of the weak learner. It is called
		* automatically in main, when the list of arguments is built up.
		* Use this method to declare the arguments that belongs to
		* the weak learner only.
		* 
		* This class declares the argument "rsample" only.
		* \param args The Args class reference which can be used to declare
		* additional arguments.
		* \date 19/07/2006
		*/
		virtual void declareArguments(nor_utils::Args& args);

		/**
		* Set the arguments of the algorithm using the standard interface
		* of the arguments. Call this to set the arguments asked by the user.
		* \param args The arguments defined by the user in the command line.
		* \date 14/11/2005
		* \remark These options are used for training only!
		*/
		virtual void initLearningOptions(const nor_utils::Args& args);

		/**
		* Save the current object information needed for classification,
		* that is: \a _selectedColumn, the column of the 
		* data with that yielded the lowest error
		* \param outputStream The stream where the data will be saved
		* \param numTabs The number of tabs before the tag. Useful for indentation
		* \remark To fully save the object it is \b very \b important to call
		* also the super-class method.
		* \see AbstainableLearner::save()
		* \date 13/11/2005
		*/
		virtual void save(ofstream& outputStream, int numTabs = 0);

		/**
		* Load the xml file that contains the serialized information
		* needed for the classification and that belongs to this class
		* \param st The stream tokenizer that returns tags and values as tokens
		* \see save()
		* \date 13/11/2005
		*/
		virtual void load(nor_utils::StreamTokenizer& st);

		/**
		* Copy all the info we need in classify().
		* pBaseLearner was created by subCreate so it has the correct (sub) type.
		* Usually one must copy the same fields that are loaded and saved. Don't 
		* forget to call the parent's subCopyState().
		* \param pBaseLearner The sub type pointer into which we copy.
		* \see save
		* \see load
		* \see classify
		* \see ProductLearner::run()
		* \date 25/05/2007
		*/
		virtual void subCopyState(BaseLearner *pBaseLearner);



		////////////////////////////////////////////////////////////////////////////
		// For bandits
		////////////////////////////////////////////////////////////////////////////
		//getter and setter of the bandit algorithm
		virtual GenericBanditAlgorithmLS<double,string>* getBanditAlgoObject() { return _banditAlgo; }
		virtual void setBanditAlgoObject( GenericBanditAlgorithmLS<double,string>* banditAlgo ) { _banditAlgo = banditAlgo; }
		virtual BanditAlgoLS getAlgoType() { return _banditAlgoName; }
		virtual updateType getUpdateRule() { return _updateRule; }
		virtual double getReward() { return _reward; }
		virtual int getK() { return _K; }
		virtual void getArmsForPulling( vector<int>& arms ) { arms = _armsForPulling; };

		virtual void init();
		virtual void getArms( void );
		virtual void provideRewardForBanditAlgo( void );
		virtual double getRewardFromEdge( float edge );
		virtual string getKeyToString();
		////////////////////////////////////////////////////////////////////////////
		
	protected:
		////////////////////////////////////////////////////////////////////////////
		// For bandits
		////////////////////////////////////////////////////////////////////////////
		enum updateType								_updateRule;
		double										_reward; //the rewards of the learner

		GenericBanditAlgorithmLS<double,string>*	_banditAlgo;
		BanditAlgoLS								_banditAlgoName;

		vector<int>									_armsForPulling; //which feature are allowed to use
		int											_K; // the number of the features to be used
		locale										_loc; 
		////////////////////////////////////////////////////////////////////////////
	};

} // end of namespace shogun


#endif // __PRODUCT_LEARNER_H
