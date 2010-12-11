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


#include "ProductLearnerUCT.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/Others/Example.h"
#include "classifier/boosting/Utils/StreamTokenizer.h"

#include <math.h>
#include <limits>

namespace shogun {

	//REGISTER_LEARNER_NAME(Product, ProductLearnerUCT)
	REGISTER_LEARNER(ProductLearnerUCT)
	int ProductLearnerUCT::_numOfCalling = 0; //number of the single stump learner have been called
	InnerNodeUCTSparse ProductLearnerUCT::_root;
	// -----------------------------------------------------------------------

		void ProductLearnerUCT::declareArguments(nor_utils::Args& args)
	{
		BaseLearner::declareArguments(args);

		args.declareArgument("baselearnertype", 
			"The name of the learner that serves as a basis for the product\n"
			"  and the number of base learners to be multiplied\n"
			"  Don't forget to add its parameters\n",
			2, "<baseLearnerType> <numBaseLearners>");

		args.declareArgument("updaterule", 
			"The update weights in the UCT can be the 1-sqrt( 1- edge^2 ) [edge]\n"
			"  or the alpha [alphas]\n"
			"  Default is the first one\n",
			1, "<type>");

	}

	// ------------------------------------------------------------------------------

	void ProductLearnerUCT::initLearningOptions(const nor_utils::Args& args)
	{
		BaseLearner::initLearningOptions(args);

		string baseLearnerName;
		args.getValue("baselearnertype", 0, baseLearnerName);   
		args.getValue("baselearnertype", 1, _numBaseLearners);   

		// get the registered weak learner (type from name)
		BaseLearner* pWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner(baseLearnerName);

		for( int ib = 0; ib < _numBaseLearners; ++ib ) {
			_baseLearners.push_back(pWeakHypothesisSource->create());
			_baseLearners[ib]->initLearningOptions(args);
		}

		string updateRule = "";
		if ( args.hasArgument( "updaterule" ) )
			args.getValue("updaterule", 0, updateRule );   

		if ( updateRule.compare( "edge" ) == 0 )
			_updateRule = EDGE_SQUARE;
		else if ( updateRule.compare( "alphas" ) == 0 )
			_updateRule = ALPHAS;
		else if ( updateRule.compare( "edgesquare" ) == 0 )
			_updateRule = ESQUARE;
		else {
			cerr << "Unknown update rule in ProductLearnerUCT (set to default [edge]" << endl;
			_updateRule = EDGE_SQUARE;
		}
	}

	// ------------------------------------------------------------------------------

	float ProductLearnerUCT::classify(InputData* pData, int idx, int classIdx)
	{
		float result  = 1;
		for( int ib = 0; ib < _numBaseLearners; ++ib )
			result *= _baseLearners[ib]->classify( pData, idx, classIdx );
		return result;
	}

	// ------------------------------------------------------------------------------

	float ProductLearnerUCT::run()
	{
		if ( _numOfCalling == 0 ) {
			if (_verbose > 0) {
				cout << "Initializing tree..." << endl;
			}
			InnerNodeUCTSparse::setDepth( _numBaseLearners );
			InnerNodeUCTSparse::setBranchOrder( _pTrainingData->getNumAttributes() );
			_root.setChildrenNum();
			//createUCTTree();
		}
		_numOfCalling++;

		const int numClasses = _pTrainingData->getNumClasses();
		const int numExamples = _pTrainingData->getNumExamples();

		// Backup original labels
		for (int i = 0; i < numExamples; ++i) {
			const vector<Label>& labels = _pTrainingData->getLabels(i);
			vector<char> exampleLabels;
			for (int l = 0; l < numClasses; ++l)
				exampleLabels.push_back(labels[l].y);
			_savedLabels.push_back(exampleLabels);
		}

		for(int ib = 0; ib < _numBaseLearners; ++ib)
			_baseLearners[ib]->setTrainingData(_pTrainingData);

		float energy = numeric_limits<float>::max();
		float previousEnergy, hx, previousAlpha;
		BaseLearner* pPreviousBaseLearner = 0;

		vector< int > trajectory(0);
		_root.getBestTrajectory( trajectory );
		
		if (_verbose > 2) {
			cout << "Trajectory: ";
			for( int i = 0; i < trajectory.size(); i++ ) {
				cout << trajectory[i] << " ";				
			}

			cout << endl;
		}


		bool firstLoop = true;
		int ib = -1;
		while (1) {
			ib += 1;
			if (ib >= _numBaseLearners) {
				ib = 0;
				firstLoop = false;
			}
			previousEnergy = energy;
			previousAlpha = _alpha;
			if (pPreviousBaseLearner)
				delete pPreviousBaseLearner;
			if ( !firstLoop ) {
				// take the old learner off the labels
				for (int i = 0; i < numExamples; ++i) {
					vector<Label>& labels = _pTrainingData->getLabels(i);
					for (int l = 0; l < numClasses; ++l) {
						// Here we could have the option of using confidence rated setting so the
						// real valued output of classify instead of its sign
						hx = _baseLearners[ib]->classify(_pTrainingData,i,l);
						if ( hx < 0 )
							labels[l].y *= -1;
						else if ( hx == 0 ) { // have to redo the multiplications, haven't been tested
							for(int ib1 = 0; ib1 < _numBaseLearners && labels[l].y != 0; ++ib1) {
								if (ib != ib1) {
									hx = _baseLearners[ib1]->classify(_pTrainingData,i,l);
									if (hx < 0)
										labels[l].y *= -1;
									else if (hx == 0)
										labels[l].y = 0;
								}
							}
						}
					}
				}
			}
			pPreviousBaseLearner = _baseLearners[ib]->copyState();
			energy = ((FeaturewiseLearner*)_baseLearners[ib])->run( trajectory[ib] );
			_alpha = _baseLearners[ib]->getAlpha();
			if (_verbose > 2) {
				cout << "E[" << (ib+1) <<  "] = " << energy << endl << flush;
				cout << "alpha[" << (ib+1) <<  "] = " << _alpha << endl << flush;
			}
			for (int i = 0; i < numExamples; ++i) {
				vector<Label>& labels = _pTrainingData->getLabels(i);
				for (int l = 0; l < numClasses; ++l) {
					// Here we could have the option of using confidence rated setting so the
					// real valued output of classify instead of its sign
					if (labels[l].y != 0) { // perhaps replace it by nor_utils::is_zero(labels[l].y)
						hx = _baseLearners[ib]->classify(_pTrainingData,i,l);
						if ( hx < 0 )
							labels[l].y *= -1;
						else if ( hx == 0 )
							labels[l].y = 0;
					}
				}
			}

			// We have to do at least one full iteration. For real it's not guaranteed
			// Alternatively we could initialize all of them to constant
			//      if ( !firstLoop && energy >= previousEnergy ) {
			//	 if (energy > previousEnergy) {
			//	    _baseLearners[ib] = pPreviousBaseLearner->copyState();
			//           delete pPreviousBaseLearner;
			//	    energy = previousEnergy;
			//	    _alpha = _baseLearners[ib]->getAlpha();
			//	 }
			//	 break;
			//      }
			if ( energy >= previousEnergy ) {
				_alpha = previousAlpha;
				energy = previousEnergy;
				if (firstLoop) {
					for(int ib2 = ib; ib2 < _numBaseLearners; ++ib2)
						delete _baseLearners[ib2];
					_numBaseLearners = ib;
				}
				else {
					_baseLearners[ib] = pPreviousBaseLearner->copyState();
				}
				delete pPreviousBaseLearner;
				break;
			} 
		}

		// Restore original labels
		for (int i = 0; i < numExamples; ++i) {
			vector<Label>& labels = _pTrainingData->getLabels(i);
			for (int l = 0; l < numClasses; ++l)
				labels[l].y = _savedLabels[i][l];
		}


		//update the weights in the UCT tree
		double updateWeight = 0.0;
		if ( _updateRule == EDGE_SQUARE ) {
			double edge = getEdge();
			updateWeight = 1 - sqrt( 1 - ( edge * edge ) );
		} else if ( _updateRule == ALPHAS ) {
			double alpha = this->getAlpha();
			updateWeight = alpha;
		} else if ( _updateRule == ESQUARE ) {
			double edge = getEdge();
			updateWeight = edge * edge;
		}

				
		_root.updateInnerNodes( updateWeight, trajectory );
		
		if (_verbose > 2) {
			cout << "Update weight (" <<  updateWeight << ")" << "\tUpdate rule index: "<< _updateRule << endl << flush;
		}
		
		_id = _baseLearners[0]->getId();
		for(int ib = 1; ib < _numBaseLearners; ++ib)
			_id += "_x_" + _baseLearners[ib]->getId();
		return energy;
	}

	// -----------------------------------------------------------------------

	void ProductLearnerUCT::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class method
		BaseLearner::save(outputStream, numTabs);

		// save numBaseLearners
		outputStream << Serialization::standardTag("numBaseLearners", _numBaseLearners, numTabs) << endl;

		for( int ib = 0; ib < _numBaseLearners; ++ib )
			_baseLearners[ib]->save(outputStream, numTabs + 1);
	}

	// -----------------------------------------------------------------------

	void ProductLearnerUCT::load(nor_utils::StreamTokenizer& st)
	{
		BaseLearner::load(st);

		_numBaseLearners = UnSerialization::seekAndParseEnclosedValue<int>(st, "numBaseLearners");
		//   _numBaseLearners = 2;

		for(int ib = 0; ib < _numBaseLearners; ++ib)
			UnSerialization::loadHypothesis(st, _baseLearners, _pTrainingData, _verbose);

	}

	// -----------------------------------------------------------------------

	void ProductLearnerUCT::subCopyState(BaseLearner *pBaseLearner)
	{
		BaseLearner::subCopyState(pBaseLearner);

		ProductLearnerUCT* pProductLearnerUCT =
			dynamic_cast<ProductLearnerUCT*>(pBaseLearner);

		pProductLearnerUCT->_numBaseLearners = _numBaseLearners;

		// deep copy
		for(int ib = 0; ib < _numBaseLearners; ++ib)
			pProductLearnerUCT->_baseLearners.push_back(_baseLearners[ib]->copyState());
	}

	// -----------------------------------------------------------------------

} // end of namespace shogun
