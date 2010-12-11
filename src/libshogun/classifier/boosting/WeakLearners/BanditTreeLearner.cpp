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


#include "BanditTreeLearner.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/Others/Example.h"
#include "classifier/boosting/Utils/StreamTokenizer.h"

#include <cmath>
#include <limits>
#include <queue>

namespace shogun {

	//REGISTER_LEARNER_NAME(Product, BanditTreeLearner)
	REGISTER_LEARNER(BanditTreeLearner)

		// -----------------------------------------------------------------------

		void BanditTreeLearner::declareArguments(nor_utils::Args& args)
	{
		BanditLearner::declareArguments(args);

		args.declareArgument("baselearnertype", 
			"The name of the learner that serves as a basis for the product\n"
			"  and the number of base learners to be multiplied\n"
			"  Don't forget to add its parameters\n",
			2, "<baseLearnerType> <numBaseLearners>");
	}

	// ------------------------------------------------------------------------------

	void BanditTreeLearner::initLearningOptions(const nor_utils::Args& args)
	{
		BanditLearner::initLearningOptions(args);

		string baseLearnerName;
		args.getValue("baselearnertype", 0, baseLearnerName);   
		args.getValue("baselearnertype", 1, _numBaseLearners);   

		// get the registered weak learner (type from name)
		BaseLearner* pWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
		pWeakHypothesisSource->initLearningOptions(args);

		for( int ib = 0; ib < _numBaseLearners; ++ib ) {
			_baseLearners.push_back(pWeakHypothesisSource->create());
			_baseLearners[ib]->initLearningOptions(args);

			vector< int > tmpVector( 2, -1 );
			_idxPairs.push_back( tmpVector );
		}
	}

	// ------------------------------------------------------------------------------

	float BanditTreeLearner::classify(InputData* pData, int idx, int classIdx)
	{
		float result  = 1;
		int ib = 0;
		while ( 1 ) {
			float phix = _baseLearners[ib]->classify(pData,idx,0);
			if ( phix > 0 ) {
				if ( _idxPairs[ ib ][ 0 ] > 0 ) { 
					ib = _idxPairs[ ib ][ 0 ];
				} else {
					return _baseLearners[ib]->classify( pData, idx, classIdx ); 
				}
			} else {
				if ( _idxPairs[ ib ][ 1 ] > 0 ) { 
					ib = _idxPairs[ ib ][ 1 ];
				} else {
					return _baseLearners[ib]->classify( pData, idx, classIdx ); 
				}
			}
		}
	}


	//-------------------------------------------------------------------------------
	float BanditTreeLearner::run()
	{
		if ( ! this->_banditAlgo->isInitialized() ) {
			init();
		}
		// the bandit algorithm selects the subset the tree learner is aloowed to use
		// the armindexes will be stored in _armsForPulling
		getArms();

		//declarations
		NodePoint tmpNodePoint, nodeLeft, nodeRight;
		BaseLearner* pPreviousBaseLearner = 0;
		set< int > tmpIdx;
		//this contains the current number of baselearners 
		int ib = 0;

		floatInnerNode innerNode;
		// for storing the inner point (learneres) which will be extended
		priority_queue<floatInnerNode, deque<floatInnerNode>, greater_first<floatInnerNode> > pq;


		for(int ib = 0; ib < _numBaseLearners; ++ib)
			_baseLearners[ib]->setTrainingData(_pTrainingData);

		//train the first learner on the whole dataset 
		_pTrainingData->clearIndexSet();
		pPreviousBaseLearner = _baseLearners[0]->copyState();
		float energy = static_cast< FeaturewiseLearner* >(pPreviousBaseLearner)->run( _armsForPulling );
		if ( energy != energy )
		{
			if (_verbose > 2) {
				cout << "Cannot find weak hypothesis, constant learner is used!!" << endl;
			}

			delete pPreviousBaseLearner;
			//cannot find better than constant learner
			BaseLearner* pConstantWeakHypothesisSource = 
				BaseLearner::RegisteredLearners().getLearner("ConstantLearner");

			BaseLearner* cLearner = pConstantWeakHypothesisSource->create();
			cLearner->setTrainingData(_pTrainingData);
			float constantEnergy = cLearner->run();

			tmpNodePoint._edge = 0.0;
			tmpNodePoint._learner = cLearner;
			tmpNodePoint._idx = 0;
			tmpNodePoint._learnerIdxSet = tmpIdx;
			tmpNodePoint._extended = false;
		} else {
			//try to extend the root
			for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) tmpIdx.insert( i );
			tmpNodePoint._edge = pPreviousBaseLearner->getEdge();
			tmpNodePoint._learner = pPreviousBaseLearner;
			tmpNodePoint._idx = 0;
			tmpNodePoint._learnerIdxSet = tmpIdx;
			calculateChildrenAndEnergies( tmpNodePoint );
		}
		////////////////////////////////////////////////////////

		//insert the root into the priority queue

		if ( tmpNodePoint._extended ) 
		{
			if (_verbose > 2) {
				//cout << "Edges: (parent, pos, neg): " << bLearnerVector[0].first << " " << bLearnerVector[1].first << " " << bLearnerVector[2].first << endl << flush;
				//cout << "alpha[" << (ib) <<  "] = " << _alpha << endl << flush;
				cout << "Edges: (parent, pos, neg): " << tmpNodePoint._edge << " " << tmpNodePoint._leftEdge << " " << tmpNodePoint._rightEdge << endl << flush;
			}

			// if the energy is getting higher then we push it into the priority queue
			if ( tmpNodePoint._edge < ( tmpNodePoint._leftEdge + tmpNodePoint._rightEdge ) ) {
				float deltaEdge = abs( 	tmpNodePoint._edge - ( tmpNodePoint._leftEdge + tmpNodePoint._rightEdge ) );
				innerNode.first = deltaEdge;
				innerNode.second = tmpNodePoint;
				pq.push( innerNode ); 
			} else {
				//delete bLearnerVector[0].second.first.first;
				delete tmpNodePoint._leftChild;
				delete tmpNodePoint._rightChild;
			}

		}  

		// we cannot extend the root node
		if ( pq.empty() ) {
			// we don't extend the root			
			BaseLearner* tmpBL = _baseLearners[0];
			_baseLearners[0] = tmpNodePoint._learner;
			delete tmpBL;				
			ib = 1;
		}







		while ( ! pq.empty() && ( ib < _numBaseLearners ) ) {
			//get the best learner from the priority queue
			innerNode = pq.top();

			if (_verbose > 2) {
				cout << "Delta edge: " << innerNode.first << endl << flush;
				cout << "Size of priority queue: " << pq.size() << endl << flush;
			}

			pq.pop();
			tmpNodePoint = innerNode.second;


			nodeLeft._edge = tmpNodePoint._leftEdge;
			nodeLeft._learner = tmpNodePoint._leftChild;
			nodeLeft._learnerIdxSet = tmpNodePoint._leftChildIdxSet;


			nodeRight._edge = tmpNodePoint._rightEdge;
			nodeRight._learner = tmpNodePoint._rightChild;
			nodeRight._learnerIdxSet = tmpNodePoint._rightChildIdxSet;


			//store the baselearner if the delta enrgy will be higher
			if ( _verbose > 3 ) {
				cout << "Insert learner: " << ib << endl << flush;
			}
			int parentIdx = tmpNodePoint._parentIdx;
			BaseLearner* tmpBL = _baseLearners[ib];
			_baseLearners[ib] = tmpNodePoint._learner;				
			delete tmpBL;				

			nodeLeft._parentIdx = ib;
			nodeRight._parentIdx = ib;
			nodeLeft._leftOrRightChild = 0;
			nodeRight._leftOrRightChild = 1;

			if ( ib > 0 ) {
				//set the descendant idx
				_idxPairs[ parentIdx ][ tmpNodePoint._leftOrRightChild ] = ib;
			}

			ib++;

			if ( ib >= _numBaseLearners ) break;



			//extend positive node
			if ( nodeLeft._learner ) {
				calculateChildrenAndEnergies( nodeLeft );
			} else {
				nodeLeft._extended = false;
			}


			//calculateChildrenAndEnergies( nodeLeft );

			// if the energy is getting higher then we push it into the priority queue
			if ( nodeLeft._extended ) 
			{
				if (_verbose > 2) {
					//cout << "Edges: (parent, pos, neg): " << bLearnerVector[0].first << " " << bLearnerVector[1].first << " " << bLearnerVector[2].first << endl << flush;
					//cout << "alpha[" << (ib) <<  "] = " << _alpha << endl << flush;
					cout << "Edges: (parent, pos, neg): " << nodeLeft._edge << " " << nodeLeft._leftEdge << " " << nodeLeft._rightEdge << endl << flush;
				}

				// if the energy is getting higher then we push it into the priority queue
				if ( nodeLeft._edge < ( nodeLeft._leftEdge + nodeLeft._rightEdge ) ) {
					float deltaEdge = abs( 	nodeLeft._edge - ( nodeLeft._leftEdge + nodeLeft._rightEdge ) );
					innerNode.first = deltaEdge;
					innerNode.second = nodeLeft;
					pq.push( innerNode ); 
				} else {
					//delete bLearnerVector[0].second.first.first;
					delete nodeLeft._leftChild;
					delete nodeLeft._rightChild;

					if ( ib >= _numBaseLearners ) {
						delete nodeLeft._learner;
						break;
					} else {
						//this will be a leaf, we do not extend further
						int parentIdx = nodeLeft._parentIdx;
						BaseLearner* tmpBL = _baseLearners[ib];
						_baseLearners[ib] = nodeLeft._learner;				
						delete tmpBL;
						_idxPairs[ parentIdx ][ 0 ] = ib;
						ib += 1;
					}
				}
			} 

			//extend negative node
			if ( nodeRight._learner ) {
				calculateChildrenAndEnergies( nodeRight );
			} else {
				nodeRight._extended = false;
			}

			// if the energie is getting higher then we push it into the priority queue
			if ( nodeRight._extended ) 
			{
				if (_verbose > 2) {
					cout << "Edges: (parent, pos, neg): " << nodeRight._edge << " " << nodeRight._leftEdge << " " << nodeRight._rightEdge << endl << flush;
					//cout << "alpha[" << (ib) <<  "] = " << _alpha << endl << flush;
				}

				// if the energie is getting higher then we push it into the priority queue
				if ( nodeRight._edge < ( nodeRight._leftEdge + nodeRight._rightEdge ) ) {
					float deltaEdge = abs( 	nodeRight._edge - ( nodeRight._leftEdge + nodeRight._rightEdge ) );
					innerNode.first = deltaEdge;
					innerNode.second = nodeRight;
					pq.push( innerNode ); 
				} else {
					//delete bLearnerVector[0].second.first.first;
					delete nodeRight._leftChild;
					delete nodeRight._rightChild;

					if ( ib >= _numBaseLearners ) {
						delete nodeRight._learner;
						break;
					} else {
						//this will be a leaf, we do not extend further
						int parentIdx = nodeRight._parentIdx;
						BaseLearner* tmpBL = _baseLearners[ib];
						_baseLearners[ib] = nodeRight._learner;				
						delete tmpBL;
						_idxPairs[ parentIdx ][ 1 ] = ib;
						ib += 1;
					}
				}
			}
		}		


		for(int ib2 = ib; ib2 < _numBaseLearners; ++ib2) delete _baseLearners[ib2];
		_numBaseLearners = ib;

		if (_verbose > 2) {
			cout << "Num of learners: " << _numBaseLearners << endl << flush;
		}

		//clear the priority queur 
		while ( ! pq.empty() && ( ib < _numBaseLearners ) ) {
			//get the best learner from the priority queue
			innerNode = pq.top();
			pq.pop();
			tmpNodePoint = innerNode.second;

			delete tmpNodePoint._learner;
			delete tmpNodePoint._leftChild;
			delete tmpNodePoint._rightChild;
		}

		_id = _baseLearners[0]->getId();
		for(int ib = 0; ib < _numBaseLearners; ++ib)
			_id += "_x_" + _baseLearners[ib]->getId();

		//calculate alpha
		this->_alpha = 0.0;
		float eps_min = 0.0, eps_pls = 0.0;

		_pTrainingData->clearIndexSet();
		for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) {
			vector< Label> l = _pTrainingData->getLabels( i );
			for( vector< Label >::iterator it = l.begin(); it != l.end(); it++ ) {
				float result  = this->classify( _pTrainingData, i, it->idx );

				if ( ( result * it->y ) < 0 ) eps_min += it->weight;
				if ( ( result * it->y ) > 0 ) eps_pls += it->weight;
			}

		}

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( (float)(1.0 / _pTrainingData->getNumExamples() * 0.01 ) );

		this->_alpha = getAlpha( eps_min, eps_pls );

		// calculate the energy (sum of the energy of the leaves
		energy = this->getEnergy( eps_min, eps_pls );

		//bandit part we calculate the reward
		_reward = getRewardFromEdge( getEdge() );
		provideRewardForBanditAlgo();

		return energy;
	}
	// -----------------------------------------------------------------------

	void BanditTreeLearner::calculateChildrenAndEnergies( NodePoint& bLearner ) {
		bLearner._extended = true;
		_pTrainingData->loadIndexSet( bLearner._learnerIdxSet );

		//separate the dataset
		set< int > idxPos, idxNeg;
		idxPos.clear();
		idxNeg.clear();
		float phix;
		float energy;

		for (int i = 0; i < _pTrainingData->getNumExamples(); ++i) {
			// this returns the phi value of classifier
			phix = bLearner._learner->classify(_pTrainingData,i,0);
			if ( phix <  0 )
				idxNeg.insert( _pTrainingData->getRawIndex( i ) );
			else if ( phix > 0 ) { // have to redo the multiplications, haven't been tested
				idxPos.insert( _pTrainingData->getRawIndex( i ) );
			}
		}

		if ( (idxPos.size() < 1 ) || (idxNeg.size() < 1 ) ) {
			bLearner._extended = false;
		}

		_pTrainingData->loadIndexSet( idxPos );
		energy = numeric_limits<float>::signaling_NaN();	

		if ( ! _pTrainingData->isSamplesFromOneClass() ) {
			BaseLearner* posLearner = _baseLearners[0]->copyState();

			energy = static_cast< FeaturewiseLearner* >(posLearner)->run( _armsForPulling );
			if ( energy == energy ) {
				bLearner._leftEdge = posLearner->getEdge();

				bLearner._leftChild = posLearner;
				bLearner._leftChildIdxSet = idxPos;
			} else {
				delete posLearner;
			}
		}

		if ( energy != energy ) { //we didn't find column, this can occur when we have sparse data
			BaseLearner* pConstantWeakHypothesisSource = 
				BaseLearner::RegisteredLearners().getLearner("ConstantLearner");

			BaseLearner* posLearner = pConstantWeakHypothesisSource->create();
			posLearner->setTrainingData(_pTrainingData);
			float constantEnergy = posLearner->run();

			bLearner._leftEdge = posLearner->getEdge();
			bLearner._leftChild = posLearner;
			bLearner._leftChildIdxSet = idxPos;
		}

		_pTrainingData->loadIndexSet( idxNeg );
		energy = numeric_limits<float>::signaling_NaN();

		if ( ! _pTrainingData->isSamplesFromOneClass() ) {
			BaseLearner* negLearner = _baseLearners[0]->copyState();


			energy = static_cast< FeaturewiseLearner* >(negLearner)->run( _armsForPulling );
			if ( energy == energy ) 
			{
				bLearner._rightEdge = negLearner->getEdge();
				bLearner._rightChild = negLearner;
				bLearner._rightChildIdxSet = idxNeg;
			} else {
				delete negLearner;
			}
		}

		if ( energy != energy ) 
		{
			BaseLearner* pConstantWeakHypothesisSource = 
				BaseLearner::RegisteredLearners().getLearner("ConstantLearner");

			BaseLearner* negLearner =  pConstantWeakHypothesisSource->create();
			negLearner->setTrainingData(_pTrainingData);
			float constantEnergy = negLearner->run();

			bLearner._rightEdge = negLearner->getEdge();
			bLearner._rightChild = negLearner;

			bLearner._rightChildIdxSet = idxNeg;
		}

	}


	// -----------------------------------------------------------------------

	void BanditTreeLearner::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class method
		BanditLearner::save(outputStream, numTabs);

		// save numBaseLearners
		outputStream << Serialization::standardTag("numBaseLearners", _numBaseLearners, numTabs) << endl;

		for( int ib = 0; ib < _numBaseLearners; ++ib ) {
			outputStream << Serialization::standardTag("leftChild", _idxPairs[ib][0], numTabs) << endl;
			outputStream << Serialization::standardTag("rightChild", _idxPairs[ib][1], numTabs) << endl;
		}

		for( int ib = 0; ib < _numBaseLearners; ++ib ) {
			_baseLearners[ib]->save(outputStream, numTabs + 1);
		}
	}

	// -----------------------------------------------------------------------

	void BanditTreeLearner::load(nor_utils::StreamTokenizer& st)
	{
		BanditLearner::load(st);

		_numBaseLearners = UnSerialization::seekAndParseEnclosedValue<int>(st, "numBaseLearners");
		//   _numBaseLearners = 2;
		_idxPairs.clear();
		for(int ib = 0; ib < _numBaseLearners; ++ib) {
			int leftChild = UnSerialization::seekAndParseEnclosedValue<int>(st, "leftChild");
			int rightChild = UnSerialization::seekAndParseEnclosedValue<int>(st, "rightChild");
			vector< int > p( 2, -1 );
			p[0] = leftChild;
			p[1] = rightChild;
			_idxPairs.push_back( p );
		}


		for(int ib = 0; ib < _numBaseLearners; ++ib) {
			UnSerialization::loadHypothesis(st, _baseLearners, _pTrainingData, _verbose);
		}

	}

	// -----------------------------------------------------------------------

	void BanditTreeLearner::subCopyState(BaseLearner *pBaseLearner)
	{
		BanditLearner::subCopyState( pBaseLearner );

		BanditTreeLearner* pBanditTreeLearner =
			dynamic_cast<BanditTreeLearner*>(pBaseLearner);

		pBanditTreeLearner->_numBaseLearners = _numBaseLearners;

		// deep copy
		for(int ib = 0; ib < _numBaseLearners; ++ib)
			pBanditTreeLearner->_baseLearners.push_back(_baseLearners[ib]->copyState());
	}



} // end of namespace shogun
