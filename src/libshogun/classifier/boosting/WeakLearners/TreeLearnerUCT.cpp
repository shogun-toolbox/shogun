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


#include "TreeLearnerUCT.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/Others/Example.h"
#include "classifier/boosting/Utils/StreamTokenizer.h"

#include <cmath>
#include <limits>
#include <queue>

namespace MultiBoost {

	//REGISTER_LEARNER_NAME(Product, TreeLearnerUCT)
	REGISTER_LEARNER(TreeLearnerUCT)
	int TreeLearnerUCT::_numOfCalling = 0; //number of the single stump learner have been called
	InnerNodeUCTSparse TreeLearnerUCT::_root;

	// -----------------------------------------------------------------------

	void TreeLearnerUCT::declareArguments(nor_utils::Args& args)
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
			"  or edgesquare [edgesquare]\n"
			"  Default is the first one\n",
			1, "<type>");

	}

	// ------------------------------------------------------------------------------

	void TreeLearnerUCT::initLearningOptions(const nor_utils::Args& args)
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

			vector< int > tmpVector( 2, -1 );
			_idxPairs.push_back( tmpVector );
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

	float TreeLearnerUCT::classify(InputData* pData, int idx, int classIdx)
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

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------


	float TreeLearnerUCT::run()
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

		set< int > tmpIdx, idxPos, idxNeg;
		
		_pTrainingData->clearIndexSet();
		for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) tmpIdx.insert( i );


		vector< int > trajectory(0);
		_root.getBestTrajectory( trajectory );

		// for UCT

		for(int ib = 0; ib < _numBaseLearners; ++ib)
			_baseLearners[ib]->setTrainingData(_pTrainingData);

		float edge = numeric_limits<float>::max();

		BaseLearner* pPreviousBaseLearner = 0;
		//floatBaseLearner tmpPair, tmpPairPos, tmpPairNeg;

		// for storing the inner point (learneres) which will be extended
		//vector< floatBaseLearner > bLearnerVector;
		floatInnerNode innerNode;
		priority_queue<floatInnerNode, deque<floatInnerNode>, greater_first<floatInnerNode> > pq;




		//train the first learner
		//_baseLearners[0]->run();
		pPreviousBaseLearner = _baseLearners[0]->copyState();
		((FeaturewiseLearner*)pPreviousBaseLearner)->run( trajectory[0] );

		//this contains the number of baselearners 
		int ib = 0;

		NodePoint tmpNodePoint, nodeLeft, nodeRight;

		////////////////////////////////////////////////////////
		//set the edge
		//tmpPair.first = getEdge( pPreviousBaseLearner, _pTrainingData );
		
		
		//tmpPair.second.first.first = pPreviousBaseLearner;
		// set the pointer of the parent
		//tmpPair.second.first.second.first = 0;
		// set that this is a neg child
		//tmpPair.second.first.second.second = 0;
		//tmpPair.second.second = tmpIdx;
		//bLearnerVector = calculateChildrenAndEnergies( tmpPair );


		///
		pPreviousBaseLearner->setTrainingData( _pTrainingData );
		tmpNodePoint._edge = pPreviousBaseLearner->getEdge();
		tmpNodePoint._learner = pPreviousBaseLearner;
		tmpNodePoint._idx = 0;
		tmpNodePoint._depth = 0;
		tmpNodePoint._learnerIdxSet = tmpIdx;
		calculateChildrenAndEnergies( tmpNodePoint, trajectory[1] );

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
				cout << "Delta energy: " << innerNode.first << endl << flush;
				cout << "Size of priority queue: " << pq.size() << endl << flush;
			}

			pq.pop();
			tmpNodePoint = innerNode.second;
			
			//tmpPair = bLearnerVector[0];
			//tmpPairPos = bLearnerVector[1];
			//tmpPairNeg = bLearnerVector[2];

			nodeLeft._edge = tmpNodePoint._leftEdge;
			nodeLeft._learner = tmpNodePoint._leftChild;
			nodeLeft._learnerIdxSet = tmpNodePoint._leftChildIdxSet;


			nodeRight._edge = tmpNodePoint._rightEdge;
			nodeRight._learner = tmpNodePoint._rightChild;
			nodeRight._learnerIdxSet = tmpNodePoint._rightChildIdxSet;


			//store the baselearner if the deltaenrgy will be higher
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

			nodeLeft._depth = tmpNodePoint._depth + 1;
			nodeRight._depth = tmpNodePoint._depth + 1;
						
			if ( ib > 0 ) {
				//set the descendant idx
				_idxPairs[ parentIdx ][ tmpNodePoint._leftOrRightChild ] = ib;
			}

			ib++;

			if ( ib >= _numBaseLearners ) break;


			
			//extend positive node
			if ( nodeLeft._learner ) {
				calculateChildrenAndEnergies( nodeLeft, trajectory[ nodeLeft._depth + 1 ] );
			} else {
				nodeLeft._extended = false;
			}
			

			//calculateChildrenAndEnergies( nodeLeft );

			// if the energie is getting higher then we push it into the priority queue
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
				calculateChildrenAndEnergies( nodeRight, trajectory[ nodeRight._depth + 1 ] );
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


		// set the smoothing value to avoid numerical problem
	   // when theta=0.
	   setSmoothingVal( (float)1.0 / (float)_pTrainingData->getNumExamples() * (float)0.01 );

		this->_alpha = getAlpha( eps_min, eps_pls );

		// calculate the energy (sum of the energy of the leaves
		float energy = this->getEnergy( eps_min, eps_pls );

		return energy;
}


// -----------------------------------------------------------------------


void TreeLearnerUCT::calculateChildrenAndEnergies( NodePoint& bLearner, int depthIndex ) {
	bLearner._extended = true;
	_pTrainingData->loadIndexSet( bLearner._learnerIdxSet );
	
	//separate the dataset
	set< int > idxPos, idxNeg;
	idxPos.clear();
	idxNeg.clear();
	float phix;

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
		//retval.clear();
		bLearner._extended = false;
		//return retval;
	}
	
	_pTrainingData->loadIndexSet( idxPos );
	
	if ( ! _pTrainingData->isSamplesFromOneClass() ) {
		BaseLearner* posLearner = _baseLearners[0]->copyState();

		//posLearner->run();
		((FeaturewiseLearner*)posLearner)->run( depthIndex ); 
		//
		//float posEdge = getEdge( posLearner, _pTrainingData );
		posLearner->setTrainingData( _pTrainingData );
		bLearner._leftEdge = posLearner->getEdge();

		//tmpPair.first = posEdge;
		//tmpPair.second.first.first = posLearner;
		bLearner._leftChild = posLearner;
		//set the parent idx to zero
		//tmpPair.second.first.second.first = 0;
		//this means that it will be a left child in the tree
		//tmpPair.second.first.second.second = 0;
		//tmpPair.second.second = idxPos;
		bLearner._leftChildIdxSet = idxPos;
	} else {
		BaseLearner* pConstantWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner("ConstantLearner");

		BaseLearner* posLearner = pConstantWeakHypothesisSource->create();
		posLearner->setTrainingData(_pTrainingData);
		//float constantEnergy = posLearner->run();
		((FeaturewiseLearner*)posLearner)->run( depthIndex ); 

		//BaseLearner* posLearner = _baseLearners[0]->copyState();
		//float posEdge = getEdge( posLearner, _pTrainingData );
		posLearner->setTrainingData( _pTrainingData );
		bLearner._leftEdge = posLearner->getEdge();

		//tmpPair.first = posEdge;
		//tmpPair.second.first.first = posLearner;
		bLearner._leftChild = posLearner;
		//set the parent idx to zero
		//tmpPair.second.first.second.first = 0;
		//this means that it will be a left child in the tree
		//tmpPair.second.first.second.second = 0;
		//tmpPair.second.second = idxPos;
		bLearner._leftChildIdxSet = idxPos;
	}

	//retval.push_back( tmpPair );

	_pTrainingData->loadIndexSet( idxNeg );

	if ( ! _pTrainingData->isSamplesFromOneClass() ) {
		BaseLearner* negLearner = _baseLearners[0]->copyState();

		
		//negLearner->run();
		((FeaturewiseLearner*)negLearner)->run( depthIndex ); 
		//float negEdge = getEdge( negLearner, _pTrainingData );

		negLearner->setTrainingData( _pTrainingData );
		bLearner._rightEdge = negLearner->getEdge();
		//tmpPair.first = negEdge;
		//tmpPair.second.first.first = negLearner;
		bLearner._rightChild = negLearner;
		//set the parent idx to zero
		//tmpPair.second.first.second.first = 0;
		//this means that it will be a right child in the tree
		//tmpPair.second.first.second.second = 1;
		//tmpPair.second.second = idxNeg;
		bLearner._rightChildIdxSet = idxNeg;
	} else {
		BaseLearner* pConstantWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner("ConstantLearner");

		BaseLearner* negLearner =  pConstantWeakHypothesisSource->create();
		negLearner->setTrainingData(_pTrainingData);
		//float constantEnergy = negLearner->run();
		((FeaturewiseLearner*)negLearner)->run( depthIndex ); 

		//tmpPair.first = getEdge( negLearner, _pTrainingData );;
		bLearner._rightChild = negLearner;
		bLearner._rightChild = negLearner;
		//tmpPair.second.first.first = negLearner;
		//set the parent idx to zero
		//tmpPair.second.first.second.first = 0;
		//this means that it will be a right child in the tree
		//tmpPair.second.first.second.second = 1;
		//tmpPair.second.second = idxNeg;
		bLearner._rightChildIdxSet = idxNeg;
	}

	//retval.push_back( tmpPair );

	//return retval;
}


// -----------------------------------------------------------------------

void TreeLearnerUCT::save(ofstream& outputStream, int numTabs)
{
	// Calling the super-class method
	BaseLearner::save(outputStream, numTabs);

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

void TreeLearnerUCT::load(nor_utils::StreamTokenizer& st)
{
	BaseLearner::load(st);

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

void TreeLearnerUCT::subCopyState(BaseLearner *pBaseLearner)
{
	BaseLearner::subCopyState(pBaseLearner);

	TreeLearnerUCT* pTreeLearnerUCT =
		dynamic_cast<TreeLearnerUCT*>(pBaseLearner);

	pTreeLearnerUCT->_numBaseLearners = _numBaseLearners;

	// deep copy
	for(int ib = 0; ib < _numBaseLearners; ++ib)
		pTreeLearnerUCT->_baseLearners.push_back(_baseLearners[ib]->copyState());
}

// -----------------------------------------------------------------------

} // end of namespace MultiBoost
