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
* \file SingleStumpLearner.h A single threshold decision stump learner. 
*/

#ifndef __UCTUTILS_H
#define __UCTUTILS_H

#include "classifier/boosting/WeakLearners/FeaturewiseLearner.h"
#include "classifier/boosting/Utils/Args.h"

#include <vector>
#include <fstream>
#include <cassert>
#include <math.h>

using namespace std;

#define INITIAL_X	0.0

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace shogun {

	enum updateType 
	{
		EDGE_SQUARE, // 1-sqrt( 1-edge^2 )
		ALPHAS, // the alpha of the weak learner
		ESQUARE, //the square of edge
		LOGEDGE
	};

	
	class NodeUCTSparse;
	class InnerNodeUCTSparse;
	class LeafNodeUCTSparse;

	class NodeUCTSparse {
	protected:
		NodeUCTSparse*		_parent;		
		int					_featureID;
		int					_ni;
		double				_Xini;
		int					_myDepth;

		static int			_depth;
		static int			_branchOrder;

	public:

		virtual bool isLeaf() = 0;
		bool isRoot() { 
			if ( _parent == 0 ) return true;
			else return false;
		}

		NodeUCTSparse() : _parent( 0 ), _ni( 1 ), _Xini( INITIAL_X ), _myDepth(0) {}

		//virtual double getUpperConfidenceBound() = 0;

		virtual int getNi() { return _ni; }
		virtual double getXini() { return _Xini / ((double)_ni); }
		virtual void incNi() { _ni++; }
		virtual NodeUCTSparse* getParent() { return _parent; }
		virtual void addValue( double val ) { _Xini += val; }

		virtual double getUpperConfidenceBound() {
			return getXini() + sqrt( ( 2 * log( (double) _parent->getNi() )) / _ni );
		}
		virtual NodeUCTSparse* getithChild( int num ) = 0;
		virtual void clearInnerNodeChild() = 0;
		virtual int getNumOfChild() = 0;

		virtual int getChildIndWithMaxBi() = 0;
		static void setDepth( int d ) { NodeUCTSparse::_depth = d; }
		static int getDepth() { return NodeUCTSparse::_depth; }

		static void setBranchOrder( int bo ) { NodeUCTSparse::_branchOrder = bo; }

		virtual int getMyDepth() { return _myDepth; }
		virtual void setMyDepth( int d ) { _myDepth = d; }
	};

	class LeafNodeSparse : public NodeUCTSparse {
	public:
		LeafNodeSparse() : NodeUCTSparse() {}
		LeafNodeSparse( NodeUCTSparse* n) : NodeUCTSparse() {
			_parent = n;
			setMyDepth( n->getMyDepth() + 1 );
		}

		virtual bool isLeaf() { return true; }
		
		virtual NodeUCTSparse* getithChild( int num ) { return 0; }
		virtual void clearInnerNodeChild() {}
		virtual int getNumOfChild() {return 0;}
		virtual int getChildIndWithMaxBi() {return -1; }
	};

	class InnerNodeUCTSparse : public NodeUCTSparse {
	protected:
		vector<NodeUCTSparse*>	_children;
	public:
		virtual bool isLeaf() { return false; }
		
		InnerNodeUCTSparse() : NodeUCTSparse() { 
			_children.resize( _branchOrder );
			for( int i = 0; i < _branchOrder; i++ ) _children[i]=0;
			//fill( _children.begin(), _children.end(), 0 );
			setMyDepth( 0 ); //root
		}

		InnerNodeUCTSparse( NodeUCTSparse* n) : NodeUCTSparse() {
			_children.resize( _branchOrder );
			for( int i = 0; i < _branchOrder; i++ ) _children[i]=0;
			_parent = n;
			setMyDepth( n->getMyDepth() + 1 );
		}

		virtual void createInnerNodeChild( int num ) {
			_children.resize( num );
			for( int i = 0; i < num; i++ ) {
				_children[i] = new InnerNodeUCTSparse( (NodeUCTSparse*) this );
			}
		}

		virtual void createInnerNodeOneChild( int num ) {
			if ( _children[num] == 0 )
				_children[num] = new InnerNodeUCTSparse( (NodeUCTSparse*) this );
		}

		virtual void createLeafNodeChild( int num ) {
			_children.resize( num );
			for( int i = 0; i < num; i++ ) {
				_children[i] = new LeafNodeSparse( (NodeUCTSparse*) this );
			}
		}

		virtual void createLeafNodeOneChild( int num ) {
			if ( _children[num] == 0 )
				_children[num] = new LeafNodeSparse( (NodeUCTSparse*) this );
		}


		virtual void clearInnerNodeChild() {
			for( int i = 0; i < (int)_children.size(); i++ ) {
				if ( _children[i] ) delete _children[i];
			}		
		}

		virtual bool hasIthChild( int num ) {
			if ( _children[num] != 0 ) return true;
			else return false;
		}

		virtual NodeUCTSparse* getithChild( int num ) { return _children[num]; }
		virtual int getNumOfChild() {return (int)_children.size();}

		virtual void createRecursiveUCTTree( int depth, int numOfChilds ) {
			if ( depth == 1 ) {
				createLeafNodeChild( numOfChilds );
				return;
			} else {
				createInnerNodeChild( numOfChilds );
				for ( int i = 0; i < numOfChilds; i++ ) {
					((InnerNodeUCTSparse* )getithChild( i ))->createRecursiveUCTTree( depth - 1, numOfChilds );
				}
			}
		}

		virtual void setChildrenNum() { 
			_children.resize( _branchOrder ); 
			for( int i = 0; i < _branchOrder; i++ ) _children[i]=0;
			//fill( _children.begin(), _children.end(), 0 );
		} 

		virtual void clearRecursiveUCTTree( ) {
			if ( isLeaf() ) {
				return;
			} else {
				for ( int i = 0; i < getNumOfChild(); i++ ) {
					InnerNodeUCTSparse* currNode = (InnerNodeUCTSparse* )getithChild( i );
					if ( currNode != 0 ) currNode->clearRecursiveUCTTree( );
				}
				clearInnerNodeChild();
			}
		}

		virtual int getChildIndWithMaxBi() {
			int retVal = 0;
			double ucb = 0.0;
			double maxB = numeric_limits< double >::min();
			for( int i = 0; i < getNumOfChild(); i++ ) {
				if ( _children[i] != 0 ) ucb = _children[i]->getUpperConfidenceBound();
				else ucb = ((double) rand() / (double )RAND_MAX ) + sqrt( ( 2 * log( (double) getNi() )) / 1.0 );
				if ( maxB < ucb ) {
					retVal = i;
					maxB = ucb;
				}
			}
			return retVal;
		}


		virtual void getBestTrajectory( vector< int >& trajectory){
			trajectory.clear();

			int bInd; 
			
			/*
			bInd = getChildIndWithMaxBi();
			trajectory.push_back( bInd );

			if ( hasIthChild( bInd ) ) { 
				if ( trajectory.size() == ( _depth - 1 ) ) 
					createLeafNodeOneChild( bInd );
				else
					createInnerNodeOneChild( bInd );
			}
				
			InnerNodeUCTSparse* currNode = getithChild( bInd );
			*/

			InnerNodeUCTSparse* currNode = this;
			

			while ( 1 ) {
				bInd = currNode->getChildIndWithMaxBi();
				trajectory.push_back( bInd );

				if ( ! currNode->hasIthChild( bInd ) ) { 
					if ( trajectory.size() == ( _depth  ) )  {
						currNode->createLeafNodeOneChild( bInd );
						break;
					} else
						currNode->createInnerNodeOneChild( bInd );
				} else { 
					if ( currNode->getithChild( bInd )->isLeaf() ) break;
				}


				currNode = ( InnerNodeUCTSparse* )currNode->getithChild( bInd );
				
			}
		}

		void updateInnerNodes( double updateValue, vector< int >& trajectory ) {
			NodeUCTSparse* currNode = this;
			for( int i=0; i < (int)trajectory.size(); i++ ) {
				currNode = currNode->getithChild( trajectory[i] );
				currNode->addValue( updateValue );
				currNode->incNi();

			}

		}


	};

//////////////////////////////////////////////////////////////////////////

} // end of namespace shogun

#endif
