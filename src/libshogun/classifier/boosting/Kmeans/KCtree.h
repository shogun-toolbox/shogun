//----------------------------------------------------------------------
//	File:		KCtree.h
//	Programmer:	David Mount
//	Last modified:	08/10/2005
//	Description:	Declarations for standard kc-tree routines
//----------------------------------------------------------------------
// Copyright (C) 2004-2005 David M. Mount and University of Maryland
// All Rights Reserved.
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.  See the file Copyright.txt in the
// main directory.
// 
// The University of Maryland and the authors make no representations
// about the suitability or fitness of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//----------------------------------------------------------------------

#ifndef KC_TREE_H
#define KC_TREE_H

#include "KMeans.h"				// all k-means includes
#include "KCutil.h"				// kc-tree utilities

class KMfilterCenters;				// see KMfilterCenters.h

//----------------------------------------------------------------------
//  kc-tree - the k-center tree.
//	This is a stripped-down modification of the kd-tree of the ANN
//	library (see ann/include/ANN/ANN.h and ann/src/kd_tree.h).
//	There are a number of technical difficulties in attempting to
//	design this structure through inheritance.)  The main difference
//	is that we do not support nearest neighbor searching, but
//	instead support an operation (getNeighbors) which given a set of
//	center points, computes the subset of data points in the tree
//	that is closest to each center.
//
//	In addition to the kd-tree information, the nodes of the kc-tree
//	also store the number of data points associated with each node
//	of the tree and they keep an associated sum and sums of squares
//	for these points.  There are also a few 'hooks' inserted for
//	the eventual goal of extending this to a dynamic structure.
//
//	The tree is constructed in three phases.  The first phase is
//	borrowed from the ANN library, and builds the kc-tree for the
//	data points.  The second phase computes sum and sum of squares
//	for each node, by a simple postorder tree traversal.  The third
//	phase (which may be repeated) is given a set of centers, and
//	computes the candidates for each node in the tree.
//----------------------------------------------------------------------
class KCnode;
typedef KCnode	*KCptr;			// pointer to kc-node

class KCtree {
protected:
    int			dim;		// dimension of space
    int			n_pts;		// number of points in tree
    int			max_pts;	// max number of points in tree
    KMdataArray		pts;		// the points (of size max(n,m_max))
    KMdatIdxArray	pidx;		// point indices (to pts)
    KCptr		root;		// root of kc-tree
    KMorthRect		bnd_box;	// bounding box
//----------------------------------------------------------------------
//  Protected utilities
//  	skeletonTree	Initializes the basic tree elements (without
//  			building the tree).
//  	builtKc_tree	Recursive utility that actually builds the
//  			kc-tree from a set of points.
//----------------------------------------------------------------------
    void skeletonTree(			// construct skeleton tree
	KMdataArray	pa,		// point array (with at least n pts)
	int		n,		// number of points
	int		dd,		// dimension
	int		n_max,		// maximum number of points (optional)
	KMpoint		bb_lo,		// bounding box low point (optional)
	KMpoint		bb_hi,		// bounding box high point (optional)
	KMdatIdxArray	pi);		// point indices (optional)

    KCptr buildKcTree(		// recursive construction of kc-tree
	KMdataArray	pa,		// point array
	KMdatIdxArray	pidx,		// point indices to store in subtree
	int		n,		// number of points
	int		dim,		// dimension of space
	KMorthRect	&bnd_box);	// bounding box for current node

public:
    KCtree(				// build from point array
	KMdataArray	pa,			// point array
	int		n,			// number of points
	int		dd,			// dimension
	int		n_max = 0,		// max num of points (def = n)
	KMpoint		bb_lo = NULL,		// bounding box low point
	KMpoint		bb_hi = NULL);		// bounding box high point

    					// compute neighbors for centers
    void getNeighbors(KMfilterCenters& ctrs);

    void getAssignments(		// compute assignments for points
	KMfilterCenters&    ctrs,		// the current centers
	KMctrIdxArray 	    closeCtr,		// closest center per point
	double*	 	    sqDist);		// sq'd distance to center

    ~KCtree();				// tree destructor

    void sampleCtr(KMpoint c);		// sample a center point c

    void print(				// print the tree (for debugging)
	bool with_pts);				// print points as well?
};

//----------------------------------------------------------------------
//  Generic kc-tree node
//	Nodes in kc-trees are of two types, splitting nodes which contain
//	splitting information (a splitting hyperplane orthogonal to one
//	of the coordinate axes) and leaf nodes which contain point
//	information (an array of points stored in a bucket).  This is
//	handled by making a generic class kc-node.  The kc-node contains
//	the following basic information.
//
//		n_data		The number of data points associated
//				with this node.
//		sum		This is the sum of points (i.e., the
//				weighted centroid) of all the points
//				associated with this node.
//		sumSq		This is the sum of squares (i.e., the
//				sum of the dot products of each point
//				with itself).
//		bnd_box		Bounding box for the cell.
//
//	NOTE: The constructor does no (interesting) initialization.
//	This is handled in getSums().
//----------------------------------------------------------------------

class KCnode {			// generic kc-tree node
protected:
    const int		multCand;	// multiple candidate flag
    int			n_data;		// number of data points
    KMpoint		sum;		// sum of points
    double		sumSq;		// sum of squares
    KMorthRect		bnd_box;	// bounding box for cell
public:
    KCnode(				// basic constructor
	int		dim,		// dimension
	KMorthRect	&bb)		// bounding box
	: multCand(-1), bnd_box(dim, bb)// create bounding box
    {  sum = kmAllocPt(dim, 0); sumSq = 0; }
    	
    virtual ~KCnode();		// destructor

    void cellMidpt(KMpoint pt);		// get cell's midpoint (pt modified)

    KMorthRect &bndBox()		// get cell's bounding box
    {  return bnd_box;  }

    virtual void makeSums(		// compute sums of points
	int		&n,			// number of points (returned)
	KMpoint		&theSum,		// sum (returned)
	double		&theSumSq) = 0;		// sum of squares (returned)

    virtual void getNeighbors(		// compute neighbors for centers
	KMctrIdxArray	cands,			// candidate centers
	int		kCands) = 0;		// number of centers

    virtual void getAssignments(	// get assignments for leaf node
	KMctrIdxArray	cands,			// candidate centers
	int		kCands,			// number of centers
	KMctrIdxArray 	closeCtr,		// closest center per point
	double*	 	sqDist) = 0;		// sq'd distance to center

					// sample a center point c
    virtual void sampleCtr(KMpoint c, KMorthRect& bb) = 0;
						//
    virtual void print(int level) = 0;	// print node

    int n_nodes()			// number of nodes in this subtree
    { return 2*n_data - 1; }			// this assumes bucket size=1!

    friend class KCtree;			// allow kc-tree to access us
};

//----------------------------------------------------------------------
//  Leaf kc-tree node
//	Leaf nodes of the kc-tree store the set of points associated
//	with this bucket, stored as an array of point indices.  These
//	are indices in the array points, which resides with the
//	root of the kc-tree.  We also store the number of points
//	that reside in this bucket.
//----------------------------------------------------------------------

class KCleaf: public KCnode
{
protected:
    KMidxArray		bkt;		// bucket of points
public:
    KCleaf(				// constructor
	int		dim,		// dimension
	KMorthRect	&bb,		// bounding box
	int		n,		// number of points
	KMdatIdxArray	b)		// the bucket
	: KCnode(dim, bb)		// create kc-node
	{  assert(n <= 1);  n_data = n;  bkt = b;  }
    	
    virtual ~KCleaf() {}		// destructor (none)

    KMpoint getPoint();			// get data point

    virtual void makeSums(		// compute sums
	int		&n,			// number of points (returned)
	KMpoint		&theSum,		// sum (returned)
	double		&theSumSq);		// sum of squares (returned)

    virtual void getNeighbors(		// compute neighbors for centers
	KMctrIdxArray	cands,			// candidate centers
	int		kCands);		// number of centers

    virtual void getAssignments(	// get assignments for leaf node
	KMctrIdxArray	cands,			// candidate centers
	int		kCands,			// number of centers
	KMctrIdxArray 	closeCtr,		// closest center per point
	double*	 	sqDist);		// sq'd distance to center

					// sample a center point c
    virtual void sampleCtr(KMpoint c, KMorthRect& bb);

					// print node
    virtual void print(int level);
};

//----------------------------------------------------------------------
//  kc-tree splitting node.
//	Splitting nodes contain a cutting dimension and a cutting value.
//	These indicate the axis-parellel plane which subdivide the
//	box for this node.  The extent of the bounding box along the
//	cutting dimension is maintained (this is used to speed up point
//	to box distance calculations) [we do not store the entire bounding
//	box since this may be wasteful of space in high dimensions].
//	We also store pointers to the 2 children.
//----------------------------------------------------------------------

class KCsplit : public KCnode	// splitting node of a kc-tree
{
protected:
    int			cut_dim;	// dim orthogonal to cutting plane
    KMcoord		cut_val;	// location of cutting plane
    KMcoord		cd_bnds[2];	// lower and upper bounds of
					// rectangle along cut_dim
    KCptr		child[2];	// left and right children
public:
    KCsplit(				// constructor
	int dim,				// cutting dimension
	KMorthRect &bb,				// bounding box
	int cd,					// cutting dimension
	KMcoord cv,				// cutting value
	KMcoord lv, KMcoord hv,			// low and high values
	KCptr lc=NULL, KCptr hc=NULL)		// children
	: KCnode(dim, bb)			// create kc-node
	{
	    cut_dim	= cd;			// cutting dimension
	    cut_val	= cv;			// cutting value
	    cd_bnds[KM_LO] = lv;		// lower bound for rectangle
	    cd_bnds[KM_HI] = hv;		// upper bound for rectangle
	    child[KM_LO] = lc;			// left child
	    child[KM_HI] = hc;			// right child
	}

    virtual ~KCsplit()			// destructor
	{
	    if (child[KM_LO] != NULL) delete child[KM_LO];
	    if (child[KM_HI] != NULL) delete child[KM_HI];
	}

    virtual void makeSums(	// compute sums
	int		&n,			// number of points (returned)
	KMpoint		&theSum,		// sum (returned)
	double		&theSumSq);		// sum of squares (returned)

    virtual void getNeighbors(		// compute neighbors for centers
	KMctrIdxArray	cands,			// candidate centers
	int		kCands);		// number of centers

    virtual void getAssignments(	// get assignments for leaf node
	KMctrIdxArray	cands,			// candidate centers
	int		kCands,			// number of centers
	KMctrIdxArray 	closeCtr,		// closest center per point
	double*	 	sqDist);		// sq'd distance to center

					// sample a center point c
    virtual void sampleCtr(KMpoint c, KMorthRect& bb);

					// print node
    virtual void print(int level);
};

//----------------------------------------------------------------------
//  kc-splitting function:
//	kd_splitter is a pointer to a splitting procedure for preprocessing.
//	Different splitting procedures result in different strategies
//	for building the tree.
//----------------------------------------------------------------------

typedef void (*KMkd_splitter)(		// splitting procedure for kd-trees
    KMpointArray	pa,		// point array (unaltered)
    KMidxArray		pidx,		// point indices (permuted on return)
    const KMorthRect	&bnds,		// bounding rectangle for cell
    int			n,		// number of points
    int			dim,		// dimension of space
    int			&cut_dim,	// cutting dimension (returned)
    KMcoord		&cut_val,	// cutting value (returned)
    int			&n_lo);		// num of points on low side (returned)

#endif
