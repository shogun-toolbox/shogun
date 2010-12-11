//----------------------------------------------------------------------
//	File:		KCtree.cc
//	Programmer:	David Mount
//	Last modified:	08/10/2005
//	Description:	Basic methods for kc-trees.
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

#include "KCtree.h"			// kc-tree declarations
#include "KMfilterCenters.h"		// center set structure
#include "KMrand.h"			// random number includes

//----------------------------------------------------------------------
//  Declaration of local utilities.  These are used in getNeighbors().
//----------------------------------------------------------------------
static int closestToBox(		// get closest point to box center
    KMctrIdxArray	cands,			// candidates for closest
    int			kCands,			// number of candidates
    KMorthRect		&bnd_box);		// bounding box of cell

static bool pruneTest(			// test whether to prune candidate
    KMcenter		cand,			// candidate to test
    KMcenter		closeCand,		// closest candidate
    KMorthRect		&bnd_box);		// bounding box

static void postNeigh(			// assign neighbors to center
    KCptr		p,			// the node posting
    KMpoint		sum,			// the sum of coordinates
    double		sumSq,			// the sum of squares
    int			n_data,			// number of points
    KMctrIdx		ctrIdx);		// center index

//----------------------------------------------------------------------
//  KCtree constructors
//	There is a skeleton kc-tree constructor which does (almost)
//	all the initialization, without actually building the tree.
//	The arguments are essentially the same as the constructor
//	for the kc-tree.  The last argument (pi) is an optional array
//	for the point indices.  Normally, this will be NULL (meaning
//	that the constructor should initialize the array of indices),
//	but for the load constructor the point indices will be set
//	created and passesd through this argument.
//----------------------------------------------------------------------

void KCtree::skeletonTree(		// construct skeleton tree
    KMdataArray		pa,		// point array (with at least n pts)
    int			n,		// number of points
    int			dd,		// dimension
    int			n_max,		// maximum number of points (optional)
    KMpoint		bb_lo,		// bounding box low point (optional)
    KMpoint		bb_hi,		// bounding box high point (optional)
    KMdatIdxArray	pi)		// point indices (optional)
{
    					// initialize basic elements
    dim = dd;				// dimension
    n_pts = n;				// number of points
    if (n_max < n) n_max = n;		// max_pts must be >= n
    max_pts = n_max;			// set max_pts

    if (pa == NULL) {			// no points supplied?
    	kmError("Points must be supplied to construct tree.", KMabort);
    }
    pts = pa;				// initialize point array

    if (pi == NULL) {			// point indices provided?
	pidx = new KMdataIdx[max_pts];	// no, allocate them
	for (int i = 0; i < n; i++)	// initialize to identity
	    pidx[i] = i;
    }
    else pidx = pi;			// yes, just use them

    			
    if (bb_lo == NULL || bb_hi == NULL) // boundng box fully specified?
	kmEnclRect(pa, pidx, n, dd, bnd_box);	// no, construct from points
					// save bounding box
    if (bb_lo != NULL)			// if lower point given, then use it
    	bnd_box.lo = kmAllocCopyPt(dd, bb_lo);

    if (bb_hi != NULL)			// same for upper point
    	bnd_box.hi = kmAllocCopyPt(dd, bb_hi);

    root = NULL;			// no associated tree yet
}

//----------------------------------------------------------------------
//  BasicGlobals - global variables
// 	To prevent long argument lists in a number of the tree traversal
// 	programs, we store a number of common global variables here.
// 	In the case of construction, these are initialized before
// 	calling buildKcTree.  They are used in getNeighbors() and by
// 	sampleCtr().
//----------------------------------------------------------------------

int		kcDim;			// dimension of space
int		kcDataSize;		// number of data points
KMdataArray	kcPoints;		// data points

//----------------------------------------------------------------------
//  initBasicGlobals - initialize basic globals
//----------------------------------------------------------------------

static void initBasicGlobals(		// initialize basic globals
    int			dim,			// dimension
    int			data_size,		// number of data points
    KMdataArray		data_pts)		// data points
{
    kcDim = dim;
    kcDataSize = data_size;
    kcPoints = data_pts;
}

//----------------------------------------------------------------------
// kc-tree constructor
//	This is the main constructor for kc-trees given a set of
//	points.  It first builds a skeleton tree, then computes the
//	bounding box, and then invokes buildKcTree() to actually
//	build the tree.  It passes in the appropriate splitting
//	routine.
//
//	The constructor has a number of optional arguments.
//
//	n_max		Max number of points.  (default: n).
//	bb_lo, bb_hi	Bounding box low and high points (default:
//			compute bounding box from points).
//
//	As long as the number of points is nonzero, or if a bounding
//	box is provided, then the constructor will build a tree with
//	at least one node (even an empty leaf).  Otherwise, it returns
//	with a null tree.
//
//	Under the current implementation, point insertion generates
//	leaf nodes with a bucket size of 1.  If the requested bucket
//	size is higher, and it looks like insertion is requested, then
//	we generate a warning message.
//----------------------------------------------------------------------

KCtree::KCtree(			// construct from point array
    KMdataArray		pa,		// point array (with at least n pts)
    int			n,		// number of points
    int			dd,		// dimension
    int			n_max,		// maximum number of points (optional)
    KMpoint		bb_lo,		// bounding box low point (optional)
    KMpoint		bb_hi) :	// bounding box high point (optional)
    bnd_box(dd)				// create initial bounding box
{
    					// set up the basic stuff
    skeletonTree(pa, n, dd, n_max, bb_lo, bb_hi, NULL);
    initBasicGlobals(dd, n, pa);	// initialize globals

    root = buildKcTree(pa, pidx, n, dd, bnd_box);

    int ignoreMe1;			// ignore results of call
    KMpoint ignoreMe2;
    double ignoreMe3;
    					// compute sums
    root->makeSums(ignoreMe1, ignoreMe2, ignoreMe3);
    assert(ignoreMe1 == n);		// should be all the points
}

//----------------------------------------------------------------------
//  buildKcTree - recursive procedure to build a kc-tree
//	Builds a kc-tree for points in pa as indexed through the array
//	pidx[0..n-1] (typically a subarray of the array used in the
//	top-level call).  This routine permutes the array pidx, but does
//	not alter pa[].
//
//	The construction is based on a standard algorithm for
//	constructing the kc-tree (see Friedman, Bentley, and Finkel,
//	``An algorithm for finding best matches in logarithmic expected
//	time,'' ACM Transactions on Mathematical Software, 3(3):209-226,
//	1977).  The procedure operates by a simple divide-and-conquer
//	strategy, which determines an appropriate orthogonal cutting
//	plane (see below), and splits the points.  When the number of
//	points falls below 1, we simply store the points in a leaf
//	node's bucket.
//
//	This procedure selects a cutting dimension and cutting value,
//	partitions pa about these values, and returns the number of
//	points on the low side of the cut.
//
//	Note that this procedure is not only used for constructing full
//	trees, but is also used by the insertion routine to rebuild a
//	subtree.
//	
//----------------------------------------------------------------------

KCptr KCtree::buildKcTree(	// recursive construction of kc-tree
    KMdataArray		pa,		// point array
    KMdatIdxArray	pidx,		// point indices to store in subtree
    int			n,		// number of points
    int			dim,		// dimension of space
    KMorthRect		&bnd_box)	// bounding box for current node
{
    if (n <= 1) {			// n small, make a leaf node
	return new KCleaf(dim, bnd_box, n, pidx); 
    }
    else {				// n large, make a splitting node
	int cd;				// cutting dimension
	KMcoord cv;			// cutting value
	int n_lo;			// number on low side of cut
	KCptr lo, hi;			// low and high children

					// invoke splitting procedure
	sl_midpt_split(pa, pidx, bnd_box, n, dim, cd, cv, n_lo);

	KMcoord lv = bnd_box.lo[cd];	// save bounds for cutting dimension
	KMcoord hv = bnd_box.hi[cd];

	bnd_box.hi[cd] = cv;		// modify bounds for left subtree
	lo = buildKcTree(		// build left subtree
		pa, pidx, n_lo,		// ...from pidx[0..n_lo-1]
		dim, bnd_box);
	bnd_box.hi[cd] = hv;		// restore bounds

	bnd_box.lo[cd] = cv;		// modify bounds for right subtree
	hi = buildKcTree(		// build right subtree
		pa, pidx + n_lo, n-n_lo,// ...from pidx[n_lo..n-1]
		dim, bnd_box);
	bnd_box.lo[cd] = lv;		// restore bounds

					// create the splitting node
	KCsplit *ptr = new KCsplit(dim, bnd_box, cd, cv,
						lv, hv, lo, hi);
	return ptr;			// return pointer to this node
    }
} 

//----------------------------------------------------------------------
//  cellMidpt - return bounding box midpoint
//	The result is returned by modifying the argument.
//  getPoint - return point in a leaf cell
//	If the leaf cell has no point, then NULL is returned.
//	(This cannot be inlined in KCtree.h because of reference
//	to thePoints.)
//----------------------------------------------------------------------

void KCnode::cellMidpt(	// compute cell midpoint
    KMpoint	pt)			// the midpoint (returned)
{
    for (int d = 0; d < kcDim; d++) {		// compute box midpoint
	pt[d] = (bnd_box.lo[d] + bnd_box.hi[d])/2;
    }
}

KMpoint KCleaf::getPoint()		// get data point
{  return (n_data == 1 ? kcPoints[bkt[0]] : NULL);  }


//----------------------------------------------------------------------
//  kc-tree make sums (part of constructor)
//	Computes the sums of points for each node of the kc-tree,
//	and the sums of squares (the sums of dot products of each
//	point with itself).  These values are returned through the
//	arguments.
//
//	The sum of points is assumed to have been allocated as part
//	of the constructor, and hence already exists.
//----------------------------------------------------------------------

void KCsplit::makeSums(
    int			&n,			// number of points (returned)
    KMpoint		&theSum,		// sum (returned)
    double		&theSumSq)		// sum of squares (returned)
{
    assert(sum != NULL);			// should already be allocated
    int n_child = 0;				// n_data of child
    KMpoint s_child = NULL;			// sum of child
    double ssq_child = 0;			// sum of squares for child

    n_data = 0;					// initialize no. points
    						// process each child
    for (int i = KM_LO; i <= KM_HI; i++) {
    						// visit low child
	child[i]->makeSums(n_child, s_child, ssq_child);
	n_data += n_child;			// increment no. points
	for (int d = 0; d < kcDim; d++) {	// update sum and sumSq
	    sum[d] += s_child[d];
	}
	sumSq += ssq_child;
    }
    n = n_data;					// return results
    theSum = sum;
    theSumSq = sumSq;
}

//----------------------------------------------------------------------
void KCleaf::makeSums(
    int			&n,			// number of points (returned)
    KMpoint		&theSum,		// sum (returned)
    double		&theSumSq)		// sum of squares (returned)
{
    assert(sum != NULL);			// should already be allocated

    sumSq = 0;
    for (int i = 0; i < n_data; i++) {		// compute sum
	for (int d = 0; d < kcDim; d++) {
	    KMcoord theCoord = kcPoints[bkt[i]][d];
	    sum[d] += theCoord;
	    sumSq += theCoord * theCoord;
	}
    }

    n = n_data;					// return results
    theSum = sum;
    theSumSq = sumSq;
}

//----------------------------------------------------------------------
//  kc-tree destructors - deletes kc-tree 
//	(The other elements of the underlying kd-tree are deleted along
//	with the base class.)
//
//	Because of our "dirty trick" of recasting all child pointers
//	to type KCptr (from KMkd_ptr), we must be careful to not
//	allow the kd-tree constructors apply themselves recursively
//	to the child nodes.  We do this by setting the child pointers
//	to NULL after deleting them, which keeps the KMkd_split
//	destructor from activating itself.
//----------------------------------------------------------------------

KCtree::~KCtree()		// tree destructor
{
    if (root != NULL) delete root;
    if (pidx != NULL) delete [] pidx;
}

KCnode::~KCnode()		// node destructor
{
    if (sum != NULL) kmDeallocPt(sum);	// deallocate sum
}

//----------------------------------------------------------------------
//  Sample a center point
//	This implements an approach suggested by Matoushek for sampling
//	a center point.  A node of the kd-tree is selected at random.
//	If this is an interior node, a point is sampled uniformly from a
//	3x expansion of the cell about its center.  If the node is a
//	leaf, then a data point is sampled at random from the associated
//	bucket.
//	
//	Here is how sampling is done from an interior node.  Let m
//	denote the number of data points descended from this node.  Then
//	with probability 1/(2m-1), this cell is chosen.  Otherwise, let
//	mL and mR denote the number of points associated with the left
//	and right subtrees, respectively.  We sample from the left
//	subtree with probability (2mL-1)/(2m-1) and sample from the
//	right subtree with probability (2mR-1)/(2m-1).
//
//	The rationale is that, assuming there is exactly one point per
//	leaf node, a subtree with m points has exactly 2m-1 total nodes.
//	(This should be true for this implementation, but it depends in
//	general on the fact that there is exactly one point per leaf
//	node.)  Hence the root should be sampled with probability
//	1/(2m-1), and the subtrees should be sampled with the given
//	probabilities.
//----------------------------------------------------------------------

void KCtree::sampleCtr(KMpoint c)		// sample a point
{
    initBasicGlobals(dim, n_pts, pts);		// initialize globals
    // TODO: bb_save check is just for debugging.
    KMorthRect bb_save(dim, bnd_box);		// save bounding box
    root->sampleCtr(c, bnd_box);		// start at root
    for (int i = 0; i < dim; i++) {		// check that bnd_box unchanged
	assert(bb_save.lo[i] == bnd_box.lo[i] &&
	       bb_save.hi[i] == bnd_box.hi[i]);
    }
}

void KCsplit::sampleCtr(			// sample from splitting node
    KMpoint		c,			// the sampled point (returned)
    KMorthRect		&bnd_box)		// bounding box for current node
{
    int r = kmRanInt(n_nodes());		// random integer [0..n_nodes-1]
    if (r == 0) {				// sample from this node
	KMorthRect expBox(kcDim);
	bnd_box.expand(kcDim, 3, expBox);	// compute 3x expanded box
	expBox.sample(kcDim, c);		// sample c from box
    }
    else if (r <= child[KM_LO]->n_nodes()) {	// sample from left
	KMcoord save = bnd_box.hi[cut_dim];	// save old upper bound
	bnd_box.hi[cut_dim] = cut_val;		// modify for left subtree
	child[KM_LO]->sampleCtr(c, bnd_box);
	bnd_box.hi[cut_dim] = save;		// restore upper bound
    }
    else {					// sample from right subtree
	KMcoord save = bnd_box.lo[cut_dim];	// save old lower bound
	bnd_box.lo[cut_dim] = cut_val;		// modify for right subtree
	child[KM_HI]->sampleCtr(c,  bnd_box);
	bnd_box.lo[cut_dim] = save;		// restore lower bound
    }
}

void KCleaf::sampleCtr(				// sample from leaf node
    KMpoint		c,			// the sampled point (returned)
    KMorthRect		&bnd_box)		// bounding box for current node
{
    int ri = kmRanInt(n_data);			// generate random index
    kmCopyPt(kcDim, kcPoints[bkt[ri]], c);	// copy to destination
}

//----------------------------------------------------------------------
//  Printing the kc-tree 
//	These routines print a kc-tree in reverse inorder (high then
//	root then low).  (This is so that if you look at the output
//	from the right side it appear from left to right in standard
//	inorder.)  When outputting leaves we output only the point
//	indices rather than the point coordinates.  There is an option
//	to print the point coordinates separately.
//
//	The tree printing routine calls the printing routines on the
//	individual nodes of the tree, passing in the level or depth
//	in the tree.  The level in the tree is used to print indentation
//	for readability.
//----------------------------------------------------------------------

void KCsplit::print(		// print splitting node
    int		level)			// depth of node in tree
{
    					// print high child
    child[KM_HI]->print(level+1);

    *kmOut << "    ";			// print indentation
    for (int i = 0; i < level; i++)
	*kmOut << ".";

    kmOut->precision(4);
    *kmOut << "Split"			// print without address
        << " cd=" << cut_dim << " cv=" << setw(6) << cut_val
       	<< " nd=" << n_data
       	<< " sm=";  kmPrintPt(sum, kcDim, true);
    *kmOut << " ss=" << sumSq << "\n";
    					// print low child
    child[KM_LO]->print(level+1);
}

//----------------------------------------------------------------------
void KCleaf::print(			// print leaf node
    int		level)			// depth of node in tree
{
    *kmOut << "    ";
    for (int i = 0; i < level; i++)	// print indentation
	*kmOut << ".";

    // *kmOut << "Leaf <" << (void*) this << ">";
    *kmOut << "Leaf";			// print without address
    *kmOut << " n=" << n_data << " <";
    for (int j = 0; j < n_data; j++) {
	*kmOut << bkt[j];
	if (j < n_data-1) *kmOut << ",";
    }
    *kmOut << ">"
       	<< " sm=";  kmPrintPt(sum, kcDim, true);
    *kmOut << " ss=" << sumSq << "\n";
}

//----------------------------------------------------------------------
void KCtree::print(			// print entire tree
    bool	with_pts)			// print points as well?
{
    if (with_pts) {			// print point coordinates
	*kmOut << "    Points:\n";
	for (int i = 0; i < n_pts; i++) {
	    *kmOut << "\t" << i << ": ";
	    kmPrintPt(pts[i], kcDim, true);
            *kmOut << "\n";
	}
    }
    if (root == NULL)			// empty tree?
	*kmOut << "    Null tree.\n";
    else {
    	root->print(0);			// invoke printing at root
    }
}

//----------------------------------------------------------------------
// DistGlobals - globals used in computing distortions
// 	To prevent long argument lists in the computation of
// 	distortions, we store a number of common global variables here.
// 	These are initialized in KCtree::getNeighbors.
//
// 	Note: kcDim and kcPoints (from Basic Globals) are used as well.
//----------------------------------------------------------------------

int		kcKCtrs;		// number of centers
int*		kcWeights;		// weights of each point
KMpointArray	kcCenters;		// the center points
KMpointArray	kcSums;			// sums
double*		kcSumSqs;		// sum of squares
double*		kcDists;		// distortions
KMpoint		kcBoxMidpt;		// bounding-box midpoint

//----------------------------------------------------------------------
//  initDistGlobals - initialize distortion globals
//----------------------------------------------------------------------

static void initDistGlobals(		// initialize distortion globals
    KMfilterCenters& ctrs)			// the centers
{
    initBasicGlobals(ctrs.getDim(), ctrs.getNPts(), ctrs.getDataPts());
    kcKCtrs	= ctrs.getK();
    kcCenters	= ctrs.getCtrPts();		// get ptrs to KMcenter arrays
    kcWeights	= ctrs.getWeights(false);
    kcSums	= ctrs.getSums(false);
    kcSumSqs	= ctrs.getSumSqs(false);
    kcDists	= ctrs.getDists(false);
    kcBoxMidpt  = kmAllocPt(kcDim);

    for (int j = 0; j < kcKCtrs; j++) {		// initialize sums
	kcWeights[j] = 0;
	kcSumSqs[j] = 0;
	for (int d = 0; d < kcDim; d++) {
    	    kcSums[j][d] = 0;
	}
    }
}

static void deleteDistGlobals()		// delete distortion globals
{
    kmDeallocPt(kcBoxMidpt);
}

//----------------------------------------------------------------------
// getNeighbors - get neighbors for each candidate
//	This is the heart of the filter-based k-means algorithm.  It is
//	given an array of centers (ctrs) and an array of center sums
//	(sums), and an array of sums of squares (sumSqs).  All three
//	arrays consist of k points.  It computes the sum, sum of
//	squares, and weights of all the neighbors of each center, and
//	stores the results in these arrays.  From these quantities, the
//	final centroid and distortion (mean squared error) can be
//	computed.
//
//	This is done by determining the set of candidates for each node
//	in the kc-tree.  When the number of candidates for a node is
//	equal to 1 (it cannot be 0) then all of the points in the
//	subtree rooted at this node are assigned as neighbors to this
//	center.  This means that the centroid and weight for this cell
//	is added into the neighborhood centroid sum for this center.  If
//	this node is a leaf, then we compute (by brute-force) the
//	distance from each candidate to each data point, and assign the
//	data point to the closest center.
//
//	The key to pruning the set of candidates for each node is
//	handled by two functions.  The function nearCand() finds the
//	candidate that is nearest to the midpoint of the cell.  The
//	function pruneTest() determines whether another candidate is
//	close enough to the cell to be closer to some part of the cell
//	than the nearest candidate.
//----------------------------------------------------------------------

void KCtree::getNeighbors(		// compute neighbors for centers
    KMfilterCenters& ctrs)			// the centers
{
    initDistGlobals(ctrs);			// initialize globals
    int *candIdx = new int[kcKCtrs];		// allocate center indices
    for (int j = 0; j < kcKCtrs; j++) {		// initialize everything
    	candIdx[j] = j;				// initialize indices
    }
    root->getNeighbors(candIdx, kcKCtrs);	// get neighbors for tree
    delete [] candIdx;				// delete center indices
    deleteDistGlobals();			// delete globals
}

//----------------------------------------------------------------------
void KCsplit::getNeighbors(		// get neighbors for internal node
    KMctrIdxArray	cands,			// candidate centers
    int			kCands)			// number of centers
{
    if (kCands == 1) {				// only one cand left?
						// post points as neighbors
    	postNeigh(this, sum, sumSq, n_data, cands[0]);
    }
    else {
    						// get closest cand to box
	int cc = closestToBox(cands, kCands, bnd_box);
	KMctrIdx closeCand = cands[cc];		// closest candidate index
						// space for new candidates
	KMctrIdxArray newCands = new KMctrIdx[kCands];
	int newK = 0;				// number of new candidates
	for (int j = 0; j < kCands; j++) {
	    if (j == cc || !pruneTest(		// is candidate close enough?
	    			kcCenters[cands[j]],
	    			kcCenters[closeCand],
				bnd_box)) {
	    	newCands[newK++] = cands[j];	// yes, keep it
	    }
	}
						// apply to children
	child[KM_LO]->getNeighbors(newCands, newK);
	child[KM_HI]->getNeighbors(newCands, newK);
	delete [] newCands;			// delete new candidates
    }
}

//----------------------------------------------------------------------
void KCleaf::getNeighbors(		// get neighbors for leaf node
    KMctrIdxArray	cands,			// candidate centers
    int			kCands)			// number of centers
{
    if (kCands == 1) {				// only one cand left?
						// post points as neighbors
    	postNeigh(this, sum, sumSq, n_data, cands[0]);
    }
    else {					// find closest centers
	for (int i = 0; i < n_data; i++) {	// for each point in bucket
	    KMdist minDist = KM_DIST_INF;	// distance to nearest point
	    int minK = 0;			// index of this point
	    KMpoint thisPt = kcPoints[bkt[i]];	// this data point

	    for (int j = 0; j < kCands; j++) {	// compute closest candidate
		KMdist dist = kmDist(kcDim, kcCenters[cands[j]], thisPt);
        	if (dist < minDist) {		// best so far?
        	    minDist = dist;		// yes, save it
		    minK = j;			// ...and its index
		}
	    }
    	    postNeigh(this, kcPoints[bkt[i]], sumSq, 1, cands[minK]);
	}
    }
}

//----------------------------------------------------------------------
// getAssignments 
//	This determines the assignments of the closest center to each of
//	the data points.  It is basically a structural copy of the
//	procedure getNeighbors, but rather than incrementing the various
//	sums and sums of squares, it simply records the assignment of
//	each data point to its closest center.  Unlike the filtering
//	search, when only one candidate remains, it does not stop the
//	search, but continues to traverse all the leaves descended from
//	this node in order to perform the assignments.
//----------------------------------------------------------------------

void KCtree::getAssignments(		// compute assignments for points
    KMfilterCenters&    ctrs,			// the current centers
    KMctrIdxArray 	closeCtr,		// closest center per point
    double*	 	sqDist)			// sq'd distance to center
{
    initDistGlobals(ctrs);			// initialize globals

    int *candIdx = new int[kcKCtrs];		// allocate center indices
    for (int j = 0; j < kcKCtrs; j++) {		// initialize everything
    	candIdx[j] = j;				// initialize indices
    }
    						// search the tree
    root->getAssignments(candIdx, kcKCtrs, closeCtr, sqDist);
    delete [] candIdx;				// delete center indices
    deleteDistGlobals();			// delete globals
}

//----------------------------------------------------------------------
void KCsplit::getAssignments(		// get assignments for internal node
    KMctrIdxArray	cands,			// candidate centers
    int			kCands,			// number of centers
    KMctrIdxArray 	closeCtr,		// closest center per point
    double*	 	sqDist)			// sq'd distance to center
{
    if (kCands == 1) {				// only one cand left?
						// no more pruning needed
	child[KM_LO]->getAssignments(cands, kCands, closeCtr, sqDist);
	child[KM_HI]->getAssignments(cands, kCands, closeCtr, sqDist);
    }
    else {
    						// get closest cand to box
	int cc = closestToBox(cands, kCands, bnd_box);
	KMctrIdx closeCand = cands[cc];		// closest candidate index
						// space for new candidates
	KMctrIdxArray newCands = new KMctrIdx[kCands];
	int newK = 0;				// number of new candidates
	for (int j = 0; j < kCands; j++) {
	    if (j == cc || !pruneTest(		// is candidate close enough?
	    			kcCenters[cands[j]],
	    			kcCenters[closeCand],
				bnd_box)) {
	    	newCands[newK++] = cands[j];	// yes, keep it
	    }
	}
						// apply to children
	child[KM_LO]->getAssignments(newCands, newK, closeCtr, sqDist);
	child[KM_HI]->getAssignments(newCands, newK, closeCtr, sqDist);
	delete [] newCands;			// delete new candidates
    }
}

//----------------------------------------------------------------------
void KCleaf::getAssignments(		// get assignments for leaf node
    KMctrIdxArray	cands,			// candidate centers
    int			kCands,			// number of centers
    KMctrIdxArray 	closeCtr,		// closest center per point
    double*	 	sqDist)			// sq'd distance to center
{
    for (int i = 0; i < n_data; i++) {		// for each point in bucket
	KMdist minDist = KM_DIST_INF;		// distance to nearest point
	int minK = 0;				// index of this point
	KMpoint thisPt = kcPoints[bkt[i]];	// this data point

	for (int j = 0; j < kCands; j++) {	// compute closest candidate
	    KMdist dist = kmDist(kcDim, kcCenters[cands[j]], thisPt);
	    if (dist < minDist) {		// best so far?
		minDist = dist;			// yes, save it
		minK = j;			// ...and its index
	    }
	}
	if (closeCtr != NULL) closeCtr[bkt[i]] = cands[minK];
	if (sqDist != NULL) sqDist[bkt[i]] = minDist;
    }
}

//----------------------------------------------------------------------
//  Local utilities
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//  closestToBox - compute the closest point to the box
//	This procedure is given a list of candidates (cands), the number
//	of candidates (kCands), and a cell (bnd_box), and returns the
//	index (in cands) of the element of cands that is closest to the
//	midpoint of the cell.  The global variable kcBoxMidpt is used to
//	store the cell midpoint.
//----------------------------------------------------------------------

static int closestToBox(		// get closest point to box center
    KMctrIdxArray	cands,			// candidates for closest
    int			kCands,			// number of candidates
    KMorthRect		&bnd_box)		// bounding box of cell
{
    for (int d = 0; d < kcDim; d++) {		// compute midpoint
	kcBoxMidpt[d] = (bnd_box.lo[d] + bnd_box.hi[d])/2;
    }

    KMdist minDist = KM_DIST_INF;		// distance to nearest point
    int minK = 0;				// index of this point

    for (int j = 0; j < kCands; j++) {		// compute dist to each point
        KMdist dist = kmDist(kcDim, kcCenters[cands[j]], kcBoxMidpt);
        if (dist < minDist) {			// best so far?
            minDist = dist;			// yes, save it
	    minK = j;				// ...and its index
	}
    }
    return minK;				// return closest index
}

//----------------------------------------------------------------------
//  pruneTest - determine whether a point should be pruned
//	This procedure is given a cell of the kc-tree (bnd_box or B),
//	and candidate (cand or c) and the closest candidate to the
//	cell (closeCand or c').  It determines whether the entire
//	cell is closer to c' than it is to c.
//
//	The procedure works by considering the relationship between
//	two vectors.  Let (a.b) denote the dot product of vectors a
//	and b.  Observe that a point p is closer c than to c' if and
//	only if
//
//		(p-c).(p-c) < (p-c').(p-c')
//	
//	after simple manipulations this is true if and only if
//
//		(c-c').(c-c') < 2(p-c').(c-c').
//	
//	We want to know whether this relation is satisfied for any
//	point p in B.  If so then it is satisfied for the point p
//	in B that maximizes (p-c').(c-c').  Observe that p will be
//	a vertex of B.  To determine p, we consider the sign of each
//	coordinate of (c-c').  If the d-th coordinate is positive then
//	we set p[d] = B.hi[d], and otherwise we set it to B.lo[d].
//
//	NOTE: This procedure assumes that Euclidean distances are
//	used.
//----------------------------------------------------------------------

static bool pruneTest(
    KMcenter		cand,			// candidate to test
    KMcenter		closeCand,		// closest candidate
    KMorthRect		&bnd_box)		// bounding box
{
    double boxDot = 0;				// holds (p-c').(c-c')
    double ccDot = 0;				// holds (c-c').(c-c')
    for (int d = 0; d < kcDim; d++) {
    	double ccComp = cand[d] - closeCand[d];	// one component c-c'
	ccDot += ccComp * ccComp;		// increment dot product
	if (ccComp > 0) {			// candidate on high side
	   					// use high side of box
	   boxDot += (bnd_box.hi[d] - closeCand[d]) * ccComp;
	}
	else {					// candidate on low side
	   					// use low side of box
	   boxDot += (bnd_box.lo[d] - closeCand[d]) * ccComp;
	}
    }
    return (ccDot >= 2*boxDot);			// return final result
}

//----------------------------------------------------------------------
// postNeigh - registers neighbors for a given candidate
//	This procedure registers a set of points as neighbors
//	of a given center (cand).  The points are represented by
//	their sum and sum of squares.  A pointer to the
//	node doing the posting is passed along, but it is used
//	only if tracing.
//----------------------------------------------------------------------

static void postNeigh(
    KCptr		p,			// the node posting
    KMpoint		sum,			// the sum of coordinates
    double		sumSq,			// the sum of squares
    int			n_data,			// number of points
    KMctrIdx		ctrIdx)			// center index
{
    for (int d = 0; d < kcDim; d++) {			// increment sum
	kcSums[ctrIdx][d] += sum[d];
    }
    kcWeights[ctrIdx] += n_data;			// increment weight
    kcSumSqs[ctrIdx] += sumSq;				// incr sum of squares
}
