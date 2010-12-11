//----------------------------------------------------------------------
//      File:           KM_ANN.h
//      Programmer:     David Mount
//      Last modified:  05/14/04
//      Description:    Stripped down ANN utilities
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
//	This file contains basic definitions for points, rectangles and
//	distance computations, which have been taken from ANN.  These
//	are heavily used by the kc-tree, which was adapted from ANN's
//	kd-tree routines.
//----------------------------------------------------------------------

#ifndef KM_ANN_H
#define KM_ANN_H

//----------------------------------------------------------------------
//  basic includes
//----------------------------------------------------------------------
#include <cstdlib>                      // standard libs and definitions
#include <fstream>                      // file I/O streams
#include <iomanip>                      // I/O manipulators (for setw)
#include <climits>                      // for INT_MAX
#include <cfloat>                       // for DBL_MAX
#include <ctime>                        // for system clock
#include <cassert>                      // assertion checking
#include <string>                       // STL strings
#include <climits>			// numeric limits

//----------------------------------------------------------------------
//  Basic Types:  KMcoord, KMdist, KMidx
//	KMcoord and KMdist are the types used for representing
//	point coordinates and distances.  They can be modified by the
//	user, with some care.  It is assumed that they are both numeric
//	types, and that KMdist is generally of an equal or higher type
//	from KMcoord.  A variable of type KMdist should be large
//	enough to store the sum of squared components of a variable
//	of type KMcoord for the number of dimensions needed in the
//	application.  For example, the following combinations are
//	legal:
//
//		KMcoord	KMdist
//		---------	-------------------------------
//		short		short, int, long, float, double
//		int		int, long, float, double
//		long		long, float, double
//		float		float, double
//		double		double
//
//	It is the user's responsibility to make sure that overflow does
//	not occur in distance calculation.
//
//	The code assumes that there is an infinite distance, KM_DIST_INF
//	(as large as any legal distance).  Possible values are given below:
//
//	    Examples:
//	    KMdist:		double, float, long, int, short
//	    KM_DIST_INF:	DBL_MAX, FLT_MAX, LONG_MAX, INT_MAX, SHRT_MAX
//
//	The routine that dumps a tree needs to know roughly how many
//	significant digits there are in a KMcoord, so it can output
//	points to full precision.  This is defined in KMcoordPrec.
//	If you have ANSI C++, you should be able to use the following
//	values from values.h to help in computing this:
//
//		KMcoord			KMcoordBits
//		--------------------	-------------------------------
//		short, int, long,...	BITS(short), BITS(int), ...
//		float			FSIGNIF
//		double			DSIGNIF
//
//	Then KMcoordPrec = KMcoordBits/(log_2 10) where log_2 10
//	is the base 2 logarithm of 10.
//
//	KMidx is a point index.  When the data structure is built,
//	the points are given as an array.  Nearest neighbor results are
//	returned as an index into this array.  To make it clearer when
//	this is happening, we define the integer type KMidx.
//		
//----------------------------------------------------------------------

typedef	double	KMcoord;		// coordinate data type
typedef	double	KMdist;			// distance data type
typedef int	KMidx;			// point index

					// largest possible distance
const KMdist	KM_DIST_INF	=  DBL_MAX;

#ifdef DSIGNIF				// number of sig. digits in KMcoord
    const int	 KMcoordPrec	= DBL_DIG;
#else
    const int	 KMcoordPrec	= 15;	// default precision
#endif

//----------------------------------------------------------------------
//  Norms and metrics:
//	KM supports any Minkowski norm for defining distance.  In
//	particular, for any p >= 1, the L_p Minkowski norm defines the
//	length of a d-vector (v0, v1, ..., v(d-1)) to be
//
//		(|v0|^p + |v1|^p + ... + |v(d-1)|^p)^(1/p),
//
//	(where ^ denotes exponentiation, and |.| denotes absolute
//	value).  The distance between two points is defined to be
//	the norm of the vector joining them.  Some common distance
//	metrics include
//
//		Euclidean metric	p = 2
//		Manhattan metric	p = 1
//		Max metric		p = infinity
//
//	In the case of the max metric, the norm is computed by
//	taking the maxima of the absolute values of the components.
//	KM is highly "coordinate-based" and does not support general
//	distances functions (e.g. those obeying just the triangle
//	inequality).  It also does not support distance functions
//	based on inner-products.
//
//	For the purpose of computing nearest neighbors, it is not
//	necessary to compute the final power (1/p).  Thus the only
//	component that is used by the program is |v(i)|^p.
//
//	KM parameterizes the distance computation through the following
//	macros.  (Macros are used rather than procedures for efficiency.)
//	Recall that the distance between two points is given by the length
//	of the vector joining them, and the length or norm of a vector v
//	is given by formula:
//
//		|v| = ROOT(POW(v0) # POW(v1) # ... # POW(v(d-1)))
//
//	where ROOT, POW are unary functions and # is an associative and
//	commutative binary operator satisfying:
//
//	    **	POW:	coord		--> dist
//	    **	#:	dist x dist	--> dist
//	    **	ROOT:	dist (>0)	--> double
//
//	For early termination in distance calculation (partial distance
//	calculation) we assume that POW and # together are monotonically
//	increasing on sequences of arguments, meaning that for all
//	v0..vk and y:
//
//	POW(v0) #...# POW(vk) <= (POW(v0) #...# POW(vk)) # POW(y).
//
//	Due to the use of incremental distance calculations in the code
//	for searching k-d trees, we assume that there is an incremental
//	update function DIFF(x,y) for #, such that if:
//
//		    s = x0 # ... # xi # ... # xk 
//
//	then if s' is s with xi replaced by y, that is, 
//	
//		    s' = x0 # ... # y # ... # xk
//
//	can be computed by:
//
//		    s' = s # DIFF(xi,y).
//
//	Thus, if # is + then DIFF(xi,y) is (yi-x).  For the L_infinity
//	norm we make use of the fact that in the program this function
//	is only invoked when y > xi, and hence DIFF(xi,y)=y.
//
//	Finally, for approximate nearest neighbor queries we assume
//	that POW and ROOT are related such that
//
//		    v*ROOT(x) = ROOT(POW(v)*x)
//
//	Here are the values for the various Minkowski norms:
//
//	L_p:	p even:				p odd:
//		-------------------------	------------------------
//		POW(v)		= v^p		POW(v)		= |v|^p
//		ROOT(x)		= x^(1/p)	ROOT(x)		= x^(1/p)
//		#		= +		#		= +
//		DIFF(x,y)	= y - x		DIFF(x,y)	= y - x	
//
//	L_inf:
//		POW(v)		= |v|
//		ROOT(x)		= x
//		#		= max
//		DIFF(x,y)  	= y
//
//	By default the Euclidean norm is assumed.  To change the norm,
//	uncomment the appropriate set of macros below.
//
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Use the following for the Euclidean norm
//----------------------------------------------------------------------
#define KM_POW(v)		((v)*(v))
#define KM_ROOT(x)		sqrt(x)
#define KM_SUM(x,y)		((x) + (y))
#define KM_DIFF(x,y)		((y) - (x))

//----------------------------------------------------------------------
//  Array types
//
//  KMpoint:
//	A point is represented as a (dimensionless) vector of
//	coordinates, that is, as a pointer to KMcoord.  It is the
//	user's responsibility to be sure that each such vector has
//	been allocated with enough components.  Because only
//	pointers are stored, the values should not be altered
//	through the lifetime of the nearest neighbor data structure.
//  KMpointArray is a dimensionless array of KMpoint.
//  KMdistArray is a dimensionless array of KMdist.
//  KMidxArray is a dimensionless array of KMidx.  This is used for
//	storing buckets of points in the search trees, and for returning
//	the results of k nearest neighbor queries.
//----------------------------------------------------------------------

typedef KMcoord		*KMpoint;		// a point
typedef KMpoint 	*KMpointArray;		// an array of points 
typedef KMdist  	*KMdistArray;		// an array of distances 
typedef KMidx		*KMidxArray;		// an array of point indices

//----------------------------------------------------------------------
//  Point utilities:
//
//	kmDist(dim, p, q)
//	  Returns the squared distances between p and q.
//
//	kmEqualPts(dim, p, q)
//	  Returns true if p and q are the same points.
//
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//  Point and Point-array Allocation/Deallocation:
//
//	Because points (somewhat like strings in C) are stored as
//	pointers.  Consequently, creating and destroying copies of
//	points may require storage allocation.  These procedures do
//	this.
//
//	p = kmAllocPt(dim, c=0)
//	  Allocates storage for a single point, and return a pointer
//	  to it.  The second argument is used to initialize all the
//	  points components.
//
//	kmDeallocPt(p)
//	  Deallocates a point allocated by kmAllocPt().
//
//	pa = kmAllocPts(n, dim)
//	  Allocates an array of n points in dimension d.  It performs
//	  no initialization.
//
// 	kmDeallocPts(pa)
//	  Deallocates points allocated by kmAllocPts().
//----------------------------------------------------------------------
   
KMdist kmDist(				// compute squared distance
    int			dim,			// dimension of space
    KMpoint		p,			// points
    KMpoint		q);

bool kmEqualPts(			// are two points equal?
    int			dim,			// dimension
    KMpoint		p1,			// the points
    KMpoint		p2);

KMpoint kmAllocPt(			// allocate point storage
    int			dim,			// dimension
    KMcoord		c = 0);			// coordinate value (all equal)

void kmDeallocPt(			// deallocate a point
    KMpoint		&p);

KMpointArray kmAllocPts(		// allocate point array
    int			n,			// number of points
    int			dim);			// dimension
   
void kmDeallocPts(			// deallocate a point array
    KMpointArray	&pa);			// the array

//----------------------------------------------------------------------
//  Point and other type copying:
//
//	kmCopyPt(dim, source, dest)
//	  Copies point source to point dest, without allocation.
//
//	dest = kmAllocCopyPt(dim, source)
//	  Allocates storage for and copies a point source to dest.
//
//	kmCopyPts(n, dim, source, dest)
//	  Copies point array source to point dest, without allocation.
//
//	dest = kmAllocCopyPts(n, dim, source)
//	  Allocates storage for and copies a point array source to dest.
//
//	kmCopy(n, source, dest)
//	  A generic copy routine for any time for which "=" is defined.
//
//	kmAllocCopy(n, source, dest)
//	  A generic allocate and copy routine for any time for which
//	  "=" is defined.
//----------------------------------------------------------------------

void kmCopyPt(				// copy point without allocation
    int			dim,			// dimension
    KMpoint		source,			// source point
    KMpoint		dest);			// destination point

KMpoint kmAllocCopyPt(			// allocate and copy point
    int			dim,			// dimension
    KMpoint		source);		// point to copy

void kmCopyPts(				// copy point array without allocation
    int			n,			// number of points
    int			dim,			// dimension
    const KMpointArray	source,			// source point
    KMpointArray	dest);			// destination point

KMpointArray kmAllocCopyPts(		// allocate and copy point array
    int			n,			// number of points
    int			dim,			// dimension
    const KMpointArray	source);		// source point

template <typename Object>
void kmCopy(				// copy anything without allocation
    int			n,			// number of object
    const Object*	source,			// source array
    Object*		dest)			// destination array
{
    for (int i = 0; i < n; i++) {		// copy contents
    	dest[i] = source[i];
    }
}

template <typename Object>
Object* kmAllocCopy(			// allocate and copy anything
    int			n,			// number of object
    const Object*	source)			// source array
{
    Object* dest = new Object[n];		// allocate array
    for (int i = 0; i < n; i++) {		// copy contents
    	dest[i] = source[i];
    }
    return dest;
}

//----------------------------------------------------------------------
//  Global constants and types
//----------------------------------------------------------------------
enum KMtreeType {KM_KD_TREE, KM_BD_TREE};	// tree types
enum		 {KM_LO=0, KM_HI=1};		// splitting indices
enum		 {KM_IN=0, KM_OUT=1};		// shrinking indices

const int KM_STRING_LEN = 100;		// default string length

//----------------------------------------------------------------------
//  Orthogonal (axis aligned) rectangle
//	Orthogonal rectangles are represented by two points, one
//	for the lower left corner (min coordinates) and the other
//	for the upper right corner (max coordinates).
//
//	The constructor initializes from either a pair of coordinates,
//	pair of points, or another rectangle.  Note that all constructors
//	allocate new point storage.  The destructor deallocates this
//	storage.
//
//	BEWARE: Orthogonal rectangles should be passed ONLY BY REFERENCE.
//	(C++'s default copy constructor will not allocate new point
//	storage, then on return the destructor free's storage, and then
//	you get into big trouble in the calling procedure.)
//----------------------------------------------------------------------

class KMorthRect {
public:
    KMpoint	lo;			// rectangle lower bounds
    KMpoint	hi;			// rectangle upper bounds
//
    KMorthRect(			// basic constructor
	int dd,				// dimension of space
	KMcoord l=0,			// default is empty
	KMcoord h=0)
    {  lo = kmAllocPt(dd, l);  hi = kmAllocPt(dd, h);  }

    KMorthRect(			// (almost a) copy constructor
	int dd,				// dimension
	const KMorthRect &r)		// rectangle to copy
    {  lo = kmAllocCopyPt(dd, r.lo);  hi = kmAllocCopyPt(dd, r.hi); }

    KMorthRect(			// construct from points
	int dd,				// dimension
	KMpoint l,			// low point
	KMpoint h)			// hight point
    {  lo = kmAllocCopyPt(dd, l);  hi = kmAllocCopyPt(dd, h);  }

    ~KMorthRect()			// destructor
    {  kmDeallocPt(lo);  kmDeallocPt(hi); }

    bool inside(int dim, KMpoint p);	// is point p inside rectangle?
    					// expand by factor x and store in r
    void expand(int dim, double x, KMorthRect r);
    void sample(int dim, KMpoint p);	// sample point p uniformly
};

void kmAssignRect(		// assign one rect to another
    int			dim,		// dimension (both must be same)
    KMorthRect		&dest,		// destination (modified)
    const KMorthRect	&source);	// source

#endif
