/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2006 Fabio De Bona
 * Written (W) 2006 Konrad Rieck
 * Written (W) 2006-2008 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdlib.h> 
#include <stdio.h> 
#include "lib/config.h"
#include "lib/memory.h"

#ifdef SUNOS
#define bool int
#define false 0
#define true 1
#endif

#ifndef LINUX
#define RANDOM_MAX 2147483647
#else
#define RANDOM_MAX RAND_MAX
#endif

/**@name Standard Types 
 * Definition of Platform independent Types
*/
//@{

/// Type CHAR
typedef char CHAR;
typedef CHAR* P_CHAR;

/// Type BYTE 
typedef unsigned char BYTE;
typedef BYTE* P_BYTE;

/// Type SHORT is 2 bytes in size
typedef short int SHORT;
typedef SHORT* P_SHORT;

/// Type WORD is 2 bytes in size
typedef unsigned short int WORD;
typedef WORD* P_WORD;

/// Type INT is 4 bytes in size
typedef int INT;
typedef INT* P_INT;

/// Type INT is 4 bytes in size
typedef unsigned int UINT;
typedef UINT* P_UINT;

/// Type LONG is 8 bytes in size
#ifndef SUNOS
#include <stdint.h>
typedef int64_t LONG;
#else
typedef long LONG;
#endif
typedef LONG* P_LONG;

/// Type ULONG is 8 bytes in size
#ifndef SUNOS
#include <stdint.h>
typedef uint64_t ULONG;
#else
typedef unsigned long ULONG;
#endif
typedef ULONG* P_ULONG;

/// Type SHORTREAL is 4 bytes in size
typedef float SHORTREAL;
typedef SHORTREAL* P_SHORTREAL;

/// Type DREAL is 8 bytes in size
typedef double DREAL;
typedef DREAL* P_DREAL;

/// Type LONGREAL is 16 bytes in size
typedef long double LONGREAL;
typedef LONGREAL* P_LONGREAL;

#ifdef USE_SHORTREAL_KERNELCACHE
	typedef SHORTREAL KERNELCACHE_ELEM;
#else
	typedef DREAL KERNELCACHE_ELEM;
#endif

typedef LONG KERNELCACHE_IDX;

/// The io libs output [DEBUG] etc in front of every message
/// 'higher' messages filter output depending on the loglevel, i.e. CRITICAL messages
/// will print all M_CRITICAL TO M_EMERGENCY messages to
enum EMessageType
{
	M_DEBUG,
	M_INFO,
	M_NOTICE,
	M_WARN,
	M_ERROR,
	M_CRITICAL,
	M_ALERT,
	M_EMERGENCY,
	M_MESSAGEONLY
};

enum EOptimizationType
{
	FASTBUTMEMHUNGRY,
	SLOWBUTMEMEFFICIENT
};

enum EWDKernType
{
	E_WD=0,
	E_EXTERNAL=1,

	E_BLOCK_CONST=2,
	E_BLOCK_LINEAR=3,
	E_BLOCK_SQPOLY=4,
	E_BLOCK_CUBICPOLY=5,
	E_BLOCK_EXP=6,
	E_BLOCK_LOG=7,
	E_BLOCK_EXTERNAL=8
};

enum EKernelType
{
	K_UNKNOWN = 0,
	K_LINEAR = 10,
	K_SPARSELINEAR = 11,
	K_POLY = 20,
	K_GAUSSIAN = 30,
	K_SPARSEGAUSSIAN = 31,
	K_GAUSSIANSHIFT = 32,
	K_HISTOGRAM = 40,
	K_SALZBERG = 41,
	K_LOCALITYIMPROVED = 50,
	K_SIMPLELOCALITYIMPROVED = 60,
	K_FIXEDDEGREE = 70,
	K_WEIGHTEDDEGREE =    80,
	K_WEIGHTEDDEGREEPOS = 81,
	K_WEIGHTEDCOMMWORDSTRING = 90,
	K_POLYMATCH = 100,
	K_ALIGNMENT = 110,
	K_COMMWORDSTRING = 120,
	K_COMMULONGSTRING = 121,
	K_COMBINED = 140,
	K_AUC = 150,
	K_CUSTOM = 160,
	K_SIGMOID = 170,
	K_CHI2 = 180,
	K_DIAG = 190,
	K_CONST = 200,
	K_MINDYGRAM = 210,
	K_DISTANCE = 220,
	K_LOCALALIGNMENT = 230,
	K_PYRAMIDCHI2 = 240,
	K_OLIGO = 250
};

enum EClassifierType
{
	CT_NONE = 0,
	CT_LIGHT = 10,
	CT_LIBSVM = 20,
	CT_LIBSVMONECLASS=30,
	CT_LIBSVMMULTICLASS=40,
	CT_MPD = 50,
	CT_GPBT = 60,
	CT_CPLEXSVM = 70,
	CT_PERCEPTRON = 80,
	CT_KERNELPERCEPTRON = 90,
	CT_LDA = 100,
	CT_LPM = 110,
	CT_LPBOOST = 120,
	CT_KNN = 130,
	CT_SVMLIN=140,
	CT_KRR = 150,
    CT_GNPPSVM = 160,
    CT_GMNPSVM = 170,
	CT_SUBGRADIENTSVM = 180,
	CT_SUBGRADIENTLPM = 190,
	CT_SVMPERF = 200,
	CT_LIBSVR = 210,
	CT_SVRLIGHT = 220,
	CT_LIBLINEAR = 230,
	CT_KMEANS = 240,
	CT_HIERARCHICAL = 250,
	CT_SVMOCAS = 260,
	CT_WDSVMOCAS = 270,
	CT_SVMSGD = 280,
};

enum EDistanceType
{
	D_UNKNOWN = 0,
	D_MINKOWSKI = 10,
	D_MANHATTAN = 20,
	D_CANBERRA = 30,
	D_CHEBYSHEW = 40,
	D_GEODESIC = 50,
	D_JENSEN = 60,
	D_MANHATTANWORD = 70,
	D_HAMMINGWORD = 80 ,
	D_CANBERRAWORD = 90,
	D_SPARSEEUCLIDIAN = 100,
	D_EUCLIDIAN = 110,
	D_CHISQUARE = 120,
	D_TANIMOTO = 130,
	D_COSINE = 140,
	D_BRAYCURTIS =150
};

enum ERegressionType
{
	RT_NONE = 0,
	RT_LIGHT = 10,
	RT_LIBSVM = 20
};

enum EPreProcType
{
	P_UNKNOWN=0,
	P_NORMONE=10,
	P_LOGPLUSONE=20,
	P_SORTWORDSTRING=30,
	P_SORTULONGSTRING=40,
	P_SORTWORD=50,
	P_PRUNEVARSUBMEAN=60
};

enum EKernelProperty
{
	KP_NONE = 0,
	KP_LINADD = 1, 	// Kernels that can be optimized via doing normal updates w + dw
	KP_KERNCOMBINATION = 2,	// Kernels that are infact a linear combination of subkernels K=\sum_i b_i*K_i
	KP_BATCHEVALUATION = 4  // Kernels that can on the fly generate normals in linadd and more quickly/memory efficient process batches instead of single examples
};

enum EFeatureType
{
	F_UNKNOWN = 0,
	F_CHAR = 10,
	F_BYTE = 20,
	F_SHORT = 30,
	F_WORD = 40,
	F_INT = 50,
	F_UINT = 60,
	F_LONG = 70,
	F_ULONG = 80,
	F_SHORTREAL = 90,
	F_DREAL = 100,
	F_LONGREAL = 110,
	F_ANY = 1000
};

enum EFeatureClass
{
	C_UNKNOWN = 0,
	C_SIMPLE = 10,
	C_SPARSE = 20,
	C_STRING = 30,
	C_COMBINED = 40,
	C_MINDYGRAM = 50,
	C_ANY = 1000
};

/// Alphabet of charfeatures/observations
enum E_ALPHABET
{
	/// DNA - letters A,C,G,T,*,N,n
	DNA=0,

	/// RAWDNA - letters 0,1,2,3
	RAWDNA=1,

	/// RNA - letters A,C,G,U,*,N,n
	RNA=2,

	/// PROTEIN - letters a-z
	PROTEIN=3,

	/// ALPHANUM - [0-9a-z]
	ALPHANUM=5,

	/// CUBE - [1-6]
	CUBE=6,

	/// RAW BYTE - [0-255]
	RAWBYTE=7,

	/// IUPAC_NUCLEIC_ACID
	IUPAC_NUCLEIC_ACID=8,

	/// IUPAC_AMINO_ACID
	IUPAC_AMINO_ACID=9,

	/// NONE - type has no alphabet
	NONE=10,

	/// unknown alphabet
	UNKNOWN=11,
};

//@}

//#define TMP_DIR "/tmp/"

#endif
