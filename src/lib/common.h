#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdlib.h> 
#include <stdio.h> 
#include "lib/config.h"

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
typedef long LONG;
typedef LONG* P_LONG;

/// Type SHORTREAL is 4 bytes in size
typedef float SHORTREAL;
typedef SHORTREAL* P_SHORTREAL;

/// Type REAL is 8 bytes in size
typedef double REAL;
typedef REAL* P_REAL;

/// Type LONGREAL is 16 bytes in size
typedef long double LONGREAL;
typedef LONGREAL* P_LONGREAL;

#ifdef USE_SHORTREAL_KERNELCACHE
	typedef SHORTREAL KERNELCACHE_ELEM;
#else
	typedef REAL KERNELCACHE_ELEM;
#endif

typedef KERNELCACHE_ELEM P_KERNELCACHE_ELEM;

typedef LONG KERNELCACHE_IDX;

/// The io libs output [DEBUG] etc in front of every CIO::message
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
	M_MESSAGEONLY,
	M_PROGRESS
};

enum EKernelType
{
	K_UNKNOWN = 0,
	K_OPTIMIZABLE = 4096,
	K_LINEAR = 10 | K_OPTIMIZABLE,
	K_POLY	= 20,
	K_GAUSSIAN = 30,
	K_HISTOGRAM = 40,
	K_SALZBERG = 41,
	K_LOCALITYIMPROVED = 50,
	K_SIMPLELOCALITYIMPROVED = 60,
	K_FIXEDDEGREE = 70,
	K_WEIGHTEDDEGREE =    80 | K_OPTIMIZABLE,
	K_WEIGHTEDDEGREEPOS = 81 | K_OPTIMIZABLE,
	K_WEIGHTEDDEGREEPOLYA =    82,
	K_WD = 83,
	K_COMMWORD = 90 | K_OPTIMIZABLE ,
	K_POLYMATCH = 100,
	K_ALIGNMENT = 110,
	K_COMMWORDSTRING = 120 | K_OPTIMIZABLE,
	K_SPARSENORMSQUARED = 130,
	K_COMBINED = 140 | K_OPTIMIZABLE
};

enum EFeatureType
{
	F_UNKNOWN = 0,
	F_REAL = 10,
	F_SHORT = 20,
	F_CHAR = 30,
	F_INT = 40,
	F_BYTE = 50,
	F_WORD = 60
};

enum EFeatureClass
{
	C_UNKNOWN = 0,
	C_SIMPLE = 10,
	C_SPARSE = 20,
	C_STRING = 30,
	C_COMBINED = 40
};

/// Alphabet of charfeatures/observations
enum E_ALPHABET
{
	/// DNA - letters A,C,G,T,*,N,n
	DNA=0,

	/// PROTEIN - letters a-z
	PROTEIN=1,

	/// ALPHANUM - [0-9a-z]
	ALPHANUM=2,

	/// CUBE - [1-6]
	CUBE=3,

	/// NONE - type has no alphabet
	NONE=4
};

//@}

#define TMP_DIR "/tmp/"
//#define TMP_DIR "/short/x46/tmp/"

#endif
