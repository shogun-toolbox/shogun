#ifndef __COMMON_H__
#define __COMMON_H__

#ifdef SUNOS_CC
#define bool int
#define false 0
#define true 1
#endif

/**@name Standard Types 
 * Definition of Platform independent Types
*/
//@{
/// Type WORD is 2 bytes in size
typedef unsigned short int WORD ;

/// Type BYTE 
typedef unsigned char BYTE ;

/// Type REAL (can be float/double/long double...)
typedef double REAL ;
//@}
#endif
