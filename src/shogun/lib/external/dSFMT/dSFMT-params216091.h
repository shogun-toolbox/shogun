#ifndef DSFMT_PARAMS216091_H
#define DSFMT_PARAMS216091_H

#include <shogun/lib/config.h>

/* #define DSFMT_N	2077 */
/* #define DSFMT_MAXDEGREE	216136 */
#define DSFMT_POS1	1890
#define DSFMT_SL1	23
#define DSFMT_MSK1	UINT64_C(0x000bf7df7fefcfff)
#define DSFMT_MSK2	UINT64_C(0x000e7ffffef737ff)
#define DSFMT_MSK32_1	0x000bf7dfU
#define DSFMT_MSK32_2	0x7fefcfffU
#define DSFMT_MSK32_3	0x000e7fffU
#define DSFMT_MSK32_4	0xfef737ffU
#define DSFMT_FIX1	UINT64_C(0xd7f95a04764c27d7)
#define DSFMT_FIX2	UINT64_C(0x6a483861810bebc2)
#define DSFMT_PCV1	UINT64_C(0x3af0a8f3d5600000)
#define DSFMT_PCV2	UINT64_C(0x0000000000000001)
#define DSFMT_IDSTR	"dSFMT2-216091:1890-23:bf7df7fefcfff-e7ffffef737ff"


/* PARAMETERS FOR ALTIVEC */
#if defined(__APPLE__)	/* For OSX */
    #define ALTI_SL1	(vector unsigned char)(7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7)
    #define ALTI_SL1_PERM \
	(vector unsigned char)(2,3,4,5,6,7,30,30,10,11,12,13,14,15,0,1)
    #define ALTI_SL1_MSK \
	(vector unsigned int)(0xffffffffU,0xff800000U,0xffffffffU,0xff800000U)
    #define ALTI_MSK	(vector unsigned int)(DSFMT_MSK32_1, \
			DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4)
#else	/* For OTHER OSs(Linux?) */
    #define ALTI_SL1	{7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7}
    #define ALTI_SL1_PERM \
	{2,3,4,5,6,7,30,30,10,11,12,13,14,15,0,1}
    #define ALTI_SL1_MSK \
	{0xffffffffU,0xff800000U,0xffffffffU,0xff800000U}
    #define ALTI_MSK \
	{DSFMT_MSK32_1, DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4}
#endif

#endif /* DSFMT_PARAMS216091_H */
