#ifndef DSFMT_PARAMS132049_H
#define DSFMT_PARAMS132049_H

#include <shogun/lib/config.h>

/* #define DSFMT_N	1269 */
/* #define DSFMT_MAXDEGREE	132104 */
#define DSFMT_POS1	371
#define DSFMT_SL1	23
#define DSFMT_MSK1	UINT64_C(0x000fb9f4eff4bf77)
#define DSFMT_MSK2	UINT64_C(0x000fffffbfefff37)
#define DSFMT_MSK32_1	0x000fb9f4U
#define DSFMT_MSK32_2	0xeff4bf77U
#define DSFMT_MSK32_3	0x000fffffU
#define DSFMT_MSK32_4	0xbfefff37U
#define DSFMT_FIX1	UINT64_C(0x4ce24c0e4e234f3b)
#define DSFMT_FIX2	UINT64_C(0x62612409b5665c2d)
#define DSFMT_PCV1	UINT64_C(0x181232889145d000)
#define DSFMT_PCV2	UINT64_C(0x0000000000000001)
#define DSFMT_IDSTR	"dSFMT2-132049:371-23:fb9f4eff4bf77-fffffbfefff37"


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

#endif /* DSFMT_PARAMS132049_H */
