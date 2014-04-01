#ifndef DSFMT_PARAMS11213_H
#define DSFMT_PARAMS11213_H

#include <shogun/lib/config.h>

/* #define DSFMT_N	107 */
/* #define DSFMT_MAXDEGREE	11256 */
#define DSFMT_POS1	37
#define DSFMT_SL1	19
#define DSFMT_MSK1	UINT64_C(0x000ffffffdf7fffd)
#define DSFMT_MSK2	UINT64_C(0x000dfffffff6bfff)
#define DSFMT_MSK32_1	0x000fffffU
#define DSFMT_MSK32_2	0xfdf7fffdU
#define DSFMT_MSK32_3	0x000dffffU
#define DSFMT_MSK32_4	0xfff6bfffU
#define DSFMT_FIX1	UINT64_C(0xd0ef7b7c75b06793)
#define DSFMT_FIX2	UINT64_C(0x9c50ff4caae0a641)
#define DSFMT_PCV1	UINT64_C(0x8234c51207c80000)
#define DSFMT_PCV2	UINT64_C(0x0000000000000001)
#define DSFMT_IDSTR	"dSFMT2-11213:37-19:ffffffdf7fffd-dfffffff6bfff"


/* PARAMETERS FOR ALTIVEC */
#if defined(__APPLE__)	/* For OSX */
    #define ALTI_SL1	(vector unsigned char)(3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
    #define ALTI_SL1_PERM \
	(vector unsigned char)(2,3,4,5,6,7,30,30,10,11,12,13,14,15,0,1)
    #define ALTI_SL1_MSK \
	(vector unsigned int)(0xffffffffU,0xfff80000U,0xffffffffU,0xfff80000U)
    #define ALTI_MSK	(vector unsigned int)(DSFMT_MSK32_1, \
			DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4)
#else	/* For OTHER OSs(Linux?) */
    #define ALTI_SL1	{3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}
    #define ALTI_SL1_PERM \
	{2,3,4,5,6,7,30,30,10,11,12,13,14,15,0,1}
    #define ALTI_SL1_MSK \
	{0xffffffffU,0xfff80000U,0xffffffffU,0xfff80000U}
    #define ALTI_MSK \
	{DSFMT_MSK32_1, DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4}
#endif

#endif /* DSFMT_PARAMS11213_H */
