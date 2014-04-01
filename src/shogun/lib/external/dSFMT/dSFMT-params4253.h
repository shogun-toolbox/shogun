#ifndef DSFMT_PARAMS4253_H
#define DSFMT_PARAMS4253_H

#include <shogun/lib/config.h>

/* #define DSFMT_N	40 */
/* #define DSFMT_MAXDEGREE	4288 */
#define DSFMT_POS1	19
#define DSFMT_SL1	19
#define DSFMT_MSK1	UINT64_C(0x0007b7fffef5feff)
#define DSFMT_MSK2	UINT64_C(0x000ffdffeffefbfc)
#define DSFMT_MSK32_1	0x0007b7ffU
#define DSFMT_MSK32_2	0xfef5feffU
#define DSFMT_MSK32_3	0x000ffdffU
#define DSFMT_MSK32_4	0xeffefbfcU
#define DSFMT_FIX1	UINT64_C(0x80901b5fd7a11c65)
#define DSFMT_FIX2	UINT64_C(0x5a63ff0e7cb0ba74)
#define DSFMT_PCV1	UINT64_C(0x1ad277be12000000)
#define DSFMT_PCV2	UINT64_C(0x0000000000000001)
#define DSFMT_IDSTR	"dSFMT2-4253:19-19:7b7fffef5feff-ffdffeffefbfc"


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

#endif /* DSFMT_PARAMS4253_H */
