#ifndef DSFMT_PARAMS86243_H
#define DSFMT_PARAMS86243_H

/* #define DSFMT_N	829 */
/* #define DSFMT_MAXDEGREE	86344 */
#define DSFMT_POS1	231
#define DSFMT_SL1	13
#define DSFMT_MSK1	UINT64_C(0x000ffedff6ffffdf)
#define DSFMT_MSK2	UINT64_C(0x000ffff7fdffff7e)
#define DSFMT_MSK32_1	0x000ffedfU
#define DSFMT_MSK32_2	0xf6ffffdfU
#define DSFMT_MSK32_3	0x000ffff7U
#define DSFMT_MSK32_4	0xfdffff7eU
#define DSFMT_FIX1	UINT64_C(0x1d553e776b975e68)
#define DSFMT_FIX2	UINT64_C(0x648faadf1416bf91)
#define DSFMT_PCV1	UINT64_C(0x5f2cd03e2758a373)
#define DSFMT_PCV2	UINT64_C(0xc0b7eb8410000001)
#define DSFMT_IDSTR	"dSFMT2-86243:231-13:ffedff6ffffdf-ffff7fdffff7e"


/* PARAMETERS FOR ALTIVEC */
#if defined(__APPLE__)	/* For OSX */
    #define ALTI_SL1	(vector unsigned char)(5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5)
    #define ALTI_SL1_PERM \
	(vector unsigned char)(1,2,3,4,5,6,7,31,9,10,11,12,13,14,15,0)
    #define ALTI_SL1_MSK \
	(vector unsigned int)(0xffffffffU,0xffffe000U,0xffffffffU,0xffffe000U)
    #define ALTI_MSK	(vector unsigned int)(DSFMT_MSK32_1, \
			DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4)
#else	/* For OTHER OSs(Linux?) */
    #define ALTI_SL1	{5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5}
    #define ALTI_SL1_PERM \
	{1,2,3,4,5,6,7,31,9,10,11,12,13,14,15,0}
    #define ALTI_SL1_MSK \
	{0xffffffffU,0xffffe000U,0xffffffffU,0xffffe000U}
    #define ALTI_MSK \
	{DSFMT_MSK32_1, DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4}
#endif

#endif /* DSFMT_PARAMS86243_H */
