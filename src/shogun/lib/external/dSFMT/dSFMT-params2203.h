#ifndef DSFMT_PARAMS2203_H
#define DSFMT_PARAMS2203_H

/* #define DSFMT_N	20 */
/* #define DSFMT_MAXDEGREE	2208 */
#define DSFMT_POS1	7
#define DSFMT_SL1	19
#define DSFMT_MSK1	UINT64_C(0x000fdffff5edbfff)
#define DSFMT_MSK2	UINT64_C(0x000f77fffffffbfe)
#define DSFMT_MSK32_1	0x000fdfffU
#define DSFMT_MSK32_2	0xf5edbfffU
#define DSFMT_MSK32_3	0x000f77ffU
#define DSFMT_MSK32_4	0xfffffbfeU
#define DSFMT_FIX1	UINT64_C(0xb14e907a39338485)
#define DSFMT_FIX2	UINT64_C(0xf98f0735c637ef90)
#define DSFMT_PCV1	UINT64_C(0x8000000000000000)
#define DSFMT_PCV2	UINT64_C(0x0000000000000001)
#define DSFMT_IDSTR	"dSFMT2-2203:7-19:fdffff5edbfff-f77fffffffbfe"


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

#endif /* DSFMT_PARAMS2203_H */
