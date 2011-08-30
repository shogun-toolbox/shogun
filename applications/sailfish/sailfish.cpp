#include <shogun/base/init.h>
#include <shogun/features/Alphabet.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/DynInt.h>
#include <shogun/lib/IndirectObject.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/BitString.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/MemoryMappedFile.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <ctype.h>
#include <unistd.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

void print_warning(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

void print_error(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

void compute_overlap(uint192_t* dictionary, int32_t* locations, int32_t dic_len,
		uint192_t* avec, int32_t alen, CStringFeatures<uint192_t>* reads,
		bool output_all)
{
	int32_t last_j=0;
	int32_t old_idx = 0;

	SG_SPRINT("\n\n");
	int32_t order=reads->get_order();
	uint8_t* str=new uint8_t[order];

	if (avec && alen>0)
	{
		for (int32_t j=1; j<alen; j++)
		{
			if (avec[j-1] == (uint192_t) 0)
				continue;

			if (avec[j]==avec[j-1])
				continue;

			int32_t idx = CMath::binary_search_max_lower_equal(&(dictionary[old_idx]), dic_len-old_idx, avec[j-1]);

			if (idx!=-1)
			{
				int32_t subidx=idx-1;
				while (dictionary[subidx+old_idx] == avec[j-1] && subidx>=0)
				{
					reads->unembed_word(dictionary[subidx+old_idx], str, order);
					SG_SPRINT("%.*s: %8d\n", order, str, locations[subidx+old_idx] + 1);
					//result += dictionary_weights[subidx+old_idx]*(j-last_j);
					subidx--;
				}

				while (dictionary[idx+old_idx] == avec[j-1] && idx < dic_len)
				{
					reads->unembed_word(dictionary[idx+old_idx], str, order);
					SG_SPRINT("%.*s: %8d\n", order, str, locations[idx+old_idx] + 1);
					//result += dictionary_weights[idx+old_idx]*(j-last_j);
					idx++;
				}

				old_idx+=idx;
			}

			last_j = j;
		}

		int32_t idx = CMath::binary_search(&(dictionary[old_idx]), dic_len-old_idx, avec[alen-1]);
		if (idx!=-1)
		{
			int32_t subidx=idx-1;
			while (dictionary[subidx+old_idx] == avec[alen-1] && subidx>=0)
			{
				reads->unembed_word(dictionary[subidx+old_idx], str, order);
				SG_SPRINT("%.*s: %8d\n", order, str, locations[subidx+old_idx] + 1);
				//result += dictionary_weights[subidx+old_idx]*(alen-last_j);
				subidx--;
			}

			while (dictionary[idx+old_idx] == avec[alen-1] && idx < dic_len)
			{
				reads->unembed_word(dictionary[idx+old_idx], str, order);
				SG_SPRINT("%.*s: %8d\n", order, str, locations[idx+old_idx] + 1);
				//result += dictionary_weights[idx+old_idx]*(alen-last_j);
				idx++;
			}
		}
	}
	delete[] str;
}

void print_help()
{
	printf("\nUsage: sailfish -i genome.fasta -q reads.fastq\n\n");
	printf("Example:\n./sailfish -i /fml/ag-raetsch/share/databases/genomes/H_sapiens/hg18.old/sequences/Homo_sapiens.NCBI36.43.dna.chromosome.Y.fa \\\n"
			         "           -q /fml/ag-raetsch/share/databases/short_reads/SRP000401/SRX001872/uwgs-rw_L2_FC12227_3.fastq\n\n");
}

int main(int argc, char** argv)
{
	init_shogun(&print_message, &print_warning,
			&print_error);

	/* cd /fml/ag-raetsch/home/raetsch/Downloads/genomemapper && genomemapper \
	   -i /fml/ag-raetsch/share/databases/genomes/H_sapiens/hg18.old/sequences/Homo_sapiens.NCBI36.43.dna.chromosome.Y.fa \
	   -x ~/tmp/Homo_sapiens.NCBI36.43.dna.chromosome.Y.fa.index \
	   -t ~/tmp/Homo_sapiens.NCBI36.43.dna.chromosome.Y.fa.metaindex \
	   -q /fml/ag-raetsch/share/databases/short_reads/SRP000401/SRX001872/uwgs-rw_L2_FC12227_3.fastq  */

	opterr = 0;
	char* fasta_fname=NULL;
	char* fastq_fname=NULL;
	char c;

	while ((c = getopt (argc, argv, "hi:q:")) != -1)
	{
		switch (c)
		{
			case 'h':
				print_help();
				return 0;
			case 'i':
				fasta_fname = optarg;
				break;
			case 'q':
				fastq_fname = optarg;
				break;
			case '?':
				if (optopt == 'i' || optopt == 'q')
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				else if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr,
							"Unknown option character `\\x%x'.\n",
							optopt);
				return 1;
			default:
				abort();
		}
	}

	if (!fasta_fname)
	{
		fprintf (stderr, "genomic fasta fname required\n");
		print_help();
		return 1;
	}

	if (!fastq_fname)
	{
		fprintf (stderr, "fastq read fname required\n");
		print_help();
		return 1;
	}

	CBitString genome(DNA, 20);
	genome.load_fasta_file(fasta_fname, true);
	int32_t l=genome.get_length();

	CIndirectObject<uint64_t,CBitString*>::set_array(&genome);
	CIndirectObject<uint64_t,CBitString*>* x = new CIndirectObject<uint64_t,CBitString*>[l];
	CIndirectObject<uint64_t,CBitString*>::init_slice(x, l);
	CMath::qsort(x, l);

	/*
	for (int i=0; i<l; i++)
	{
		printf("%0lld\n", uint64_t(x[i]));
	}
	

	for (int i=0; i<l; i++)
	{
		uint64_t x=genome[i];

		for (int j=6; j>=0; j--)
		{
			switch ((x>>(2*j)) & 3)
			{
				case 0:
					printf("A");
					break;
				case 1:
					printf("C");
					break;
				case 2:
					printf("G");
					break;
				case 3:
					printf("T");
					break;
			};
		}
		printf("\n");
	}
	printf("\n");
	*/

	/*for (int i=0; i<100; i++)
	{
		printf("%0llx\n", genome[i]);

	}
	printf("\n");
	*/

	//CTime t;

	//SG_SPRINT("reading fastq file '%s'...", fastq_fname);
	//t.start();
	//CStringFeatures<uint192_t>* reads = new CStringFeatures<uint192_t>(DNA);
	//reads->load_fastq_file(fastq_fname, true, true);
	//t.cur_time_diff(true);

	//SG_SPRINT("sorting ... ");
	//t.start();
	//int32_t s2_len;
	//uint192_t* s2=reads->get_feature_vector(0, s2_len);
	//CMath::qsort(s2, s2_len);
	//t.cur_time_diff(true);

	//SG_SPRINT("reading fasta file '%s' ... ", fasta_fname);
	//t.start();
	//CStringFeatures<uint192_t>* genome = new CStringFeatures<uint192_t>(DNA);
	//genome->load_fasta_file(fasta_fname, true);
	//t.cur_time_diff(true);

	//int32_t order=reads->get_order();
	//SG_SPRINT("computing embedding of order %d... ", order);
	//t.start();
	//genome->embed_features(order);
	//t.cur_time_diff(true);

	//int32_t s1_len;
	//uint192_t* s1=genome->get_feature_vector(0, s1_len);
	//int32_t* s1_pos= new int32_t[s1_len];
	//CMath::range_fill_vector(s1_pos, s1_len);
	//SG_SPRINT("sorting ... ");
	//t.start();
	//CMath::qsort_index(s1, s1_pos, s1_len);
	//t.cur_time_diff(true);
	//
	//
	//compute_overlap(s1, s1_pos, s1_len, s2, s2_len, reads, true);
	//
	//SG_SPRINT("\n");
	//for (int i=0; i<10; i++)
	//{
	//	SG_SPRINT("i=%.5d ", i);
	//	s1[i].print_bits();
	//	SG_SPRINT("\n");
	//}
	//
	//SG_SPRINT("\n\n");
	//uint8_t* str=new uint8_t[order];
	//for (int i=0; i<10; i++)
	//{
	//	SG_SPRINT("i=%.5d ", i);
	//	genome.unembed_word(s1[i], str, order);
	//	SG_SPRINT("%.*s\n", order, (char*) str);
	//}
	//
	//
	//for (int i=s1_len-50; i<s1_len; i++)
	//{
	//	SG_SPRINT("i=%.5d ", i);
	//	s1[i].print_bits();
	//	SG_SPRINT("\n");
	//}

	//int32_t len;
	//uint192_t* s=reads.get_feature_vector(0, len);

	//for (int i=0; i<200; i++)
	//{
	//	s[i].print_bits();
	//	SG_SPRINT("\n");
	//}

	//SG_SPRINT("\n\n");
	//s[0]=(uint64_t[3]) {0xAAAAAAA, 0xAAAAAAAA, 0xAAAAAAA};

	//for (int i=0; i<192; i++)
	//{
	//	//(s[0]<<i).print_bits();
	//	s[0].print_bits();
	//	SG_SPRINT("\n");
	//	s[0]<<=1;
	//}

	//SG_SPRINT("\n\n");
	//s[0]=(uint64_t[3]) {0xAAAAAAA, 0xAAAAAAAA, 0xAAAAAAA};

	//for (int i=0; i<192; i++)
	//{
	//	//(s[0]>>i).print_bits();
	//	s[0].print_bits();
	//	SG_SPRINT("\n");
	//	s[0]>>=1;
	//}

	//SG_UNREF(reads);
	//SG_UNREF(genome);

	exit_shogun();
	return 0;
}

