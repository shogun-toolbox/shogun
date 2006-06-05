%module Classifier%{
 #include "classifier/Classifier.h" 
%}

class CClassifier
{
	public:
		CClassifier();
		virtual ~CClassifier();

		virtual bool train()=0;
		virtual CLabels* classify(CLabels* output=NULL);

		virtual DREAL classify_example(INT num)=0;

		virtual bool load(FILE* srcfile)=0;
		virtual bool save(FILE* dstfile)=0;

		virtual inline void set_labels(CLabels* lab) { labels=lab; }
		virtual inline CLabels* get_labels() { return labels; }

		virtual EClassifierType get_classifier_type()=0;

	protected:
		CLabels* labels;
};
