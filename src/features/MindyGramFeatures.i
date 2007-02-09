#ifdef HAVE_MINDY
%{
 #include "features/MindyGramFeatures.h" 
%}

%rename(MindyGramFeatures) CMindyGramFeatures;

%include "features/MindyGramFeatures.h" 

%template(import_from_char) CMindyGramFeatures::import_features<CHAR>;
%template(import_from_byte) CMindyGramFeatures::import_features<BYTE>;
#endif
