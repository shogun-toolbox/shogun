using System;
using System.Collections.Generic;

using org.shogun;
using org.jblas;

public class features_simple_modular
{
	static features_simple_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	static void Main(string[] argv)
	{
		modshogun.init_shogun_with_defaults();

		List<DoubleMatrix> result = new List<DoubleMatrix>(4);

		DoubleMatrix inputRealMatrix = Load.load_numbers("../data/fm_train_real.dat");
		RealFeatures realFeatures = new RealFeatures(inputRealMatrix);
		DoubleMatrix outputRealMatrix = realFeatures.get_feature_matrix();

		result.Add(inputRealMatrix);
		result.Add(outputRealMatrix);

		DoubleMatrix inputByteMatrix = Load.load_numbers("../data/fm_train_byte.dat");
		ByteFeatures byteFeatures = new ByteFeatures(inputByteMatrix);
		DoubleMatrix outputByteMatrix = byteFeatures.get_feature_matrix();

		result.Add(inputByteMatrix);
		result.Add(outputByteMatrix);

		DoubleMatrix inputLongMatrix = Load.load_numbers("../data/fm_train_byte.dat");
		LongFeatures byteFeatures = new LongFeatures(inputLongMatrix);
		DoubleMatrix outputLongMatrix = longFeatures.get_feature_matrix();

		result.Add(inputByteMatrix);
		result.Add(outputByteMatrix);

		Console.WriteLine(result);

		modshogun.exit_shogun();
	}
}
