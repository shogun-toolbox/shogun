using System;

public class features_dense_real_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();

		double[,] matrix = new double[6, 3]{{1,2,3},{4,0,0},{0,0,0},{0,5,0},{0,0,6},{9,9,9}};
		RealFeatures a = new RealFeatures(matrix);

		a.set_feature_vector(new double[] {1, 4, 0, 0, 0, 9}, 0);

		double[,] a_out = a.get_feature_matrix();

		foreach(double item in a_out) {
			Console.Write("{0} ", item);
		}

	}
}
