using System;

public class MatrixTest
{
	public static void Main(string[] args) {
		modshogun.init_shogun_with_defaults();

		RealFeatures x = new RealFeatures();
		double[,] y = new double[2,3] {{1, 2, 3}, {4, 5, 6}};

		x.set_feature_matrix(y);
		double [,] r = x.get_feature_matrix();
		foreach (int item in r) {
					Console.WriteLine(item);
		}

	}
}
