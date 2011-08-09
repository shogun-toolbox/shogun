using System;

public class VectorTest
{
	public static void Main(string[] args) {
		modshogun.init_shogun_with_defaults();

		double[] y = new double[4] {1, 2, 3, 4,};
		Labels x = new Labels(y);
		double[] r = x.get_labels();
		for (int i = 0; i < 4; i++) {
					Console.WriteLine(r[i]);
		}

		modshogun.exit_shogun();
	}
}
