#include <stdio.h>

#include <opencv2/opencv.hpp>

int main()
{
	cv::Mat src_8uc3_img = cv::imread("images/lena64.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)

	if (src_8uc3_img.empty()) {
        printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
    }

	//cv::imshow( "LENA", src_8uc3_img );

	cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
	cv::Mat gray_32fc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

	cv::cvtColor(src_8uc3_img, gray_8uc1_img, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	//The following converts the image to a usable type
	gray_8uc1_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0 / 255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0

	cv::Mat phase;
	gray_8uc1_img.convertTo(phase, CV_32FC1, 1.0 / 255.0);


	cv::Mat powerSpectrum;
	gray_8uc1_img.convertTo(powerSpectrum, CV_32FC1, 1.0 / 255.0);

    cv::Mat gray_64fc1_img, src_64fc1_img;
    src_8uc3_img.convertTo(src_64fc1_img, CV_64FC1, 1.0 / 255.0 );
    src_64fc1_img.convertTo(gray_64fc1_img, CV_64FC2, 1.0 / sqrt( src_8uc3_img.cols * src_8uc3_img.rows ));

    cv::Mat F = cv::Mat::zeros(src_8uc3_img.rows, src_8uc3_img.cols, CV_64FC2);

    cv::Mat inverse;
	gray_8uc1_img.convertTo(inverse, CV_32FC1, 1.0 / 255.0);


	int N = phase.cols, M = phase.rows;

	for (int k = 0; k < phase.cols; k++)
	{
		for (int l = 0; l < phase.rows; l++)
		{
			double FReal = 0.0;
			double FImag = 0.0;
			for (int m = 0; m < phase.cols; m++)
			{
				for (int n = 0; n < phase.rows; n++)
				{

					double x = 2 * CV_PI * (((m * k) / (double)M) + ((n * l) / (double)N));
					double point = gray_32fc1_img.at<float>(m, n);
					double expReal = cos(x);
					double expImag = sin(x);
					double phiReal = (1 / sqrt(M * N)) * expReal;
					double phiImag = (1 / sqrt(M * N)) * expImag;
					FReal += point * phiReal;
					FImag += point * phiImag;
				}

			}
			F.at<cv::Vec2d>(k,l) = cv::Vec2d(FReal, FImag);
			powerSpectrum.at<float>(l, k) = log(pow(FReal, 2) + pow(FImag, 2));
			phase.at<float>(l, k) = atan(FImag / FReal);
		}
	}


	for (int m = 0; m < F.cols; m++)
	{
		for (int n = 0; n < F.rows; n++)
		{
            double FREnd = 0.0;
			double FIEnd = 0.0;
			for (int k = 0; k < F.cols; k++)
			{
				for (int l = 0; l < F.rows; l++)
				{
                    double FR =F.at<cv::Vec2d>(k,l)[0];
                    double FI =F.at<cv::Vec2d>(k,l)[1];
                    double f = 2 * CV_PI * (((k* m) / (double)M) + ((l * n) / (double)N));
					double expReal = cos(f);
					double expImag = sin(f);
					double phiReal = (1 / sqrt(M * N)) * expReal;
					double phiImag = (1 / sqrt(M * N)) * expImag;
                    FREnd += FR * phiReal;
					FIEnd += FI * phiImag;
                }

			}

            inverse.at<float>(m, n) =FREnd+FIEnd;
		}
	}

	double lowerValue, higherValue = lowerValue = powerSpectrum.at<float>(0, 0);
	for (int i = 0; i < powerSpectrum.rows; i++)
	{
		for (int j = 0; j < powerSpectrum.cols; j++)
		{
			if (powerSpectrum.at<float>(i, j) < lowerValue)
				lowerValue = powerSpectrum.at<float>(i, j);
			if (powerSpectrum.at<float>(i, j) > higherValue)
				higherValue = powerSpectrum.at<float>(i, j);
		}
	}
	higherValue -= lowerValue;
	for (int i = 0; i < powerSpectrum.rows; i++)
	{
		for (int j = 0; j < powerSpectrum.cols; j++)
		{
			double x = powerSpectrum.at<float>(i, j);
			powerSpectrum.at<float>(i, j) = (powerSpectrum.at<float>(i, j) - lowerValue) / higherValue;
		}
	}

	for (int i = 0; i < powerSpectrum.rows; i++)
    cv::imshow("Lena", src_8uc3_img);
	cv::imshow("power", powerSpectrum);
	cv::imshow("phase", phase);
	cv::imshow("inverse", inverse);
	cv::waitKey();
	return 0;
}
