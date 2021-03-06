#include "FilterAndProcess.h"

namespace grip {

FilterAndProcess::FilterAndProcess() {
}
/**
* Runs an iteration of the pipeline and updates outputs.
*/
void FilterAndProcess::Process(cv::Mat& source0){
	//Step CV_cvtColor0:
	//input
	cv::Mat cvCvtcolorSrc = source0;
    int cvCvtcolorCode = cv::COLOR_BGR2HSV;
	cvCvtcolor(cvCvtcolorSrc, cvCvtcolorCode, this->cvCvtcolorOutput);
	//Step Blur0:
	//input
	cv::Mat blurInput = cvCvtcolorOutput;
	BlurType blurType = BlurType::BOX;
	double blurRadius = 2.7027027027027026;  // default Double
	blur(blurInput, blurType, blurRadius, this->blurOutput);
	//Step HSV_Threshold0:
	//input
	cv::Mat hsvThresholdInput = blurOutput;
	double hsvThresholdHue[] = {142.44604316546761, 179.0};
	double hsvThresholdSaturation[] = {204.09172661870502, 255.0};
	double hsvThresholdValue[] = {185.74640287769785, 255.0};
	hsvThreshold(hsvThresholdInput, hsvThresholdHue, hsvThresholdSaturation, hsvThresholdValue, this->hsvThresholdOutput);
	//Step Find_Contours0:
	//input
	cv::Mat findContoursInput = hsvThresholdOutput;
	bool findContoursExternalOnly = false;  // default Boolean
	findContours(findContoursInput, findContoursExternalOnly, this->findContoursOutput);
}

/**
 * This method is a generated getter for the output of a CV_cvtColor.
 * @return Mat output from CV_cvtColor.
 */
cv::Mat* FilterAndProcess::GetCvCvtcolorOutput(){
	return &(this->cvCvtcolorOutput);
}
/**
 * This method is a generated getter for the output of a Blur.
 * @return Mat output from Blur.
 */
cv::Mat* FilterAndProcess::GetBlurOutput(){
	return &(this->blurOutput);
}
/**
 * This method is a generated getter for the output of a HSV_Threshold.
 * @return Mat output from HSV_Threshold.
 */
cv::Mat* FilterAndProcess::GetHsvThresholdOutput(){
	return &(this->hsvThresholdOutput);
}
/**
 * This method is a generated getter for the output of a Find_Contours.
 * @return ContoursReport output from Find_Contours.
 */
std::vector<std::vector<cv::Point> >* FilterAndProcess::GetFindContoursOutput(){
	return &(this->findContoursOutput);
}
	/**
	 * Converts an image from one color space to another.
	 * @param src Image to convert.
	 * @param code conversion code.
	 * @param dst converted Image.
	 */
	void FilterAndProcess::cvCvtcolor(cv::Mat &src, int code, cv::Mat &dst) {
		cv::cvtColor(src, dst, code);
	}

	/**
	 * Softens an image using one of several filters.
	 *
	 * @param input The image on which to perform the blur.
	 * @param type The blurType to perform.
	 * @param doubleRadius The radius for the blur.
	 * @param output The image in which to store the output.
	 */
	void FilterAndProcess::blur(cv::Mat &input, BlurType &type, double doubleRadius, cv::Mat &output) {
		int radius = (int)(doubleRadius + 0.5);
		int kernelSize;
		switch(type) {
			case BOX:
				kernelSize = 2 * radius + 1;
				cv::blur(input,output,cv::Size(kernelSize, kernelSize));
				break;
			case GAUSSIAN:
				kernelSize = 6 * radius + 1;
				cv::GaussianBlur(input, output, cv::Size(kernelSize, kernelSize), radius);
				break;
			case MEDIAN:
				kernelSize = 2 * radius + 1;
				cv::medianBlur(input, output, kernelSize);
				break;
			case BILATERAL:
				cv::bilateralFilter(input, output, -1, radius, radius);
				break;
        }
	}
	/**
	 * Segment an image based on hue, saturation, and value ranges.
	 *
	 * @param input The image on which to perform the HSL threshold.
	 * @param hue The min and max hue.
	 * @param sat The min and max saturation.
	 * @param val The min and max value.
	 * @param output The image in which to store the output.
	 */
	void FilterAndProcess::hsvThreshold(cv::Mat &input, double hue[], double sat[], double val[], cv::Mat &out) {
		cv::cvtColor(input, out, cv::COLOR_BGR2HSV);
		cv::inRange(out,cv::Scalar(hue[0], sat[0], val[0]), cv::Scalar(hue[1], sat[1], val[1]), out);
	}

	/**
	 * Finds contours in an image.
	 *
	 * @param input The image to find contours in.
	 * @param externalOnly if only external contours are to be found.
	 * @param contours vector of contours to put contours in.
	 */
	void FilterAndProcess::findContours(cv::Mat &input, bool externalOnly, std::vector<std::vector<cv::Point> > &contours) {
		std::vector<cv::Vec4i> hierarchy;
		contours.clear();
		int mode = externalOnly ? cv::RETR_EXTERNAL : cv::RETR_LIST;
		int method = cv::CHAIN_APPROX_SIMPLE;
		cv::findContours(input, contours, hierarchy, mode, method);
	}



} // end grip namespace