#include "functions.h"

/*Function to calculate the integral histogram*/
vector<Mat> calculateIntegralHist(Mat srcImg, string histName, int spatial_scale){
	Mat xSobel, xSobel2, ySobel, orient, magnitude;
	vector<Mat> flowMap(2);

	if (strcmp(histName.c_str(), "HOG") == 0){
		/*Convert the input image to grayscale*/
		Mat grayImg;
		cvtColor(srcImg, grayImg, CV_BGR2GRAY);
		//equalizeHist(grayImg, grayImg); // is it really needed?
		//system("pause");
		/* Calculate the derivates of the grayscale image in the x and y directions using a sobel
		operator and obtain 2 gradient images for the x and y directions*/
		// k-size can be either 3 or 1. Dalal and Triggs observed that k-size=1 provides better results 
		//cv_32f depth is preferred in order to be able to calc magn. and orient. 

		Sobel(grayImg, xSobel, CV_32F, 1, 0, 1/*3*/); //xSobel = abs(xSobel);
		Sobel(grayImg, ySobel, CV_32F, 0, 1, 1/*3*/); //ySobel = abs(ySobel);

		cartToPolar(xSobel, ySobel, magnitude, orient, 1);
		//Convert to [-180,180] and then to unsigned vals
		//orient.convertTo(orient,orient.type(),1,-180); orient = abs(orient);

		//Sift-like Gaussian weighting
		GaussianBlur(magnitude, magnitude, Size(0, 0), 0.5*spatial_scale, 0);
		//Dalal and Triggs found that unsigned gradients used in conjunction with 9 histogram channels performed best in their human detection experiments
		grayImg.release();
	}
	else if (strcmp(histName.c_str(), "HOF") == 0){
		split(srcImg, flowMap);
	//	flowMap.at(1).convertTo(flowMap.at(1),CV_32F,-1,0);
	//	flowMap.at(0).convertTo(flowMap.at(0), CV_32F, -1, 0);
		
	//	FileStorage fs("descrF.txt", FileStorage::WRITE);
	//	fs << "flow0" << flowMap.at(0);
	//	fs << "flow1" << flowMap.at(1);
		
		cartToPolar(flowMap.at(0), flowMap.at(1), magnitude, orient, 1);

	//	fs << "magn" << magnitude;
	//	fs.release();
		//Sift-like Gaussian weighting
		GaussianBlur(magnitude, magnitude, Size(0, 0), 0.5*spatial_scale, 0);
		
	}
	else if (strcmp(histName.c_str(), "xMBH") == 0){
		split(srcImg, flowMap);
		Mat tempFlowX;

		//Extract the derivative for each flow channel
		Sobel(flowMap.at(0), xSobel, CV_32F, 1, 0, 1/*3*/); //xSobel = abs(xSobel);
		Sobel(flowMap.at(0), ySobel, CV_32F, 0, 1, 1/*3*/); //ySobel = abs(ySobel);
		cartToPolar(xSobel, ySobel, magnitude, orient, 1);
		//Sift-like Gaussian weighting
		GaussianBlur(magnitude, magnitude, Size(0, 0), 0.5*spatial_scale, 0);
	}
	else if (strcmp(histName.c_str(), "yMBH") == 0){
		split(srcImg, flowMap);

		//Extract the derivative for each flow channel
		Sobel(flowMap.at(1), xSobel, CV_32F, 1, 0, 1/*3*/); //xSobel = abs(xSobel);
		Sobel(flowMap.at(1), ySobel, CV_32F, 0, 1, 1/*3*/); //ySobel = abs(ySobel);
		cartToPolar(xSobel, ySobel, magnitude, orient, 1);
		//Sift-like Gaussian weighting
		GaussianBlur(magnitude, magnitude, Size(0, 0), 0.5*spatial_scale, 0);
	}

	/* Create an array of 9 images (9 because I assume bin size 20 degrees and unsigned gradient
	( 180/20 = 9), one for each bin which will have zeros for all pixels, except for the pixels
	in the original image for which the gradient values correspond to the particular bin.
	These will be referred to as bin images. These bin images will be then used to calculate the
	integral histogram, which will quicken the calculation of HOG descriptors */
#pragma region(init structs)
	vector<Mat> bins(9), integrals(9);
	/*
	if (strcmp(histName.c_str(),"HOG")!=0){
	bins.resize(18); integrals.resize(18);
	}*/

	for (int i = 0; i < (int)bins.size(); i++) {
		bins.at(i).create(srcImg.rows, srcImg.cols, CV_32F);
		bins[i] = Scalar::all(0);
	}
	for (int i = 0; i < (int)integrals.size(); i++) {
		integrals.at(i).create(srcImg.rows + 1, srcImg.cols + 1, CV_64F);
		integrals[i] = Scalar::all(0);
	}


#pragma endregion
	/* Calculate the bin images. The magnitude and orientation of the gradient at each pixel is used.{Magnitude = sqrt(sq(xsobel) + sq(ysobel) ), gradient = itan (ysobel/xsobel) }. Then according to the orientation of the gradient, the value of the corresponding pixel in the corresponding image is set */
	for (int x = 0; x <srcImg.rows; x++) {
		/*For every pixel in a row gradient orientation and magnitude are calculated and corresponding values set for the bin images. */
		for (int y = 0; y <srcImg.cols; y++) {

			/* if the xsobel derivative is zero for a pixel, a small value is added to it, to avoid division by zero. atan returns values in radians, which on being converted to degrees, correspond to values between -90 and 90 degrees. 90 is added to each orientation, to shift the orientation values range from {-90-90} to {0-180}. This is just a matter of convention. {-90-90} values can also be used for the calculation. */

			/*The bin image is selected according to the gradient values. The corresponding pixel value is made equal to the gradient magnitude at that pixel in the corresponding bin image */
			//cout << orient.at<float>(x,y) << endl;
#pragma region(binning)
			if (strcmp(histName.c_str(), "HOG") == 0){
				if (orient.at<float>(x, y) <= 40) {
					bins.at(0).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) <= 80) {
					bins.at(1).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) <= 120) {
					bins.at(2).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) <= 160) {
					bins.at(3).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) <= 200) {
					bins.at(4).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) <= 240) {
					bins.at(5).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) <= 280) {
					bins.at(6).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) <= 320) {
					bins.at(7).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else {
					bins.at(8).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				}
			else{
				if (orient.at<float>(x, y) >= 250 && orient.at<float>(x, y)<290) {
					bins.at(0).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) >= 230 && orient.at<float>(x, y) < 310) {
					bins.at(1).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) >= 210 && orient.at<float>(x, y) < 330){
					bins.at(2).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) >= 190 && orient.at<float>(x, y) < 350) {
					bins.at(3).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) >= 170 && orient.at<float>(x, y) <= 360) {
					bins.at(4).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) >= 150){
					bins.at(5).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) >= 130) {
					bins.at(6).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) >= 110) {
					bins.at(7).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else if (orient.at<float>(x, y) >= 90){
					bins.at(8).at<float>(x, y) = magnitude.at<float>(x, y);
					}
				else{
					if (orient.at<float>(x, y) < 10 && orient.at<float>(x, y) >= 0) {
						bins.at(4).at<float>(x, y) = magnitude.at<float>(x, y);
						}
					else if (orient.at<float>(x, y) < 30) {
						bins.at(5).at<float>(x, y) = magnitude.at<float>(x, y);
						}
					else if (orient.at<float>(x, y) < 50) {
						bins.at(6).at<float>(x, y) = magnitude.at<float>(x, y);
						}
					else if (orient.at<float>(x, y) < 70) {
						bins.at(7).at<float>(x, y) = magnitude.at<float>(x, y);
						}
					else {
						bins.at(8).at<float>(x, y) = magnitude.at<float>(x, y);
						}
					}
				}
#pragma endregion
			}
		}
	xSobel.release(); ySobel.release();
	flowMap.at(0).release(); flowMap.at(1).release();

	/*Integral images for each of the bin images are calculated*/
#pragma region(release structs)
	for (int i = 0; i < (int)bins.size(); i++)
		integral(bins[i], integrals[i]);
	for (int i = 0; i < (int)integrals.size(); i++)
		bins.at(i).release();
#pragma endregion

	/*The function returns an array of 9 images which consitute the integral histogram*/
	return (integrals);
	}
void calculateHOG_rect(Rect cell, Mat& hog_cell, vector<Mat> integrals, bool doNorm) {


	/* Calculate the bin values for each of the bin of the histogram one by one */

	for (int i = 0; i < (int)integrals.size(); i++){

		double a = integrals.at(i).at<double>(Point(cell.x, cell.y));// ((double*)(integrals[i]->imageData + (cell.y)*(integrals[i]->widthStep)))[cell.x];
		double b = integrals.at(i).at<double>(Point(cell.x + cell.width, cell.y + cell.height)); // ((double*)(integrals[i]->imageData + (cell.y + cell.height)*(integrals[i]->widthStep)))[cell.x + cell.width];
		double c = integrals.at(i).at<double>(Point(cell.x + cell.width, cell.y)); //((double*)(integrals[i]->imageData + (cell.y) * (integrals[i]->widthStep)))[cell.x + cell.width];
		double d = integrals.at(i).at<double>(Point(cell.x, cell.y + cell.height));//((double*)(integrals[i]->imageData + (cell.y + cell.height) * (integrals[i]->widthStep)))[cell.x];

		hog_cell.push_back((a + b) - (c + d));
		}

	/*Normalize the matrix if asked*/
	if (doNorm == 1)
		normalize(hog_cell, hog_cell, 1, 0, NORM_L2);
	}

/* This function takes in a block as a rectangle and calculates the hog features for the block by dividing
it into cells of size cell(the supplied parameter), calculating the hog features for each cell using the
function calculateHOG_rect(...), concatenating the so obtained vectors for each cell and then normalizing over
the concatenated vector to obtain the hog features for a block
NormL1: ||Src||_1 =1 , NormL2: ||Src||_2 =1*/
void calculateHist_block(Rect block, Mat& hog_block, vector<Mat> integrals, bool doNorm){
	Size cell_Size = Size(block.width / 2, block.height / 2);
	int cell_start_x, cell_start_y;
	Mat vector_cell;

	//int startcol = 0;
	for (cell_start_y = block.y; cell_start_y <= block.y + cell_Size.height; cell_start_y += cell_Size.height)
		{
		for (cell_start_x = block.x; cell_start_x <= block.x + cell_Size.width; cell_start_x += cell_Size.width)
			{
			//vector_cell=hog_block.colRange(startcol,startcol+9);
			//cvGetCols(hog_block, &vector_cell, startcol, startcol + 9);

			calculateHOG_rect(Rect(cell_start_x, cell_start_y, cell_Size.width, cell_Size.height), vector_cell, integrals, doNorm);
			//vector_cell.convertTo(vector_cell, CV_32F);
			hog_block.push_back(vector_cell);
			vector_cell.release();
			//startcol += 9;
			}
		}
	if (doNorm == 1)
		normalize(hog_block, hog_block, 1, 0, NORM_L2);
	}


// Calculates descriptors for all superpixels in current frame
void calcHogHofDescr(vector<sp>& sp_vec, Mat flow, Mat frame, int spatial_scale){

	vector<Mat>OF(2);
	split(flow,OF);
	//OF.at(1) = -OF.at(1);

	Mat hogHist, hofHist;
	Rect blockRect;
	vector<Mat> integralHOGImages, integralHOFImages;
	integralHOGImages = calculateIntegralHist(frame, "HOG", 2*spatial_scale);
	integralHOFImages = calculateIntegralHist(flow, "HOF", 2*spatial_scale);
	//Mat drawImg;
	//frame.copyTo(drawImg);

	// Loop for all superpixels
	for (int spId = 0; spId < sp_vec.size(); spId++){

		// Initialize rectangle used to calculate lbp descriptor 
		Point2i center_tmp = sp_vec.at(spId).center;
		blockRect.x = center_tmp.x - spatial_scale; blockRect.y = center_tmp.y - spatial_scale;
		blockRect.width = blockRect.height = (spatial_scale * 2);
		if (blockRect.x <= 0) blockRect.x = 0;
		if (blockRect.y <= 0) blockRect.y = 0;
		if (blockRect.x + blockRect.width > frame.cols) blockRect.x = frame.cols - blockRect.width;
		if (blockRect.y + blockRect.height > frame.rows) blockRect.y = frame.rows - blockRect.height;

		calculateHist_block(blockRect, hogHist, integralHOGImages, true);
		sp_vec.at(spId).HogDescr.push_back(hogHist);		
		hogHist.release();

		calculateHist_block(blockRect, hofHist, integralHOFImages, true);
		sp_vec.at(spId).HofDescr.push_back(hofHist);
		hofHist.release();

		//rectangle(drawImg, blockRect, Scalar(0, 0, 255), 1, 8, 0 );
	}

	//namedWindow("sampled rectangles", 1);
	//imshow("sampled rectangles", drawImg);
	//waitKey(30);
}

// Resets descriptors
void resetHogHofDescr(vector<sp>& sp_vec){
	sp sp_tmp;
	for (int spId = 0; spId < sp_vec.size(); spId++){

		sp_tmp = sp_vec.at(spId);
		sp_tmp.historyVec.clear();
		sp_tmp.HogDescr.clear();
		sp_tmp.HofDescr.clear();
		sp_tmp.finalDescr.clear();
		sp_tmp.water_flag = false;
		sp_vec[spId] = sp_tmp;
	}
}