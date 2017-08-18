#include "functions.h"
#include <lbp.h>

// Transforms an image into an LBP encoded mat
Mat LBP_block(Mat srcImg, string histName, int cellSize){

	Mat srcGray;
	if (strcmp(histName.c_str(), "LBPxy") == 0)
		cvtColor(srcImg, srcGray, COLOR_RGB2GRAY);
	else
		srcImg.copyTo(srcGray);
	srcGray.convertTo(srcGray, CV_32FC1);
	

	/* get LBP object */
	VlLbp * lbp = vl_lbp_new(VlLbpUniform, VL_TRUE);
	if (lbp == NULL){
		cout << "Houston, we got an LBP init problem here" << endl;
		return  Mat();
		}

	int lbp_dim0 = srcGray.rows / cellSize;
	int lbp_dim1 = srcGray.cols / cellSize;
	int lbp_dim2 = vl_lbp_get_dimension(lbp);
	int LBP_dim = lbp_dim0*lbp_dim1*lbp_dim2;

	float*features = (float*)vl_malloc(sizeof(float)*LBP_dim);

	vl_lbp_process(lbp, features, (float*)srcGray.data, srcImg.cols, srcImg.rows, cellSize);

	Mat LPBHist;
	LPBHist.create(1, LBP_dim, CV_32FC1);
	LPBHist = Scalar::all(0.0);
	for (int i = 0; i<LBP_dim; i++)
		LPBHist.at<float>(0, i) = features[i];

	//print the LBP descriptor
	/*	FileStorage fsLBP("lbpHist.yml", FileStorage::WRITE);;
		fsLBP << "data" << LPBHist;
		fsLBP.release();
		cout << "check LBP file" << endl;
		getchar();*/

	vl_free(features);
	vl_lbp_delete(lbp);

	return LPBHist;
}

// Calculates descriptors for all superpixels in current frame
void calcLbpDescr(vector<sp>& sp_vec, Mat flow, Mat frame, int spatial_scale){

	vector<Mat>OF(2);
	split(flow,OF);
	//OF.at(1) = -OF.at(1);

	Mat LBPxyHist, LBPxtHist, LBPytHist;
	Rect blockRect;
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

		LBPxyHist = LBP_block(frame(blockRect), "LBPxy",2 * spatial_scale);
		sp_vec.at(spId).LBPxyDescr.push_back(LBPxyHist);
		LBPxyHist.release();

		LBPxtHist = LBP_block(OF.at(0)(blockRect), "LBPxt", 2 * spatial_scale);
		sp_vec.at(spId).LBPxtDescr.push_back(LBPxtHist);
		LBPxtHist.release();

		LBPytHist = LBP_block(OF.at(1)(blockRect), "LBPyt", 2 * spatial_scale);
		sp_vec.at(spId).LBPytDescr.push_back(LBPytHist);
		LBPytHist.release();

		//rectangle(drawImg, blockRect, Scalar(0, 0, 255), 1, 8, 0 );
	}

	//namedWindow("sampled rectangles", 1);
	//imshow("sampled rectangles", drawImg);
	//waitKey(30);
}

// Resets descriptors
void resetLbpDescr(vector<sp>& sp_vec){
	sp sp_tmp;
	for (int spId = 0; spId < sp_vec.size(); spId++){

		sp_tmp = sp_vec.at(spId);
		sp_tmp.historyVec.clear();
		sp_tmp.LBPxtDescr.clear();
		sp_tmp.LBPxyDescr.clear();
		sp_tmp.LBPytDescr.clear();
		sp_tmp.finalDescr.clear();
		sp_tmp.water_flag = false;
		sp_vec[spId] = sp_tmp;
	}
}