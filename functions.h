#ifndef FUNCTIONS_H
#define FUNCTIONS_H

//opencv headers
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>

//default c headers
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <time.h>

//GMM-Fisher headers
#include <gmm.h>
#include <fisher.h>
#include <matlab/mlclasshandler/fisher_handle.h>

//LibSVM headers
#include <svm.h>

using namespace std;
using namespace cv;

//struct superpixels
struct sp{
	//sp() : water_flag(false){};	
	vector<vector<Point2f>> historyVec; // 1st dimension: timestamp, 2nd dimension: pixels forming superpixel in the specific timestamp
	vector<Mat> LBPxtDescr, LBPytDescr, LBPxyDescr; // contain lbp descriptors
	vector<Mat> HogDescr, HofDescr;
	vector<Mat> finalDescr; // 1 Mat per SVM model. it contains fisher descriptors.
	bool water_flag; // Indicates if sp corresponds to water
	Point2i center; // center of sp

	sp() : water_flag(),center() {}
	sp(bool flag, Point2i c) : water_flag(flag), center(c) {}	
};

vector<string> load_list(const string& fname);
void findSuperpixels(Mat frame, Mat imlab, vector<Point2i> centers, vector<int>& label_vec, Mat& labels, int S, int m);
void showBorders(Mat& frame, Mat& labels);
void calcSPArea(vector<sp>& sp_vec, vector<int> label_vec, vector<Point2i> centers, Mat frame, Mat labels);
Mat createMask(vector<sp> sp_vec, Mat frame, int timestamp);
void calcLbpDescr(vector<sp>& sp_vec, Mat flow, Mat frame, int spatial_scale);
void resetLbpDescr(vector<sp>& sp_vec);
void calcFisher(vector<vector<sp>>& sp_vec, int trajLen, int n_gauss,int n_dim, vector<vector<int>> mixLayersIdx, vector<svm_model*> model, int sizeFisher, string GMMstr[], string filePCA[], int trajLens[], bool calcLBP);
void multiSVM(vector<vector<sp>>& sp_vec, vector<svm_model*> model, vector<vector<int>> mixLayersIdx);
vector<vector<int>> calcMixLayers(vector<vector<sp>> sp_vec, vector<vector<Point2i>> centers);
void calcCnnModel(vector<vector<sp>>& sp_vec);
void calcHogHofDescr(vector<sp>& sp_vec, Mat flow, Mat frame, int spatial_scale);
void resetHogHofDescr(vector<sp>& sp_vec);


#endif