#include "functions.h"



// Distance metric
float dist(Point2i p1, Point2i p2, Vec3f p1_lab, Vec3f p2_lab, float compactness, float S)
{
	float dl = p1_lab[0] - p2_lab[0];
	float da = p1_lab[1] - p2_lab[1];
	float db = p1_lab[2] - p2_lab[2];

	float d_lab = sqrtf(dl*dl + da*da + db*db);

	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;

	float d_xy = sqrtf(dx*dx + dy*dy);

	return d_lab + compactness/S * d_xy;
}

// Finds superpixels giving label to each pixel. Updates Mat labels
void findSuperpixels(Mat frame, Mat imlab, vector<Point2i> centers, vector<int>& label_vec, 
	Mat& labels, int S, int m){


		// Initialize distance map
		Mat dists = -1*Mat::ones(imlab.size(), CV_32F);
		Mat window;
		Point2i p1, p2;
		Vec3f p1_lab, p2_lab;
		int width = frame.cols;
		int height = frame.rows;

		// Iterate 10 times. In practice more than enough to converge
		for (int i = 0; i < 10; i++) {
			// For each center...
			for (int c = 0; c < label_vec.size(); c++)
			{
				int label = label_vec[c];
				p1 = centers[c];
				p1_lab = imlab.at<Vec3f>(p1);
				int xmin = max(p1.x-S, 0);
				int ymin = max(p1.y-S, 0);
				int xmax = min(p1.x+S, width-1);
				int ymax = min(p1.y+S, height-1);

				// Search in a window around the center
				window = frame(Range(ymin, ymax), Range(xmin, xmax));

				// Reassign pixels to nearest center
				for (int i = 0; i < window.rows; i++) {
					for (int j = 0; j < window.cols; j++) {
						p2 = Point2i(xmin + j, ymin + i);
						p2_lab = imlab.at<Vec3f>(p2);
						float d = dist(p1, p2, p1_lab, p2_lab, m, S);
						float last_d = dists.at<float>(p2);
						if (d < last_d || last_d == -1) {
							dists.at<float>(p2) = d;
							labels.at<int>(p2) = label;
						}
					}
				}
			}
		}

}

// Shows labels' borders in the frame. 
void showBorders(Mat& frame, Mat& labels){

	Mat sobel = (Mat_<float>(3,3) << -1/16., -2/16., -1/16., 0, 0, 0, 1/16., 2/16., 1/16.);

	labels.convertTo(labels, CV_32F);

	Mat gx, gy, grad;
	filter2D(labels, gx, -1, sobel);
	filter2D(labels, gy, -1, sobel.t());
	magnitude(gx, gy, grad);
	grad = (grad > 1e-4)/255;
	Mat show = 1-grad;
	show.convertTo(show, CV_32F);

	//imwrite("output.jpg",frame);
	// Draw boundaries on original image
	vector<Mat> rgb(3);
	split(frame, rgb);
	for (int i = 0; i < 3; i++) 
		rgb[i] = rgb[i].mul(show);

	merge(rgb, frame);
}


// Calculates areas of each superpixel. Updates sp_vvec which contains all superpixels' areas.
void calcSPArea(vector<sp>& sp_vec, vector<int> label_vec, vector<Point2i> centers, Mat frame, Mat labels){

	// Loop for all labels to create areas containing particular pixels. Firstly, clear sp_vvec and initialize it.
	vector<Point2f> sp_tmpVec; // Vector containing all pixels of one superpixel

	/*
	// 	Mat spframe;
	// 	frame.copyTo(spframe);
	*/

	for (int l = 0; l < label_vec.size(); l++){
		int label_id = label_vec.at(l);
		Point2i tempPoint = centers.at(l);

		sp sp_tmp = sp_vec.at(l);
		sp_tmpVec.clear();

		/*
		// 		int r = rand()%256;
		// 		int g = rand()%256;
		// 		int b = rand()%256;
		*/
		// Search for pixels in the neighbour area
		for (int i = tempPoint.y - frame.rows/4; i < tempPoint.y + frame.rows/4; i++)
		{
			if(i < 0) i = 0;
			if(i >= frame.rows) break;

			for (int j = tempPoint.x - frame.cols/4; j < tempPoint.x + frame.cols/4; j++)
			{
				if(j < 0) j = 0;
				if(j >= frame.cols) break;

				if(labels.at<int>(i,j) == label_id) {
					sp_tmpVec.push_back(Point2f(j,i)); 
					//circle(spframe, Point2f(j,i),1,CV_RGB(r,g,b));
				}
			}
		}
		sp_tmp.historyVec.push_back(sp_tmpVec);
		sp_vec[l] = sp_tmp;
	}
	//imwrite("output_files\\superpixels.jpg", spframe);
}