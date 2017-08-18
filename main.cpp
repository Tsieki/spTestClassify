#include "functions.h"
#include <Python.h>

const char* about = "spClassifyTest v1.0.0 ";

const char* params // short name | full name | default value | help (syntax in opencv2.4)
	= "{ h | help        | false | print usage         }"
	"{f | fileName       | D:\\videos\\temporalTextures\\VideoWaterDatabase\\list_of_water_videos.txt | model configuration }"
	"{d |divideVal       | 1.0 | divide the video }"
	"{t |trajLen		 | 30 | trajectory length }"
	"{s |spatial_scale   | 8  | half size of block used for lbp extraction. if 8 block has size 16 x 16 }";

int main(int argc, char *argv[]){

	//Py_Initialize();

	vector<string> videosFilename;			//the vector<> that contains the video
	float divideVal; 		//video downscaling
	int trajLen;			//this is the object detection confidence
	string fileName;	    //filename contains the directory that the videos exist
	//int time_window = 2;	//this is the time window for the boundary mask
	int spatial_scale;	//this is the minimum size of the HOG,HOF  descriptor
	int n_gauss = 16;       //number of Gaussian centers - vocabulary size
	int n_dim = 80;         //dimension of each Gaussian center
	int sizeFisher = 2560;
	bool calcLBP = 0; // If equal to 1 LBP features are used, otherwise hoghof descriptors are calculated
	//vector<String> classesList = load_list("list_of_classes.txt");

/*
// 	string SVMmodels[] = {"model10", "model15", "model30"};
// 	string GMMstr[] = {"GMMVoc10", "GMMVoc15", "GMMVoc30"};
// 	string filePCA[] = {"pcaInfo10.xml", "pcaInfo15.xml", "pcaInfo30.xml"};
// 	int trajLens[] = {10, 15, 30}; // Trajectory lengths corresponding to each svm model.
*/
	string SVMmodels[] = {"model"};
	int trajLens[] = {30};
	string GMMstr[] = {"GMMVoc"};
	string filePCA[] = {"pcaInfo.xml"};


	cv::CommandLineParser parser(argc, argv, params);

	if (parser.get<bool>("help")){
		parser.printParams();
		return 0;
	}
	fileName = parser.get<String>("f");
	divideVal = parser.get<float>("d");
	trajLen = parser.get<int>("t");
	spatial_scale = parser.get<int>("s");
	cout << "spatial scale is: " << spatial_scale << endl;

	//functions that contain information about the trimmed data
	videosFilename = load_list(fileName);
	//Find the directory name - should change to string
	size_t pos;
	pos = fileName.find_last_of("\\");
	string dirName = fileName.substr(0,pos+1);
	cout << "dirName: " << dirName << endl;

	VideoCapture cap;
	
	vector<svm_model*> modelVec;
	for (int modelId = 0; modelId < sizeof(SVMmodels)/sizeof(*SVMmodels); modelId++){
		struct svm_model* model = svm_load_model(SVMmodels[modelId].c_str());
		modelVec.push_back(model);
	}
	

	// Loop for all videos
	for (int vidId = 0; vidId < videosFilename.size(); vidId++){

		cout << dirName + videosFilename.at(vidId)  + ".avi"/*"RGB.avi"*/ << endl;
		cap.open(dirName + videosFilename.at(vidId) + ".avi"/*"RGB.avi"*/);	//capture the video data
		if (!cap.isOpened()){
			cout << "File: " << videosFilename.at(vidId) + ".avi" << endl;
			cout << "is not found! " << endl;
			return -1;}

		string outputVideoName = "result\\videosMask\\"+videosFilename.at(vidId)+".avi";
		VideoWriter outputVideo(outputVideoName, CV_FOURCC('D','I','V','X'), cap.get(CV_CAP_PROP_FPS), Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)),true);

		// Default values for superpixels
		Mat layers_sp = (Mat_<int>(2,2) << 32,32,16,16); // Defines 2 layers of sp (e.g. 16x16 and 8x8). Top layer is found in the last row.
		int m = 100; // Compactness variable. Defines the uniformity of the superpixels.
		vector<int> S(layers_sp.rows,0); 
		vector<int> numCenters(layers_sp.rows);
		for(int idx = 0; idx < layers_sp.rows; idx++){
			numCenters[idx] = layers_sp.at<int>(idx,0) * layers_sp.at<int>(idx,1);
		}
		//int nx = 16; // Defines how many step/cells exist in horizontal axis. 
		//int ny = 16; // Defines how many step/cells exist in vertical axis. 
		//int numCenters = nx*ny; // number of superpixels/centers
		vector<vector<Point2i>> centers(layers_sp.rows);
		vector<vector<int>> label_vec(layers_sp.rows);
		vector<vector<sp>> sp_vec(layers_sp.rows); // Vector containing all superpixels in all layers.
		vector<vector<int>> mixLayersIdx; // 1st dimension: corresponds to sp center of the top layer, 2nd dimension: includes the indices of sp centers of bottom layers that will contribute to the the top layer's sp center

		// Superpixels initializations.
		for (int idx = 0; idx < layers_sp.rows; idx++){

			int nx = layers_sp.at<int>(idx,0), ny = layers_sp.at<int>(idx,1);
			float dx = cap.get(CV_CAP_PROP_FRAME_WIDTH) / float(nx);
			float dy = cap.get(CV_CAP_PROP_FRAME_HEIGHT) / float(ny);
			S[idx] = (dx + dy + 1)/2; // window width

			// Initialize centers and labels
			for (int i = 0; i < ny; i++) {
				for (int j = 0; j < nx; j++) {
					centers.at(idx).push_back( Point2f(j*dx+dx/2, i*dy+dy/2));
					sp sptmp (false, centers.at(idx).back());
					sp_vec[idx].push_back(sptmp); // Initialize vector containing all superpixels
				}
			}

			for (int i = 0; i < numCenters.at(idx); i++)
				label_vec.at(idx).push_back(i*255*255/numCenters.at(idx));

		}		

		namedWindow("input video",1);
		namedWindow("output video",1);

		Mat frame, frameGray, prevFrameGray, flowMask, flow, frameDraw;

		int frame_num = 0;
		for(;;){
			if(frame_num == 800) break;
			cout << ":" <<  cap.get(CV_CAP_PROP_POS_FRAMES);
			if ( !cap.grab()){
				cout << "exit action" << endl;
				sp_vec.clear();
				centers.clear();
				label_vec.clear();
				mixLayersIdx.clear();
				break;
			}
			cap.retrieve(frame);
			resize(frame, frame, Size((int)((float)frame.cols / divideVal), (int)((float)frame.rows / divideVal)));
			cvtColor(frame, frameGray, CV_RGB2GRAY);

			if (prevFrameGray.empty()){
				frameGray.copyTo(prevFrameGray);
				frame_num++;
				continue;
			}
			calcOpticalFlowFarneback(prevFrameGray, frameGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
			//GaussianBlur(flow, flow, Size(5,5), 0);

			frame.convertTo(frame, CV_32F, 1/255.); // Scale to [0,1] and l*a*b colorspace 
			Mat imlab;
			cvtColor(frame, imlab, CV_BGR2Lab);
			frame.copyTo(frameDraw);

			for (int idx = 0; idx < layers_sp.rows; idx ++){

				Mat labels = -1*Mat::ones(imlab.size(), CV_32S);			
				findSuperpixels(frame,imlab,centers.at(idx),label_vec.at(idx),labels,S.at(idx),m);
				calcSPArea(sp_vec.at(idx),label_vec.at(idx),centers.at(idx),frame,labels); // Calculates superpixel's area by finding pixels with common label.
				if (idx == layers_sp.rows-1) showBorders(frameDraw,labels); // show only top layer

				if(calcLBP == 1) 
					calcLbpDescr(sp_vec.at(idx), flow, frame, spatial_scale);
				else if(calcLBP == 0)
					calcHogHofDescr(sp_vec.at(idx),flow,frame,spatial_scale);
			}
			if(mixLayersIdx.empty()) mixLayersIdx = calcMixLayers(sp_vec, centers); // matches bottom layers with the top layer.
			
			imshow("input video", frame);
			imshow("output video",frameDraw);
			waitKey(10);

			if (frame_num%trajLen == 0){
				
				calcFisher(sp_vec,trajLen,n_gauss,n_dim,mixLayersIdx, modelVec,sizeFisher, GMMstr, filePCA, trajLens, calcLBP);

				// Now working only for first layer
				if (!outputVideo.isOpened()) cout << "Cannot open OutputVideo";
				for(int tmp = 0 ; tmp < trajLen; tmp++){
					Mat maskframe = createMask(sp_vec.back(), frame, tmp); // Calculate mask and store it for the last trajLen frames
					cvtColor(maskframe,maskframe,CV_GRAY2BGR);
					//stringstream ss;
					//ss << frame_num - trajLen + tmp;
					//string text = ss.str();
					//putText(maskframe, text, Point2f(300,300),  FONT_HERSHEY_SIMPLEX, 5, Scalar::all(0),5,true); 
					outputVideo.write(maskframe);
					imwrite("test.jpg",maskframe);
					//ss.str("");
				}
				for (int layerId = 0; layerId < sp_vec.size(); layerId++){
					if (calcLBP == 1) 
						resetLbpDescr(sp_vec.at(layerId));
					else if (calcLBP == 0)
						resetHogHofDescr(sp_vec.at(layerId));
				}
			}
			// Write file.
			//if (!outputVideo.isOpened()) cout << "Cannot open OutputVideo";
			//frameDraw.convertTo(frameDraw, CV_8U,255);
			//outputVideo.write(frameDraw);

			swap(prevFrameGray,frameGray);
			frame_num++;
		}// end of video
		destroyWindow("input video");
		destroyWindow("output video");
		cap.release();
		outputVideo.release();
		//Py_Finalize();
		//system("pause");
		return 1;

	}// end of all videos

	for (int modelId = 0; modelId < SVMmodels->size(); modelId++){
		svm_free_and_destroy_model(&modelVec.at(modelId));
	}
		
	return 0;
}
