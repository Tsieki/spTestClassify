#include "functions.h"

//Load text file with video list
vector<string> load_list(const string& fname)
{
	vector<string> ret;
	ifstream fobj(fname.c_str());
	if (!fobj.good()) { cerr << "File " << fname << " not found!\n"; exit(-1); }
	string line;
	while (getline(fobj, line)) {
		ret.push_back(line);
	}
	return ret;
}

// Returns mask of the frame according to superpixels inference
Mat createMask(vector<sp> sp_vec, Mat frame, int timestamp){

	Mat maskMat (frame.rows,frame.cols,CV_8U, Scalar::all(0));

	for (int spId = 0; spId < sp_vec.size(); spId++){
		
		sp sp_tmp = sp_vec.at(spId);
		if (sp_tmp.water_flag == false) continue;
		else{
			// Loop for all pixels belonging to sp.
			for (int idx = 0; idx < sp_tmp.historyVec.at(timestamp).size(); idx ++){
				maskMat.at<uchar>(sp_tmp.historyVec.at(timestamp).at(idx).y, (int)sp_tmp.historyVec.at(timestamp).at(idx).x) = 255;
			} }
	}
	imwrite("test.jpg",maskMat);
	return maskMat;
}

// Calculates the superpixel of the top layer that the sp of the bottom layer is found.
int calcTopSP(Point2i spCenter, vector<sp> topSp_vec){

	int spIndex = -1000;
	for (int idx = 0; idx < topSp_vec.size(); idx ++){
		vector<Point2f> spTemp = topSp_vec.at(idx).historyVec.back();
		if (find(spTemp.begin(),spTemp.end(),(Point2f)spCenter) != spTemp.end()){
			spIndex = idx;
			break;
		}
		else continue;
	}

	if (spIndex == -1000) cout << "Error: Superpixel of point not found!!!!\n";
	return spIndex;
}

// Matches bottom layers withe the top layer. Now working with only 2 layers (bottom and top).
// It returns a vector: 1st dimension: corresponds to sp center of the top layer, 2nd dimension: includes the indices of sp centers of bottom layers that will contribute to the the top layer's sp center
vector<vector<int>> calcMixLayers(vector<vector<sp>> sp_vec, vector<vector<Point2i>> centers){
	
	vector<vector<int>> mixLayersIdx(centers.back().size());
	for (int layerbtmId = 0; layerbtmId < centers.size()-1; layerbtmId++){

		for (int spId = 0; spId < sp_vec.at(layerbtmId).size(); spId++){
			int idxTmp = calcTopSP(sp_vec.at(layerbtmId).at(spId).center, sp_vec.back());
			mixLayersIdx.at(idxTmp).push_back(spId);
		}

	}
	
	return mixLayersIdx;
}

	

void multiSVM(vector<vector<sp>>& sp_vec, vector<svm_model*> modelVec, vector<vector<int>> mixLayersIdx){

	struct svm_node *solution;
	struct svm_model *modelSVM = modelVec.at(0);
	//struct svm_model *model = svm_load_model("model");
	
	//Load test data to solution and predict its class
	double confidence; // Stores the class testdata belongs to according to svm prediction
	int nr_class = svm_get_nr_class(modelSVM);

	fstream fs; // Stores scores.
	fs.open("scores.txt", fstream::out | fstream::app | fstream::in);
	if (!fs) { cout<<"Cannot open file to write scores!!! \n"; system("pause");}

	for (int spIdTop = 0; spIdTop < sp_vec.back().size(); spIdTop++){
		
		double avgScore = 0;;
		// Loop for all models
		for (int modelIdx = 0; modelIdx < modelVec.size(); modelIdx++){
			modelSVM = modelVec.at(modelIdx);
			
			Mat testData = sp_vec.back().at(spIdTop).finalDescr[modelIdx]; // Test data has as many rows as the fisher descriptors for current model.

			solution = (struct svm_node*) malloc((testData.cols+1)*sizeof(struct svm_node));
			for (int i=0;i<testData.cols;i++){
				solution[i].index = i+1;
				solution[i].value = (double) testData.at<float>(0,i);
			}
			solution[testData.cols].index=-1;

			vector<double> dec_Values((nr_class*(nr_class-1)/2));
			confidence = svm_predict_values(modelSVM,solution, &dec_Values[0]);
			
			for (int tmp = 0; tmp < dec_Values.size(); tmp++){
				avgScore = avgScore + dec_Values[tmp];
				fs << dec_Values.at(tmp) << " ";
			}
			free(solution);
			// Uncomment for late fusion
/*
// 			if (flag_lateFusion == true){
// 				double avgScore = dec_Values.at(0);
// 				int layerBtmId = 0;
// 				for (int spIdBtm = 0; spIdBtm < mixLayersIdx.at(spIdTop).size(); spIdBtm++){
// 					if (spIdBtm != 0 && mixLayersIdx.at(spIdTop).at(spIdBtm-1) >= mixLayersIdx.at(spIdTop).at(spIdBtm)) layerBtmId++;
// 					testData = sp_vec[layerBtmId][mixLayersIdx.at(spIdTop).at(spIdBtm)].finalDescr[0];
// 
// 					solution = (struct svm_node*) malloc((testData.cols+1)*sizeof(struct svm_node));
// 					for (int i=0;i<testData.cols;i++){
// 						solution[i].index = i+1;
// 						solution[i].value = (double) testData.at<float>(0,i);
// 					}
// 					solution[testData.cols].index=-1;
// 
// 					vector<double> dec_Values((nr_class*(nr_class-1)/2));
// 					confidence = svm_predict_values(model,solution, &dec_Values[0]);
// 					avgScore = avgScore + dec_Values.at(0);
// 
// 					if(confidence == 0) sp_vec[layerBtmId][mixLayersIdx.at(spIdTop).at(spIdBtm)].water_flag = true;
// 				}
// 				avgScore = avgScore/(mixLayersIdx.at(spIdTop).size()+1);
// 				if (avgScore >= 0) sp_vec.back().at(spIdTop).water_flag = true;
// 
// 				for (int idx = 0; idx < dec_Values.size(); idx++)
// 					fs << avgScore << " ";
// 			}
*/
		//if(confidence == 0) sp_vec.back().at(spIdTop).water_flag = true;
		
		}
		avgScore = avgScore/modelVec.size();
		if (avgScore <=0) sp_vec.back().at(spIdTop).water_flag = true;
		fs << endl;
		//if (confidence == 1.0) system("pause");
		
	}		
	fs.close();
	//svm_free_and_destroy_model(&model);
		
	return;
}