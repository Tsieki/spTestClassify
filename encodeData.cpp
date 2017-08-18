#include "functions.h"
PCA PCAtransform;

void initPCA(Mat features,int n_dim, string filePCA){
	if(!features.empty()){
		PCAtransform(features,Mat(),0 ,n_dim);
		//cout << "featsizeWrite" << features.size() << endl;
		FileStorage fs(filePCA,FileStorage::WRITE);
		cout << "eigenVals" << PCAtransform.eigenvalues.size() << endl;
		fs << "eigenVals" << PCAtransform.eigenvalues;
		cout << "eigenVecs" << PCAtransform.eigenvalues.size() << endl;
		fs << "eigenVecs" << PCAtransform.eigenvectors;
		cout << "eigenMean" << PCAtransform.eigenvalues.size() << endl;
		fs << "eigenMean" << PCAtransform.mean;
		fs.release();
	}
	else{
		//cout << "featsizeWrite" << features.size() << endl;
		FileStorage fs(filePCA, FileStorage::READ);
		fs["eigenVals"] >> PCAtransform.eigenvalues;
		fs["eigenVecs"] >> PCAtransform.eigenvectors;
		fs["eigenMean"] >> PCAtransform.mean;
		fs.release();
	}	
	return;
}

void computePCA(Mat inData,Mat& PCAdata){
	PCAtransform.project(inData,PCAdata);
	return;
}

// Descriptor is of size (num_descr x descr_dimension) which means (n x 217) for hoghof and (n x 5220) for lbp.
// Disher descripto is of type CV_32FC1!!!
Mat fisherEncode(Mat descriptor,int n_gauss,int n_dim, string GMMstr, string filePCA){
	//em_param em;geia
	const char * c = GMMstr.c_str();
	gaussian_mixture<float> *gmmproc = new gaussian_mixture<float>(c);
	//	 gmmproc->print(false,true,true);
	//Initiallize a fisher encoder based on the trained GMM
	fisher_param fisher_encoder_params;

	fisher_handle<float> *fisher_encoder = new fisher_handle<float>(*gmmproc, fisher_encoder_params);
	// initialize encoder with a GMM model (vocabulary)
	fisher_encoder->set_model(*gmmproc);

	string openFileName;
	initPCA(Mat(),n_dim, filePCA);

	Mat PCAfeatures;
	computePCA(descriptor,PCAfeatures); //Perform PCA reduction to your data
	descriptor.release();

	//		FileStorage fs("pca.xml",FileStorage::WRITE);
	//		fs << "mtx" << PCAfeatures;
	//		fs.release();

	//Load your data to a vector of pointers
	vector<float*> x(PCAfeatures.rows);
	for (int j = 0; j < PCAfeatures.rows; ++j) {
		x[j] = (float*)(PCAfeatures.data + PCAfeatures.step[1] * PCAfeatures.cols*j);
	}

		//compute your fisher kernel
	float* fk = (float*)malloc(2 * n_gauss*n_dim*sizeof(float));
	fisher_encoder->compute(x, fk);

	//normalize L2 Fisher kernel
	Mat normFisher(1,2*n_dim*n_gauss,CV_32FC1);
	for (int k=0;k<2*n_gauss*n_dim;k++)
		normFisher.at<float>(0,k) = fk[k];
		normFisher.convertTo(normFisher,CV_32FC1,1.0/norm(normFisher,NORM_L2),0);

	//pass your data from a Hellinger Kernel
	Mat hellingerData(1,2*n_dim*n_gauss,CV_32FC1);
	for (int k=0;k<2*n_gauss*n_dim;k++){
		if(normFisher.at<float>(0,k)<0)
			hellingerData.at<float>(0,k)=(-1.0)*sqrt(abs(normFisher.at<float>(0,k)));
		else
			hellingerData.at<float>(0,k)= sqrt(abs(normFisher.at<float>(0,k)));
	}
	hellingerData.convertTo(hellingerData,CV_32FC1,1.0/norm(hellingerData,NORM_L2),0);

	/*Write the Fisher Kernel to the hard drive
		FileStorage fsHdata;
		openFileName="D:\\projects\\dynamicTextures\\x64\\Release\\result\\descriptors"+ResPrefix+"\\"+descNames.at(i)+"_15.bow";
		fsHdata.open(openFileName,FileStorage::WRITE);
		fsHdata << "fsKernel" << hellingerData;
		fsHdata.release();*/

		//fopen_s(&bowDesc,openFileName.c_str(),"wb");
		//for (int k=0;k<2*n_gauss*n_dim;k++)
		//	fprintf_s(bowDesc,"%e ", fk[k]);
		//fclose(bowDesc);
		//openFileName.clear();
		//hellingerData.release();

		free(fk); x.clear();
		PCAfeatures.release();
	return hellingerData;
}

// Calculates fisher descriptor for a particular SVM model in case that HogHof features are used.
void calcModelFisherHOGHOF(vector<vector<sp>>& sp_vec, string GMMfile, string PCAfile, int winLen, int trajLen, vector<vector<int>> mixLayersIdx, int n_gauss, int n_dim, int sizeFisher){

	int num_descr = trajLen/winLen; // Number of descriptors that will be created. If 1 svm model is used num_descr = 1 as trajLen = winLen.
	vector<float> tmpdescrVec;
	Mat descriptor;
		
	// Loop for each sp in top layer
	for (int spIdTop = 0; spIdTop < mixLayersIdx.size(); spIdTop++){
		//clock_t tStart = clock();
		Mat fisherDescr(num_descr,sizeFisher,CV_32FC1);

		for (int descrId = 0; descrId < num_descr; descrId++){ // Calculates all the descriptors
				
			int layerBtmId = 0;

			// Loop for each sp in bottom layer that belongs to the specific top superpixel.
			for (int spIdBtm = 0; spIdBtm < mixLayersIdx.at(spIdTop).size(); spIdBtm++){

				if (spIdBtm != 0 && mixLayersIdx.at(spIdTop).at(spIdBtm-1) >= mixLayersIdx.at(spIdTop).at(spIdBtm)) layerBtmId++;
				sp sp_tmp = sp_vec.at(layerBtmId).at(mixLayersIdx.at(spIdTop).at(spIdBtm));

				//sampleVal is used for downsizing the feature vectors to the desired size.
				float sampleVal = 0;
				vector<Mat> hog_cell(4), hof_cell(4);
				// Initialise hog_cell and hof_cell
				for (int hog_i = 0; hog_i < (int)hog_cell.size(); hog_i++) {
					hog_cell.at(hog_i).create(9, 1, CV_64F); hog_cell.at(hog_i) = Scalar::all(0);
				}
				for (int hof_i = 0; hof_i<(int)hof_cell.size(); hof_i++) {
					hof_cell.at(hof_i).create(9, 1, CV_64F); hof_cell.at(hof_i) = Scalar::all(0);
				}

				sampleVal = (sp_tmp.historyVec.size() / 3); // Equals to winSize/3.

				for (int idx = 0; idx < (int)sp_tmp.HogDescr.size(); idx = idx++){

					// Hog and Hof descriptors are stored according to the cell they belong in hog_cell and hof_cell respectively
					for (int hog_i = 0; hog_i < (int)hog_cell.size(); hog_i++)
						add(hog_cell.at(hog_i), sp_tmp.HogDescr.at(idx).rowRange(hog_i * 9, ((hog_i + 1) * 9)), hog_cell.at(hog_i));
					for (int hof_i = 0; hof_i < (int)hof_cell.size(); hof_i++)
						add(hof_cell.at(hof_i), sp_tmp.HofDescr.at(idx).rowRange(hof_i * 9, ((hof_i + 1) * 9)), hof_cell.at(hof_i));

					if (idx == cvRound(sampleVal - 1.0)) {

						for (int hog_i = 0; hog_i < (int)hog_cell.size(); hog_i++) {
							normalize(hog_cell.at(hog_i), hog_cell.at(hog_i), 1, 0, NORM_L2);
							for (int idx2 = 0; idx2 < 9; idx2++){
								float tmp = (float)hog_cell.at(hog_i).at<double>(idx2, 0);
								tmpdescrVec.push_back(tmp);
							}
							hog_cell.at(hog_i) = Scalar::all(0);
						}
						for (int hof_i = 0; hof_i < (int)hof_cell.size(); hof_i++) {
							normalize(hof_cell.at(hof_i), hof_cell.at(hof_i), 1, 0, NORM_L2);
							for (int idx2 = 0; idx2 < 9; idx2++){
								float tmp = (float)hof_cell.at(hof_i).at<double>(idx2, 0);
								tmpdescrVec.push_back(tmp);
							}
							hof_cell.at(hof_i) = Scalar::all(0);
						}

						sampleVal += (sp_tmp.historyVec.size() / 3);

					}

				}
				//Initialize Mat descriptor
				Mat descriptorTmp(tmpdescrVec);
				descriptorTmp = descriptorTmp.reshape(0,1);
				if (descriptor.empty()) descriptorTmp.copyTo(descriptor); // Initialize descriptor Mat
				else
					descriptor.push_back(descriptorTmp);
				
				descriptorTmp.release();
				tmpdescrVec.clear();
			}

			// Finally add the descriptor of the top layer.
			sp sp_tmp = sp_vec.back().at(spIdTop);

			//sampleVal is used for downsizing the feature vectors to the desired size.
			float sampleVal = 0;
			vector<Mat> hog_cell(4), hof_cell(4);
			// Initialise hog_cell and hof_cell
			for (int hog_i = 0; hog_i < (int)hog_cell.size(); hog_i++) {
				hog_cell.at(hog_i).create(9, 1, CV_64F); hog_cell.at(hog_i) = Scalar::all(0);
			}
			for (int hof_i = 0; hof_i<(int)hof_cell.size(); hof_i++) {
				hof_cell.at(hof_i).create(9, 1, CV_64F); hof_cell.at(hof_i) = Scalar::all(0);
			}

			sampleVal = (sp_tmp.historyVec.size() / 3); // Equals to winSize/3.

			for (int idx = 0; idx < (int)sp_tmp.HogDescr.size(); idx = idx++){

				// Hog and Hof descriptors are stored according to the cell they belong in hog_cell and hof_cell respectively
				for (int hog_i = 0; hog_i < (int)hog_cell.size(); hog_i++)
					add(hog_cell.at(hog_i), sp_tmp.HogDescr.at(idx).rowRange(hog_i * 9, ((hog_i + 1) * 9)), hog_cell.at(hog_i));
				for (int hof_i = 0; hof_i < (int)hof_cell.size(); hof_i++)
					add(hof_cell.at(hof_i), sp_tmp.HofDescr.at(idx).rowRange(hof_i * 9, ((hof_i + 1) * 9)), hof_cell.at(hof_i));

				if (idx == cvRound(sampleVal - 1.0)) {

					for (int hog_i = 0; hog_i < (int)hog_cell.size(); hog_i++) {
						normalize(hog_cell.at(hog_i), hog_cell.at(hog_i), 1, 0, NORM_L2);
						for (int idx2 = 0; idx2 < 9; idx2++){
							float tmp = (float)hog_cell.at(hog_i).at<double>(idx2, 0);
							tmpdescrVec.push_back(tmp);
						}
						hog_cell.at(hog_i) = Scalar::all(0);
					}
					for (int hof_i = 0; hof_i < (int)hof_cell.size(); hof_i++) {
						normalize(hof_cell.at(hof_i), hof_cell.at(hof_i), 1, 0, NORM_L2);
						for (int idx2 = 0; idx2 < 9; idx2++){
							float tmp = (float)hof_cell.at(hof_i).at<double>(idx2, 0);
							tmpdescrVec.push_back(tmp);
						}
						hof_cell.at(hof_i) = Scalar::all(0);
					}
					sampleVal += (sp_tmp.historyVec.size() / 3);

				}

			}

			Mat descriptorTmp(tmpdescrVec);
			descriptorTmp = descriptorTmp.reshape(0,1);
			descriptor.push_back(descriptorTmp);

			Mat fisherTmp =	fisherEncode(descriptor, n_gauss, n_dim, GMMfile, PCAfile);
			fisherTmp.copyTo(fisherDescr.row(descrId));

			tmpdescrVec.clear();
			descriptor.release();
		}
		//FileStorage fs("fisher.txt", FileStorage::WRITE);
		//fs << "fisher" << fisherDescr.row(2);
		//fs.release();
		sp_vec[sp_vec.size()-1][spIdTop].finalDescr.push_back(fisherDescr); // Updates top layer with final descriptor
		fisherDescr.release();

	//Writes descriptor. Uncomment for c-nn.
	/*
	// 		for (int cols = 0; cols < sp_tmp.finalDescr.cols; cols++){
	// 			fs << sp_tmp.finalDescr.at<float>(0,cols) << " ";
	// 		}
	// 		fs <<endl;
	*/
	//printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
		}
}


// Calculates fisher descriptor for a particular SVM model in case that LBP features are used.
void calcModelFisherLBP(vector<vector<sp>>& sp_vec, string GMMfile, string PCAfile, int winLen, int trajLen, vector<vector<int>> mixLayersIdx, int n_gauss, int n_dim, int sizeFisher){

	int num_descr = trajLen/winLen; // Number of descriptors that will be created.
	Mat descriptorTmp, descriptor;
	vector<Mat> descrAllVec;
		
	// Loop for each sp in top layer
	for (int spIdTop = 0; spIdTop < mixLayersIdx.size(); spIdTop++){
		//clock_t tStart = clock();
		Mat fisherDescr(num_descr,sizeFisher,CV_32FC1);

		for (int descrId = 0; descrId < num_descr; descrId++){ // Calculates all the descriptors
				
			int layerBtmId = 0;

			// Loop for each sp in bottom layer that belongs to the specific top superpixel.
			for (int spIdBtm = 0; spIdBtm < mixLayersIdx.at(spIdTop).size(); spIdBtm++){

				if (spIdBtm != 0 && mixLayersIdx.at(spIdTop).at(spIdBtm-1) >= mixLayersIdx.at(spIdTop).at(spIdBtm)) layerBtmId++;
				sp sp_tmp = sp_vec.at(layerBtmId).at(mixLayersIdx.at(spIdTop).at(spIdBtm));

				// Concatenate descriptors
				descrAllVec.insert(descrAllVec.end(),sp_tmp.LBPxyDescr.begin() + descrId*winLen, sp_tmp.LBPxyDescr.begin() + descrId*winLen + winLen);
				descrAllVec.insert(descrAllVec.end(),sp_tmp.LBPxtDescr.begin() + descrId*winLen, sp_tmp.LBPxtDescr.begin() + descrId*winLen + winLen);
				descrAllVec.insert(descrAllVec.end(),sp_tmp.LBPytDescr.begin() + descrId*winLen, sp_tmp.LBPytDescr.begin() + descrId*winLen + winLen);

				for (int idx= 0; idx < descrAllVec.size(); idx++){
					descriptorTmp.push_back(descrAllVec.at(idx));
				}
				descriptorTmp = descriptorTmp.reshape(0,1);

				//if (flag_lateFusion == true){ // Stores fisher for all sps of bottom layer
				//	sp_tmp.finalDescr = fisherEncode(descriptorTmp, n_gauss,n_dim, GMMstr[modelId], filePCA[modelId]);
				//	sp_vec[layerBtmId][mixLayersIdx.at(spIdTop).at(spIdBtm)] = sp_tmp; // stores fisher for specific superpixel
				//}

				if (descriptor.empty()) descriptorTmp.copyTo(descriptor); // Initialize descriptor Mat
				else
					descriptor.push_back(descriptorTmp);

				descrAllVec.clear();
				descriptorTmp.release();
			}

			// Finally add the descriptor of the top layer.
			sp sp_tmp = sp_vec.back().at(spIdTop);

			// Concatenate descriptors
			descrAllVec.insert(descrAllVec.end(),sp_tmp.LBPxyDescr.begin() + descrId*winLen, sp_tmp.LBPxyDescr.begin() + descrId*winLen + winLen);
			descrAllVec.insert(descrAllVec.end(),sp_tmp.LBPxtDescr.begin() + descrId*winLen, sp_tmp.LBPxtDescr.begin() + descrId*winLen+ winLen);
			descrAllVec.insert(descrAllVec.end(),sp_tmp.LBPytDescr.begin() + descrId*winLen, sp_tmp.LBPytDescr.begin() + descrId*winLen + winLen);

			for (int idx= 0; idx < descrAllVec.size(); idx++){
				descriptorTmp.push_back(descrAllVec.at(idx));
			}
			descriptorTmp = descriptorTmp.reshape(0,1);

			descriptor.push_back(descriptorTmp);
			//descriptor.push_back(descriptorTmp);

			Mat fisherTmp =	fisherEncode(descriptor, n_gauss, n_dim, GMMfile, PCAfile);
			fisherTmp.copyTo(fisherDescr.row(descrId));
	
			descriptorTmp.release();
			descrAllVec.clear();
			descriptor.release();
		}
		//FileStorage fs("fisher.txt", FileStorage::WRITE);
		//fs << "fisher" << fisherDescr.row(2);
		//fs.release();
		sp_vec[sp_vec.size()-1][spIdTop].finalDescr.push_back(fisherDescr); // Updates top layer with final descriptor
		fisherDescr.release();

	//Writes descriptor. Uncomment for c-nn.
	/*
	// 		for (int cols = 0; cols < sp_tmp.finalDescr.cols; cols++){
	// 			fs << sp_tmp.finalDescr.at<float>(0,cols) << " ";
	// 		}
	// 		fs <<endl;
	*/
	//printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
		}
}

void calcFisher(vector<vector<sp>>& sp_vec, int trajLen, int n_gauss,int n_dim, vector<vector<int>> mixLayersIdx, vector<svm_model*> model, int sizeFisher, string GMMstr[], string filePCA[], int trajLens[], bool calcLBP){

	vector<Mat> descrAllVec;
	Mat descriptorTmp;
	vector<Mat> descriptorFinal; // It contains 1 Mat per SVM model.
	//fstream fs ("testDescr.txt", ios::out);
	//if (!fs.is_open()) cout << "Error opening file to write descriptors!!!";
	
	// Loop for all SVM models and return fisher descriptors for these models. vector<Mat> finalDescr of sp_vec is updated in the top layer containing all fishers for all models. 
	for (int modelId = 0; modelId < model.size(); modelId++){

		if (calcLBP == 1) 
			calcModelFisherLBP(sp_vec, GMMstr[modelId], filePCA[modelId], trajLens[modelId], trajLen, mixLayersIdx, n_gauss, n_dim, sizeFisher);
		else if (calcLBP == 0)
			calcModelFisherHOGHOF(sp_vec, GMMstr[modelId], filePCA[modelId], trajLens[modelId], trajLen, mixLayersIdx, n_gauss, n_dim, sizeFisher);
	}
	
	//fs.close();
	
	multiSVM(sp_vec, model, mixLayersIdx);
	//calcCnnModel(sp_vec);
}


