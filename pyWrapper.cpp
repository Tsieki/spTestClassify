#include "functions.h"
#include <Python.h>

void calcCnnModel(vector<vector<sp>>& sp_vec){
	
	//Py_Initialize();
	PyObject *obj = Py_BuildValue("s", "test.py");

	if (!obj) {
		printf("Error calling Python!!!\n");
		Py_DECREF(obj);
	}
	else {
		FILE *file = _Py_fopen_obj(obj, "r+");
		if(file != NULL) {
			PyRun_SimpleString("import sys\n"
								"sys.argv = ['']");
			PyRun_SimpleFile(file, "test.py");
		}
	}

	Py_DECREF(obj);
	//Py_Finalize();

	vector<double> scoreVec;
	double scoreTmp;
	ifstream scoreFile("water_scores.txt", ios::in);
	if (!scoreFile.is_open()) cout << "Error openning scorefile!!!\n";
	else {
			while(scoreFile >> scoreTmp) scoreVec.push_back(scoreTmp);
		}

	for (int spIdTop = 0; spIdTop < scoreVec.size(); spIdTop++){
		scoreTmp = scoreVec.at(spIdTop);
		if (scoreTmp >=0.5){
			sp_vec.back().at(spIdTop).water_flag = true;
		}
	}
}




