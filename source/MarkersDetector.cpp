#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//#include "opencv2/imgproc.hpp"
//#include "opencv2/calib3d.hpp"

//#define DEBGUB_DRAW

#include "MarkersDetector.h"
#include <numeric>
#include <map>
#include <array>
#include <iostream>
#include <mutex>

using namespace cv;
using namespace std;

cv::Mat MarkersDetector::androidFrame;
AfterFrameUpdateCallback MarkersDetector::afterFrameUpdateCallback;
BeforeFrameUpdateCallback MarkersDetector::beforeFrameUpdateCallback;

std::mutex frame_mutex;





#ifdef __ANDROID__

JavaVM *gJavaVm = NULL;

jclass AndroidCameraClass;
jmethodID midSetPreviewSize;
jobject AndroidCameraObject;


JNIEnv* getEnv() {
    JNIEnv *env;
    
    int status = gJavaVm->GetEnv((void **) &env, JNI_VERSION_1_6);
    if (status == JNI_EDETACHED)
    {
        status = gJavaVm->AttachCurrentThread(&env, NULL);
        if (status != JNI_OK) {
            ALOGW("AttachCurrentThread failed %d", status);
            return nullptr;
         }
    }
    return env;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *javaVm, void *reserved) {
    gJavaVm = javaVm;
    
    JNIEnv *env = getEnv();
    AndroidCameraClass = env->FindClass("org/getid/markersdetector/AndroidCamera");
    midSetPreviewSize = env->GetMethodID(AndroidCameraClass, "SetPreviewSize", "(II)V");
    
    return JNI_VERSION_1_6;
}



void setPreviewSize(int width, int height) {
    JNIEnv *env = getEnv();
 
    if (!AndroidCameraObject) {
      ALOGW("AndroidCameraObject is null. Check InitJNI call");
      return;
    }
    
    if (!midSetPreviewSize) {
      ALOGW("MethodId for SetPreviewSize is null");
      return;
    }
    
    env->CallVoidMethod(AndroidCameraObject, midSetPreviewSize, width, height);
}
#endif




float Marker::getPerimeter()
{
	float p = arcLength(points, true);
	return p;
}

string intToStr(int a)
{
	stringstream ss;
	ss << a;
    return ss.str();
}

int strToInt(string s)
{
	std::istringstream ss(s);
	int i;
	ss >> i;
	return i;
}

int debugLastYPos = 80;

void drawUsedTime(double t1, Mat frame, string id)
{
    #ifndef DEBGUB_DRAW
        return;
    #endif
    
	double t2 = cv::getTickCount();
	double fps = (t2 - t1)/getTickFrequency();
	
	debugLastYPos += 20;
	
	char strb[256];
    sprintf(strb, "%.3f", fps);
    cv::putText(frame, id + " " + std::string(strb), cv::Point(8,debugLastYPos), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,255,0,255));
}



MarkersDetector::MarkersDetector(std::array<double, 9> cameraMatrixBuf, std::array<double, 8> cameraDistortionBuf, int printedMarkerWidth)
{
	


	m_cameraMatrix = Mat(3, 3, CV_64F);
	int k = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			m_cameraMatrix.at<double>(i, j) = cameraMatrixBuf[k++];
		}
	}

	m_cameraDistortion = Mat(8, 1, CV_64F);
	for (int i = 0; i < 8; i++) {
		m_cameraDistortion.at<double>(0, i) = cameraDistortionBuf[i];
	}

    m_markerCellSize = 10;
	m_markerSize = Size(70, 70);
	m_markerCorners2d.push_back(Point(m_markerSize.width, 0));
	m_markerCorners2d.push_back(Point(m_markerSize.width, m_markerSize.height));
	m_markerCorners2d.push_back(Point(0, m_markerSize.height));
	m_markerCorners2d.push_back(Point(0, 0));
	
	float halfSize = printedMarkerWidth / 2.0;
	m_markerCorners3d.push_back(Point3f(-halfSize, halfSize, 0));
	m_markerCorners3d.push_back(Point3f(-halfSize, -halfSize, 0));
	m_markerCorners3d.push_back(Point3f(halfSize, -halfSize, 0));
	m_markerCorners3d.push_back(Point3f(halfSize, halfSize, 0));

	m_isOpen = false;
}

void MarkersDetector::setMarkersParams(map <int, std::array<float, 3>>* markersLocations)
{
    //std::map <int, std::array<float, 3>> ml = *markersLocations;
	typedef std::map <int, std::array<float, 3>>::iterator it_type;
	for (it_type iterator = markersLocations->begin(); iterator != markersLocations->end(); iterator++) {
		cv::Point3f p = { iterator->second[0], iterator->second[1], iterator->second[2] };
		m_markersLocations[iterator->first] = p;
	}
}

std::vector<Marker> MarkersDetector::detectMarkers(Mat frame)
{

    double t1;
    t1 = cv::getTickCount();
    
	Mat grey = convertToGrey(frame);
	
	drawUsedTime(t1, frame, "convert to grey");
	t1 = cv::getTickCount();
	
	Mat binaryImg = performThreshold(grey);
	
	drawUsedTime(t1, frame, "thres");
	t1 = cv::getTickCount();
	
	std::vector<std::vector<cv::Point>> contours = findContours(binaryImg);
	
    #ifdef __ANDROID__
        ALOGI("found contours: %d", contours.size());
    #endif
	
	drawUsedTime(t1, frame, "find contours");
	t1 = cv::getTickCount();

	std::vector<Marker> markers = findPossibleMarkers(contours);
	
	drawUsedTime(t1, frame, "find possible markers");
	t1 = cv::getTickCount();

	
	markers = filterMarkersByPerimiter(markers);
	
	drawUsedTime(t1, frame, "filter by per");
	t1 = cv::getTickCount();
	
	markers = filterMarkersByHammingCode(grey, markers);
	
	drawUsedTime(t1, frame, "filter by hamm");
	t1 = cv::getTickCount();
	
	detectPreciseMarkerCorners(grey, markers);
	
	drawUsedTime(t1, frame, "precise");
	t1 = cv::getTickCount();
	
	detectMarkersLocation(grey, markers);
	
	drawUsedTime(t1, frame, "detect locations");

	return markers;
}

Mat MarkersDetector::convertToGrey(Mat frame)
{
	Mat grey;
	cv::cvtColor(frame, grey, CV_BGRA2GRAY);
	return grey;
}

Mat MarkersDetector::performThreshold(Mat img)
{
	Mat binaryImg;
	//adaptiveThreshold(img, binaryImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);
	int method = THRESH_BINARY_INV | CV_THRESH_OTSU;
	if (threshold > 0) {
		method = THRESH_BINARY_INV;
	}
	cv::threshold(img, binaryImg, threshold, 255, method);

	//imshow("binary", binaryImg);

	return binaryImg;
}

std::vector<std::vector<cv::Point>> MarkersDetector::findContours(Mat binaryImg)
{
	std::vector<std::vector<cv::Point>> contours;
	int minContourPointsAllowed = 120;
	std::vector< std::vector<cv::Point> > allContours;
	cv::findContours(binaryImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	contours.clear();
	for (size_t i = 0; i<allContours.size(); i++)
	{
		int contourSize = allContours[i].size();
		if (contourSize > minContourPointsAllowed)
		{
			contours.push_back(allContours[i]);
		}
	}

	return contours;
}

std::vector<Marker> MarkersDetector::findPossibleMarkers(std::vector<std::vector<cv::Point>> contours)
{
	std::vector<Marker> possibleMarkers;

	std::vector<cv::Point> approxCurve;	

	// For each contour, analyze if it is a parallelepiped likely tobe the marker
	for (size_t i = 0; i<contours.size(); i++)
	{
		// Approximate to a polygon
		double eps = contours[i].size() * 0.05;
		cv::approxPolyDP(contours[i], approxCurve, eps, true);
		// We interested only in polygons that contains only fourpoints
		if (approxCurve.size() != 4)
			continue;
		// And they have to be convex
		if (!cv::isContourConvex(approxCurve))
			continue;
		// Ensure that the distance between consecutive points islarge enough
		float minDist = std::numeric_limits<float>::max();
		for (int i = 0; i < 4; i++)
		{
			cv::Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredSideLength = side.dot(side);
			minDist = std::min(minDist, squaredSideLength);
		}
		// Check that distance is not very small
		if (minDist < m_minContourLengthAllowed)
			continue;
		// All tests are passed. Save marker candidate:
		Marker m;
		for (int i = 0; i<4; i++)
			m.points.push_back(cv::Point2f(approxCurve[i].x, approxCurve[i].y));
		// Sort the points in anti-clockwise order
		// Trace a line between the first and second point.
		// If the third point is at the right side, then the points are anticlockwise
		cv::Point v1 = m.points[1] - m.points[0];
		cv::Point v2 = m.points[2] - m.points[0];
		double o = (v1.x * v2.y) - (v1.y * v2.x);
		if (o < 0.0) //if the third point is in the left side, then sort in anti - clockwise order
			std::swap(m.points[1], m.points[3]);

		possibleMarkers.push_back(m);
	}

	return possibleMarkers;
}

std::vector<Marker> MarkersDetector::filterMarkersByPerimiter(std::vector<Marker> possibleMarkers)
{
	std::vector<Marker> detectedMarkers;

	// Remove these elements which corners are too close to eachother.
	// First detect candidates for removal:
	std::vector< std::pair<size_t, size_t> > tooNearCandidates;
	for (size_t i = 0; i<possibleMarkers.size(); i++)
	{
		const Marker& m1 = possibleMarkers[i];
		//calculate the average distance of each corner to the nearest corner of the other marker candidate
		for (size_t j = i + 1; j<possibleMarkers.size(); j++)
		{
			const Marker& m2 = possibleMarkers[j];
			float distSquared = 0;
			for (size_t c = 0; c < 4; c++)
			{
				cv::Point v = m1.points[c] - m2.points[c];
				distSquared += v.dot(v);
			}
			distSquared /= 4;
			if (distSquared < 100)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}
	// Mark for removal the element of the pair with smaller perimeter
	std::vector<bool> removalMask(possibleMarkers.size(), false);
	for (size_t i = 0; i<tooNearCandidates.size(); i++)
	{
		float p1 = possibleMarkers[tooNearCandidates[i].first].getPerimeter();
		float p2 = possibleMarkers[tooNearCandidates[i].second].getPerimeter();
		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;
		removalMask[removalIndex] = true;
	}
	// Return candidates
	detectedMarkers.clear();
	for (size_t i = 0; i<possibleMarkers.size(); i++)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);
	}

	return detectedMarkers;
}

int MarkersDetector::getHammingError(Mat bitMatrix)
{
	int errors = 0;


	for (int row = 0; row < bitMatrix.rows; ++row) {
		int i1 = bitMatrix.at<uchar>(row, 0);
		int i2 = bitMatrix.at<uchar>(row, 1);
		int i3 = bitMatrix.at<uchar>(row, 2);
		int i4 = bitMatrix.at<uchar>(row, 3);
		int i5 = bitMatrix.at<uchar>(row, 4);

		int b1, b2, b4;

		if ((i3 + i5) % 2 == 0) {
			b1 = 1;
		}
		else {
			b1 = 0;
		}

		if (i3 == 1) {
			b2 = 0;
		}
		else {
			b2 = 1;
		}

		if (i5 == 1) {
			b4 = 0;
		}
		else {
			b4 = 1;
		}

		if (b1 != i1) {
			errors++;
		}
		if (b2 != i2) {
			errors++;
		}
		if (b4 != i4) {
			errors++;
		}
	}

	return errors;
}

Mat MarkersDetector::rotate90(Mat mat)
{
	Mat mat2;
	transpose(mat, mat2);
	flip(mat2, mat2, 1);
	return mat2;
}

int MarkersDetector::getHammingId(Mat bitMatrix)
{
	
	string idStr = "";
	for (int row = 0; row < bitMatrix.rows; ++row) {
		int i3 = bitMatrix.at<uchar>(row, 2);
		int i5 = bitMatrix.at<uchar>(row, 4);

		idStr += intToStr(i3) + intToStr(i5);
	}
	return strToInt(idStr);
}

std::vector<Marker> MarkersDetector::filterMarkersByHammingCode(Mat imgGrey, std::vector<Marker> possibleMarkers)
{
	std::vector<Marker> goodMarkers;
	
	
	for (size_t i = 0; i < possibleMarkers.size(); i++)
	{
		cv::Mat canonicalMarker;
		Marker& marker = possibleMarkers[i];
		// Find the perspective transfomation that brings current marker to rectangular form
		cv::Mat M = cv::getPerspectiveTransform(marker.points, m_markerCorners2d);
		

		// Transform image to get a canonical marker image
		cv::warpPerspective(imgGrey, canonicalMarker, M, m_markerSize);

		cv::threshold(canonicalMarker, canonicalMarker, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		cv::Rect r(m_markerCellSize, m_markerCellSize, m_markerSize.width - 2 * m_markerCellSize, m_markerSize.height - 2 * m_markerCellSize);
		cv::Mat subView = canonicalMarker(r);


		cv::Mat bitMatrix = cv::Mat::zeros(5, 5, CV_8UC1);
		//get information(for each inner square, determine if it is black or white)
		for (int y = 0; y<5; y++)
		{
			for (int x = 0; x<5; x++)
			{
				int cellX = (x)*m_markerCellSize;
				int cellY = (y)*m_markerCellSize;
				cv::Mat cell = subView(cv::Rect(cellX, cellY, m_markerCellSize, m_markerCellSize));
				int nZ = cv::countNonZero(cell);
				if (nZ>(m_markerCellSize*m_markerCellSize) / 2)
					bitMatrix.at<uchar>(y, x) = 1;
			}
		}


		//check all possible rotations
		cv::Mat rotations[4];
		int distances[4];
		rotations[0] = bitMatrix;
		distances[0] = getHammingError(rotations[0]);
		std::pair<int, int> minDist(distances[0], 0);
		for (int i = 1; i<4; i++)
		{
			//get the hamming distance to the nearest possible word
			rotations[i] = rotate90(rotations[i - 1]);
			distances[i] = getHammingError(rotations[i]);
			if (distances[i] < minDist.first)
			{
				minDist.first = distances[i];
				minDist.second = i;
			}
		}

		//sort the points so that they are always in the same order
		// no matter the camera orientation
		int nRotations = minDist.second;
		std::rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());

		//get id
		marker.id = getHammingId(rotations[minDist.second]);

		if (minDist.first == 0) {
			goodMarkers.push_back(marker);
		}


	}

	return goodMarkers;
}

void MarkersDetector::detectPreciseMarkerCorners(Mat imgGrey, std::vector<Marker> &markers)
{
	std::vector<cv::Point2f> preciseCorners(4 * markers.size());
	for (size_t i = 0; i < markers.size(); i++)
	{
		Marker& marker = markers[i];
		for (int c = 0; c < 4; c++)
		{
			preciseCorners[i * 4 + c] = marker.points[c];
		}
	}
	if (preciseCorners.size() >= 4) {

		cv::cornerSubPix(imgGrey, preciseCorners, cvSize(5, 5), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER, 30, 0.1));
		//copy back
		for (size_t i = 0; i<markers.size(); i++)
		{
			Marker &marker = markers[i];
			for (int c = 0; c<4; c++)
			{
				marker.points[c] = preciseCorners[i * 4 + c];
			}

		}
	}
}

void MarkersDetector::detectMarkersLocation(Mat imgGrey, std::vector<Marker> &markers)
{
	for (size_t i = 0; i < markers.size(); i++)
	{
		Marker& m = markers[i];
		cv::solvePnP(m_markerCorners3d, m.points, m_cameraMatrix, m_cameraDistortion, m.rotationVector, m.translationVector);
	}
}

void MarkersDetector::drawMarkers(Mat& image, std::vector<Marker> markers)
{
	for (size_t i = 0; i < markers.size(); i++)
	{
		Marker& m = markers[i];

		line(image, m.points[0], m.points[1], Scalar(255, 0, 255));
		line(image, m.points[1], m.points[2], Scalar(255, 0, 255));
		line(image, m.points[2], m.points[3], Scalar(255, 0, 255));
		line(image, m.points[3], m.points[0], Scalar(255, 0, 255));

		putText(image, "id:" + intToStr(m.id), m.points[2], CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1, 2);
	}
}

void MarkersDetector::calculateCameraPose(std::vector<Marker> markers, map <int, cv::Point3f> markersLocations, cv::Point3f& camLocation, cv::Point3f& camRotation, int& usedMarkers)
{
	std::vector<cv::Point3f> cameraLocations;
	std::vector<cv::Point3f> cameraRotations;

	for (size_t i = 0; i < markers.size(); i++)
	{
		Marker& m = markers[i];

		if (markersLocations.find(m.id) == markersLocations.end()) {
			continue;
		}
		cv::Mat_<float> mLocMat = cv::Mat_<float>(markersLocations.at(m.id));

		cv::Mat R;
		cv::Rodrigues(m.rotationVector, R);
		cv::Mat cameraRotationVector;
		cv::Rodrigues(R.t(), cameraRotationVector);
		cv::Mat_<float> cameraTranslationVector = -R.t() * m.translationVector;

		Mat camLocationMat = mLocMat + cameraTranslationVector;
		Point3f camLocation(camLocationMat);

		cameraLocations.push_back(camLocation);
		cameraRotations.push_back(cv::Point3f(m.rotationVector));
	}

	usedMarkers = cameraLocations.size();

	if (usedMarkers > 0) {
		//calculate mean of points
		
		cv::Point3f zero(0.0f, 0.0f, 0.0f);
		cv::Point3f sumLoc = std::accumulate(cameraLocations.begin(), cameraLocations.end(), zero);
		Point3f meanLoc(sumLoc * (1.0f / cameraLocations.size()));
		camLocation = meanLoc;

		cv::Point3f sumRot = std::accumulate(cameraRotations.begin(), cameraRotations.end(), zero);
		Point3f meanRot(sumRot * (1.0f / cameraRotations.size()));
		camRotation = meanRot;
	}
}

void MarkersDetector::getCameraPoseByImage(Mat& frame, cv::Point3f& camLocation, cv::Point3f& camRotation, int& usedMarkers)
{
	std::vector<Marker> markers = detectMarkers(frame);

	drawMarkers(frame, markers);
	calculateCameraPose(markers, m_markersLocations, camLocation, camRotation, usedMarkers);

	if (usedMarkers > 0) {
		string s = "x=" + intToStr(camLocation.x) + ", y=" + intToStr(camLocation.y) + ", z=" + intToStr(camLocation.z) + "; used markers: " + intToStr(usedMarkers);
		putText(frame, s, Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1, 2);

		string s2 = "r1=" + intToStr(camRotation.x) + ", r2=" + intToStr(camRotation.y) + ", r3=" + intToStr(camRotation.z);
		putText(frame, s2, Point(20, 50), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 255), 1, 2);
	}
}

bool MarkersDetector::getFirstMarkerPose(FrameData& frameData, std::array<float, 3>& translation, std::array<float, 3>& rotation, int& markersFound)
{

    markersFound = 0;

    debugLastYPos = 80;
	double t1 = cv::getTickCount();
	
	double firstT = cv::getTickCount();
	
	Mat frame = getFrame();
	if (!frame.data) {
		return false;
	}
	
	drawUsedTime(t1, frame, "get frame");
	t1 = cv::getTickCount();

	std::vector<Marker> markers = detectMarkers(frame);
    
    drawUsedTime(t1, frame, "total detect markers");

	drawMarkers(frame, markers);

	if (markers.size()) {
	    markersFound = markers.size();
		Marker fm = markers.at(0);

		cv::Point3f tv = cv::Point3f(fm.translationVector);
		cv::Point3f rv = cv::Point3f(fm.rotationVector);

		translation[0] = tv.x;
		translation[1] = tv.y;
		translation[2] = tv.z;

		rotation[0] = rv.x;
		rotation[1] = rv.y;
		rotation[2] = rv.z;

		string s = "DIST=" + intToStr(cv::norm(tv)) + "; x=" + intToStr(tv.x) + ", y=" + intToStr(tv.y) + ", z=" + intToStr(tv.z);
		putText(frame, s, Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1, 2);

		string s2 = "r1=" + intToStr(rv.x) + ", r2=" + intToStr(rv.y) + ", r3=" + intToStr(rv.z);
		putText(frame, s2, Point(20, 50), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 255), 1, 2);
	}
	
	drawUsedTime(firstT, frame, "total update marker pose");

	(frameData.buffer).assign(frame.datastart, frame.dataend);
	frameData.width = frame.cols;
	frameData.height = frame.rows;
	
	return true;
}

bool MarkersDetector::captureCamera(int cameraId, int width, int height)
{
	#ifdef __ANDROID__
	    setPreviewSize(width, height);
		return true;
	#endif

	stream = new cv::VideoCapture();
	stream->open(cameraId);
	stream->set(CV_CAP_PROP_FRAME_WIDTH, width);
	stream->set(CV_CAP_PROP_FRAME_HEIGHT, height);
	
	m_isOpen = stream->isOpened();
	return m_isOpen;
}

bool MarkersDetector::captureCameraAuto(int cameraId)
{
	#ifdef __ANDROID__
		return true;
	#endif

	if (m_isOpen) {
		return m_isOpen;
	}

	stream = new cv::VideoCapture();
	stream->open(cameraId);


	m_isOpen = stream->isOpened();
	return m_isOpen;
}

void MarkersDetector::releaseCamera()
{
	if (stream && stream->isOpened()) {
		stream->release();
		delete stream;
	}
    
	m_isOpen = false;
}

Mat MarkersDetector::getFrame()
{

	Mat f;
	
	
	#ifndef __ANDROID__
	    if (m_isOpen) {
		    stream->read(f);
	    }
    #else
        frame_mutex.lock();
		f = MarkersDetector::androidFrame.clone();
		frame_mutex.unlock();
	#endif

	return f;
}

void MarkersDetector::update(std::vector<uchar>& buffer, std::array<float, 3>& camLocation, std::array<float, 3>& camRotation, int& usedMarkers)
{

	Mat frame = getFrame();
	if (!frame.data) {
		return;
	}

	cv::Point3f cl;
	cv::Point3f cr;

	getCameraPoseByImage(frame, cl, cr, usedMarkers);

	(buffer).assign(frame.datastart, frame.dataend);

	camLocation[0] = cl.x;
	camLocation[1] = cl.y;
	camLocation[2] = cl.z;

	camRotation[0] = cr.x;
	camRotation[1] = cr.y;
	camRotation[2] = cr.z;

}

void MarkersDetector::update(unsigned char* buffer, std::array<float, 3>& camLocation, std::array<float, 3>& camRotation, int& usedMarkers)
{

	Mat frame = getFrame();
	if (!frame.data) {
		return;
	}

	cv::Point3f cl;
	cv::Point3f cr;

	getCameraPoseByImage(frame, cl, cr, usedMarkers);

	buffer = (unsigned char*) frame.data;

	camLocation[0] = cl.x;
	camLocation[1] = cl.y;
	camLocation[2] = cl.z;

	camRotation[0] = cr.x;
	camRotation[1] = cr.y;
	camRotation[2] = cr.z;

}


void MarkersDetector::updateCameraPose(std::array<float, 3>& camLocation, std::array<float, 3>& camRotation, int& usedMarkers)
{
	Mat frame = getFrame();
	if (!frame.data) {
		return;
	}

	cv::Point3f cl;
	cv::Point3f cr;

	getCameraPoseByImage(frame, cl, cr, usedMarkers);

	camLocation[0] = cl.x;
	camLocation[1] = cl.y;
	camLocation[2] = cl.z;

	camRotation[0] = cr.x;
	camRotation[1] = cr.y;
	camRotation[2] = cr.z;
}

void MarkersDetector::updateCameraPoseWithCallback()
{
#ifdef __ANDROID__
	ALOGI("update1");
#endif

	Mat frame = getFrame();
	if (!frame.data) {
		#ifdef __ANDROID__
			ALOGW("Empty frame");
		#endif
		return;
	}

	cv::Point3f cl;
	cv::Point3f cr;
	int usedMarkers;

	getCameraPoseByImage(frame, cl, cr, usedMarkers);

	std::array<float, 3> camLocation;
	std::array<float, 3> camRotation;

	camLocation[0] = cl.x;
	camLocation[1] = cl.y;
	camLocation[2] = cl.z;

	camRotation[0] = cr.x;
	camRotation[1] = cr.y;
	camRotation[2] = cr.z;



	if (afterPoseUpdateCallback) {
#ifdef __ANDROID__
		ALOGI("update call callback");
#endif
		afterPoseUpdateCallback(camLocation, camRotation, usedMarkers);
	}
}




#ifdef __ANDROID__

extern "C"
void
Java_org_getid_markersdetector_AndroidCamera_InitJNI(JNIEnv* env, jobject thiz)
{
    AndroidCameraObject = thiz;
}

extern "C"
jboolean
Java_org_getid_markersdetector_AndroidCamera_FrameProcessing(
JNIEnv* env, jobject thiz,
jint width, jint height,
jbyteArray NV21FrameData)
{
    if (MarkersDetector::beforeFrameUpdateCallback) {
		MarkersDetector::beforeFrameUpdateCallback();
	}
	
	
	jbyte* pNV21FrameData = env->GetByteArrayElements(NV21FrameData, NULL);
 
    
    //int size = height * width * 3 / 2; //12 bits per pixel
    //unsigned char* data[size];
	//memcpy(data, pNV21FrameData, size);
	//MarkersDetector::androidFrame = cv::Mat(height, width, CV_8UC4, data);
	
	
	cv::Mat yuv(height + height/2, width, CV_8UC1, (uchar*)pNV21FrameData);
    cv::Mat bgr(height, width, CV_8UC4);
    cv::cvtColor(yuv, bgr, CV_YUV2BGR_NV21);
    
    frame_mutex.lock();
	MarkersDetector::androidFrame = bgr;
	frame_mutex.unlock();
	
		
	if (MarkersDetector::afterFrameUpdateCallback) {
		MarkersDetector::afterFrameUpdateCallback();
	}

	env->ReleaseByteArrayElements(NV21FrameData, pNV21FrameData, JNI_ABORT);
	
	
	return true;
}



#endif



