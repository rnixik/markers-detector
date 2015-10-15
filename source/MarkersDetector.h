#ifdef MARKERSDETECTOR_EXPORTS
#define MARKERSDETECTOR_API __declspec(dllexport) 
#else
  #ifndef __ANDROID__
    #define MARKERSDETECTOR_API __declspec(dllimport)
  #else
    #define MARKERSDETECTOR_API 
  #endif
#endif

#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"

#include <vector>
#include <array>
#include <map>

#ifdef __ANDROID__
    #include <jni.h>
    #include <errno.h>
    #include <android/log.h>

    #define ALOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "MarkersDetector", __VA_ARGS__))
    #define ALOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "MarkersDetector", __VA_ARGS__))
#endif

using namespace cv;
using namespace std;

class Marker{

public:
	std::vector< cv::Point2f > points;
	int id;
	Mat translationVector;
	Mat rotationVector;

	float getPerimeter();

private:


};

struct FrameData
{
    std::vector<uchar> buffer;
    int width;
    int height;
};

typedef void(*AfterFrameUpdateCallback)();
typedef void(*BeforeFrameUpdateCallback)();
typedef void(*AfterPoseUpdateCallback)(std::array<float, 3> camLocation, std::array<float, 3> camRotation, int usedMarkers);

class MARKERSDETECTOR_API MarkersDetector{

public:
	MarkersDetector(std::array<double, 9> cameraMatrixBuf, std::array<double, 8> cameraDistortionBuf, int printedMarkerWidth);

    void setMarkersParams(map <int, std::array<float, 3>>* markersLocations);

	int threshold = 0;

	bool captureCamera(int cameraId, int width, int height);
	bool captureCameraAuto(int cameraId);
	void releaseCamera();

	std::vector<Marker> detectMarkers(Mat frame);
	void drawMarkers(Mat& image, std::vector<Marker> markers);
	void calculateCameraPose(std::vector<Marker> markers, map <int, cv::Point3f> markersLocations, cv::Point3f& camLocation, cv::Point3f& camRotation, int& usedMarkers);


	void getCameraPoseByImage(Mat& frame, cv::Point3f& camLocation, cv::Point3f& camRotation, int& usedMarkers);

	void update(std::vector<uchar>& buffer, std::array<float, 3>& camLocation, std::array<float, 3>& camRotation, int& usedMarkers);

	void update(unsigned char* buffer, std::array<float, 3>& camLocation, std::array<float, 3>& camRotation, int& usedMarkers);

	void updateCameraPose(std::array<float, 3>& camLocation, std::array<float, 3>& camRotation, int& usedMarkers);

	bool getFirstMarkerPose(FrameData& frameData, std::array<float, 3>& translation, std::array<float, 3>& rotation, int& markersFound);
	
	static cv::Mat androidFrame;

	static AfterFrameUpdateCallback afterFrameUpdateCallback;
	static BeforeFrameUpdateCallback beforeFrameUpdateCallback;

	void updateCameraPoseWithCallback();
	AfterPoseUpdateCallback afterPoseUpdateCallback;
	
	Mat getFrame();

private:

	map <int, cv::Point3f> m_markersLocations;
	Mat_<double> m_cameraMatrix;
	Mat_<double> m_cameraDistortion;
	int m_minContourLengthAllowed = 1000;

	VideoCapture* stream;
	bool m_isOpen = false;

	Size m_markerSize;
	int m_markerCellSize;
	std::vector<Point2f> m_markerCorners2d;
	std::vector<Point3f> m_markerCorners3d;

	Mat convertToGrey(Mat frame);
	Mat performThreshold(Mat img);
	
	std::vector<std::vector<cv::Point>> findContours(Mat binaryImg);

	std::vector<Marker> findPossibleMarkers(std::vector<std::vector<cv::Point>> contours);

	std::vector<Marker> filterMarkersByPerimiter(std::vector<Marker> possibleMarkers);

	int getHammingError(Mat bitMatrix);
	Mat rotate90(Mat mat);
	int getHammingId(Mat bitMatrix);

	std::vector<Marker> filterMarkersByHammingCode(Mat imgGrey, std::vector<Marker> possibleMarkers);

	void detectPreciseMarkerCorners(Mat imgGrey, std::vector<Marker>& markers);

	void detectMarkersLocation(Mat imgGrey, std::vector<Marker>& markers);


};


#ifdef __ANDROID__

extern "C" {
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved);
};

extern "C"
jboolean
Java_org_getid_markersdetector_AndroidCamera_FrameProcessing(
        JNIEnv* env, jobject thiz,
        jint width, jint height,
        jbyteArray NV21FrameData);

extern "C"
void
Java_org_getid_markersdetector_AndroidCamera_InitJNI(JNIEnv* env, jobject thiz);

#endif        
