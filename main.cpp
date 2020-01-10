/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <cmath>

#include <networktables/NetworkTableInstance.h>

#include <vision/VisionPipeline.h>
#include <vision/VisionRunner.h>

#include <wpi/StringRef.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>
#include <wpi/raw_ostream.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cameraserver/CameraServer.h"

using namespace std;
using namespace cv;

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
       "switched cameras": [
           {
               "name": <virtual camera name>
               "key": <network table key used for selection>
               // if NT value is a string, it's treated as a name
               // if NT value is a double, it's treated as an integer index
           }
       ]
   }
 */

static const char* configFile = "/boot/frc.json";

namespace {

unsigned int team;
bool server = false;

struct CameraConfig {
  std::string name;
  std::string path;
  wpi::json config;
  wpi::json streamConfig;
};

struct SwitchedCameraConfig {
  std::string name;
  std::string key;
};

std::vector<CameraConfig> cameraConfigs;
std::vector<SwitchedCameraConfig> switchedCameraConfigs;
std::vector<cs::VideoSource> cameras;

wpi::raw_ostream& ParseError() {
  return wpi::errs() << "config error in '" << configFile << "': ";
}

bool ReadCameraConfig(const wpi::json& config) {
  CameraConfig c;

  // name
  try {
    c.name = config.at("name").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "could not read camera name: " << e.what() << '\n';
    return false;
  }

  // path
  try {
    c.path = config.at("path").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "camera '" << c.name
                 << "': could not read path: " << e.what() << '\n';
    return false;
  }

  // stream properties
  if (config.count("stream") != 0) c.streamConfig = config.at("stream");

  c.config = config;

  cameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadSwitchedCameraConfig(const wpi::json& config) {
  SwitchedCameraConfig c;

  // name
  try {
    c.name = config.at("name").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "could not read switched camera name: " << e.what() << '\n';
    return false;
  }

  // key
  try {
    c.key = config.at("key").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "switched camera '" << c.name
                 << "': could not read key: " << e.what() << '\n';
    return false;
  }

  switchedCameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadConfig() {
  // open config file
  std::error_code ec;
  wpi::raw_fd_istream is(configFile, ec);
  if (ec) {
    wpi::errs() << "could not open '" << configFile << "': " << ec.message()
                << '\n';
    return false;
  }

  // parse file
  wpi::json j;
  try {
    j = wpi::json::parse(is);
  } catch (const wpi::json::parse_error& e) {
    ParseError() << "byte " << e.byte << ": " << e.what() << '\n';
    return false;
  }

  // top level must be an object
  if (!j.is_object()) {
    ParseError() << "must be JSON object\n";
    return false;
  }

  // team number
  try {
    team = j.at("team").get<unsigned int>();
  } catch (const wpi::json::exception& e) {
    ParseError() << "could not read team number: " << e.what() << '\n';
    return false;
  }

  // ntmode (optional)
  if (j.count("ntmode") != 0) {
    try {
      auto str = j.at("ntmode").get<std::string>();
      wpi::StringRef s(str);
      if (s.equals_lower("client")) {
        server = false;
      } else if (s.equals_lower("server")) {
        server = true;
      } else {
        ParseError() << "could not understand ntmode value '" << str << "'\n";
      }
    } catch (const wpi::json::exception& e) {
      ParseError() << "could not read ntmode: " << e.what() << '\n';
    }
  }

  // cameras
  try {
    for (auto&& camera : j.at("cameras")) {
      if (!ReadCameraConfig(camera)) return false;
    }
  } catch (const wpi::json::exception& e) {
    ParseError() << "could not read cameras: " << e.what() << '\n';
    return false;
  }

  // switched cameras (optional)
  if (j.count("switched cameras") != 0) {
    try {
      for (auto&& camera : j.at("switched cameras")) {
        if (!ReadSwitchedCameraConfig(camera)) return false;
      }
    } catch (const wpi::json::exception& e) {
      ParseError() << "could not read switched cameras: " << e.what() << '\n';
      return false;
    }
  }

  return true;
}

cs::UsbCamera StartCamera(const CameraConfig& config) {
  wpi::outs() << "Starting camera '" << config.name << "' on " << config.path
              << '\n';
  auto inst = frc::CameraServer::GetInstance();
  cs::UsbCamera camera{config.name, config.path};
  auto server = inst->StartAutomaticCapture(camera);

  camera.SetConfigJson(config.config);
  camera.SetConnectionStrategy(cs::VideoSource::kConnectionKeepOpen);

  if (config.streamConfig.is_object())
    server.SetConfigJson(config.streamConfig);

  return camera;
}

cs::MjpegServer StartSwitchedCamera(const SwitchedCameraConfig& config) {
  wpi::outs() << "Starting switched camera '" << config.name << "' on "
              << config.key << '\n';
  auto server =
      frc::CameraServer::GetInstance()->AddSwitchedCamera(config.name);

  nt::NetworkTableInstance::GetDefault()
      .GetEntry(config.key)
      .AddListener(
          [server](const auto& event) mutable {
            if (event.value->IsDouble()) {
              int i = event.value->GetDouble();
              if (i >= 0 && i < cameras.size()) server.SetSource(cameras[i]);
            } else if (event.value->IsString()) {
              auto str = event.value->GetString();
              for (int i = 0; i < cameraConfigs.size(); ++i) {
                if (str == cameraConfigs[i].name) {
                  server.SetSource(cameras[i]);
                  break;
                }
              }
            }
          },
          NT_NOTIFY_IMMEDIATE | NT_NOTIFY_NEW | NT_NOTIFY_UPDATE);

  return server;
}

/////////////////////// OPENCV Procceing /////////////////////////////////// 

//math constants
const double M_PI = 3.141592653; //PI
const double CONVERT_TO_DEGREES = (M_PI / 180.0); // convert radians to degrees 

//Angles in radians
//image size ratioed to 16:9
int imageWidth = 1280;
int imageHeight = 720;

//Center(x,y) of the image
double centerX = (imageWidth / 2) - .5;
double centerY = (imageHeight / 2) - .5;

//Lifecam 3000 from datasheet
//Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf
//convert degrees '68.5' to radians 
double diagonalView = (68.5 * M_PI) / 180.0;

//16:9 aspect ratio
int horizontalAspect = 16;
int verticalAspect = 9;

//Reasons for using diagonal aspect is to calculate horizontal field of view.
double diagonalAspect = hypot(horizontalAspect, verticalAspect);

//Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
double horizontalView = atan(tan(diagonalView / 2) * (horizontalAspect / diagonalAspect)) * 2;
double verticalView = atan(tan(diagonalView / 2) * (verticalAspect / diagonalAspect)) * 2;

//Focal Length calculations : https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
double H_FOCAL_LENGTH = imageWidth / (2 * tan((horizontalView / 2)));
double V_FOCAL_LENGTH = imageHeight / (2 * tan((verticalView / 2)));

// Tracked objects centroid coordinates
int theObject[2] = { 0,0 };
double yaw = 0, pitch = 0;
//bounding rectangle of the object, we will use the center of this as its position
Rect objectBoundingRectangle = Rect(0, 0, 0, 0);
RotatedRect rotatedRectangle;

int iLowH = 44;
int iHighH = 60;

int iLowS = 0;
int iHighS = 30;

int iLowV = 130;
int iHighV = 255;

// Include center point of your rectangle, size of your rectangle and the degrees of rotation  
//cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.02, true);
void DrawRotatedRectangle(cv::Mat& image, vector< vector<Point>> cnt){
	cv::Scalar color = cv::Scalar(255.0, 0.0, 0.0); // red because it is RGB
	cv::RotatedRect rotatedRectangle;
	rotatedRectangle = cv::fitEllipse(image);

	//We take the edges that OpenCV calculated for us
	cv::Point2f vertices2f[4];
	rotatedRectangle.points(vertices2f);

	//Convert them so we can use them in a fillConvexPoly
	cv::Point vertices[4];
	for (int i = 0; i < 4; ++i) {
		vertices[i] = vertices2f[i];
	}

	//Now we can fill the rotated rectangle with our specified color
	cv::fillConvexPoly(image,
		vertices,
		4,
		color);
}

//Uses trig and focal length of camera to find yaw.
//Link to further explanation : https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
double calculateYaw(double pixelX, double CenterX, double hFocalLength) {
	double yaw = CONVERT_TO_DEGREES * (atan((pixelX - CenterX) / hFocalLength));
	return yaw; 
}

//Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
double calculatePitch(double pixelY, double  CenterY, double vFocalLength) {
	double pitch = CONVERT_TO_DEGREES * (atan((pixelY - CenterY) / vFocalLength));
	//just stopped working have to do this:
	pitch *= -1.0;
	return pitch;
}

double calculateDistance(double heightOfCamera, double heightOfTarget ,double pitch) {
	double heightOfTargetFromCamera = heightOfTarget - heightOfCamera;
	/*
	// Uses trig and pitch to find distance to target
  
    d = distance
    h = height between camera and target
    a = angle = pitch
    tan a = h/d (opposite over adjacent)
    d = h / tan a
                         .
                        /|
                       / |
                      /  |h
                     /a  |
              camera -----
                       d
	*/
	double distance = fabs(heightOfTargetFromCamera / tan((pitch* M_PI) / 180.0));
	return distance;
}

void searchForMovement(Mat thresholdImage, Mat &cameraFeed) {
	/* Notice how we use the '&' operator for the camerafeed. This is because we wish
	 to take the values passed into the function and manipulate them, rather than just
	 working with a copy. eg. we draw to the cameraFeed in this function which is then
	 displayed in the main() function. */
	bool objectDetected = false;
	Mat temp;
	thresholdImage.copyTo(temp);
		// these two vectors needed for the output of findContours

	vector< vector<Point>> contours;
	vector<Vec4i> hierarchy;
		//findContours of filtered image using openCV findCOntours function
		//findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //retrieves all contours
	findContours(temp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);// retrieves external contours
	
		//if contours vector is not empty, we found some objects
	if (contours.size() > 0) {
		objectDetected = true;
	}
	else { objectDetected = false; }

	if (objectDetected) {
		//largest contour is found at the end of the contours vector
		//we will simply assume that the biggest contour is what we looking for.
		vector< vector<Point>> largestContourVec;
		largestContourVec.push_back(contours.at( contours.size() - 1));
		
		//making a bounding rectagle around the largest contour then find its centriod
		//this will be the object's final estimated position.
		objectBoundingRectangle = boundingRect(largestContourVec.at(0));
		int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width / 2;
		int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height / 2;

		//update the objects position by changing the 'theObject' array values
		theObject[0] = xpos, theObject[1] = ypos;

	}
	//make some temp x and y varibles so we don't have to type out so much
	int x = theObject[0];
	int y = theObject[1];

	//Calculates yaw of contour (horizontal position in degrees)
	//
	//Calculates pitch of contour (Vertical position in degrees)
	//

	//draw some crosshairs on the object
	rectangle(cameraFeed, objectBoundingRectangle, Scalar(255, 0, 0), 3, 8, 0);
	circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	//draws the rotated rectangle contours and might have to blur to make drawing more 'rectangleish'
	drawContours(cameraFeed, contours, 0, Scalar(255, 0, 0), 1, LINE_AA);

	// Displays yaw and pitch of tracked object
	//putText(cameraFeed, "Tracking object at (" + x, Point(x, y), 1, 1, Scalar(0, 0, 255), 2, LINE_AA);
	//putText(cameraFeed, "Yaw = " + yaw, Point(x - 30, y), 1, 1, Scalar(0, 0, 255), 2);
	//putText(cameraFeed, "Pitch = " + pitch, Point(x - 30, y + 30), 1, 1, Scalar(0, 0, 255), 2);
	
}

// example pipeline
class MyPipeline : public frc::VisionPipeline {
 public:
  int val = 0;

  void Process(cv::Mat& mat) override {
    ++val;
  }
};
}  // namespace



int main(int argc, char* argv[]) {
  if (argc >= 2) configFile = argv[1];

  // read configuration
  if (!ReadConfig()) return EXIT_FAILURE;

  // start NetworkTables
  auto ntinst = nt::NetworkTableInstance::GetDefault();
  //add the fricking server here bro!!!!!!!!!!!!!!!!!


  if (server) {
    wpi::outs() << "Setting up NetworkTables server\n";
    ntinst.StartServer();
  } else {
    wpi::outs() << "Setting up NetworkTables client for team " << team << '\n';
    ntinst.StartClientTeam(team);
  }

  // start cameras
  for (const auto& config : cameraConfigs)
    cameras.emplace_back(StartCamera(config));

  // start switched cameras
  for (const auto& config : switchedCameraConfigs) StartSwitchedCamera(config);

  //getting camera feed from the first camera
  //cs::CvSink cvsink = cameras[1].GetLastFrameTime();

  // start image processing on camera 0 if present
  if (cameras.size() >= 1) {
    std::thread([&] {
      frc::VisionRunner<MyPipeline> runner(cameras[0], new MyPipeline(),
                                           [&](MyPipeline &pipeline) {
        // do something with pipeline results
      });
      /* something like this for GRIP:
      frc::VisionRunner<MyPipeline> runner(cameras[0], new grip::GripPipeline(),
                                           [&](grip::GripPipeline& pipeline) {
        ...
      });
       */
      runner.RunForever();
    }).detach();
  }

  // loop forever
  for (;;) std::this_thread::sleep_for(std::chrono::seconds(10));
}
