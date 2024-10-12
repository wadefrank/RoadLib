/*******************************************************
 * Copyright (C) 2024, GREAT Group, Wuhan University
 * 
 * This file is part of RoadLib.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Yuxuan Zhou (yuxuanzhou@whu.edu.cn)
 *******************************************************/
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include<Eigen/StdVector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/flann.hpp>

#include <iostream>
#include <filesystem>
#include <chrono>
#include <unordered_map>
#include <iomanip>
#include <set>
#include <random>
#include "gv_utils.h"
#include "ipm_processer.h"
#include "utils.hpp"
#include "gviewer.h"

using namespace Eigen;
using namespace std;

/// @brief SensorConfig结构体，用于存储传感器的配置信息，例如相机配置、平滑处理的参数、patch（补丁）相关参数和建图参数等
struct SensorConfig
{
public:
	SensorConfig(string path);
	SensorConfig() {};
public:
	gv::CameraConfig cam;								// 用于存储相机配置信息
	
	int pose_smooth_window = 20;						// 平滑窗口大小，用于姿态平滑
	bool need_smooth = true;							// 是否需要进行平滑处理
	double large_slope_thresold = 1.5;					// 大斜率阈值		
	double t_start;										// 起始时间
	double t_end;										// 结束时间

	int patch_min_size = 50;							// patch最小尺寸
	double patch_dashed_min_h = 1.35;					// 虚线patch最小高度
	double patch_dashed_max_h = 10.0;					// 虚线patch最大高度
	double patch_dashed_max_dist = 12.0;				// 虚线patch最大距离
	double patch_guide_min_h = 0.0;						// 引导线patch最小高度
	double patch_guide_max_h = 1000.0;					// 引导线patch最大高度
	double patch_guide_max_dist = 20.0;					// 引导线patch最大距离
	double patch_solid_max_dist = 15.0;					// 实线patch最大距离
	double patch_stop_max_dist = 12.0;					// 停止线patch最大距离

	int mapping_step = 10;								// 建图步长
	double mapping_patch_freeze_distance = 10.0;		// patch冻结距离
	double mapping_line_freeze_distance = 10.0;			// 线冻结距离
	double mapping_line_freeze_max_length = 50.0;		// 线冻结最大长度
	double mapping_line_cluster_max_dist = 1.0;			// 线簇最大距离
	double mapping_line_cluster_max_across_dist1 = 1.0;	// 线簇最大横向距离1
	double mapping_line_cluster_max_across_dist2 = 0.4;	// 线簇最大横向距离2
	double mapping_line_cluster_max_theta = 10.0;		// 线簇最大角度

	int localization_max_windowsize = 100;				// 定位最大窗口大小
	int localization_force_last_n_frames = 2;			// 定位强制使用最后n帧
	int localization_every_n_frames = 5;				// 定位每隔n帧执行一次
	int localization_min_keyframe_dist = 1.0;			// 定位最小关键帧距离
	int localization_max_strict_match_dist = 1.0;		// 定位最大严格匹配距离
	int localization_solid_sample_interval = 3.0;		// 定位实线采样间隔

	bool enable_vis_image = true;						// 是否可视化图像
	bool enable_vis_3d = true;							// 是否可视化3d
};

extern gviewer viewer;
extern vector<VisualizedInstance> vis_instances;
extern std::normal_distribution<double> noise_distribution;
extern std::default_random_engine random_engine;

enum PatchType { EMPTY = -1, SOLID = 0, DASHED = 1, GUIDE = 2, ZEBRA = 3, STOP = 4 };

/// @brief 将灰度值转换为patch类型
/// @param gray 灰度值
/// @return patch类型
inline PatchType gray2class(int gray)
{
	if (gray == 2) return PatchType::DASHED;
	else if (gray == 3) return PatchType::GUIDE;
	else if (gray == 4) return PatchType::ZEBRA;
	else if (gray == 5) return PatchType::STOP;
	else if (gray > 0) return PatchType::SOLID;
	else return PatchType::EMPTY;
}

/// @brief 将patch类型转换为字符串
/// @param PatchType patch类型
/// @return patch类型字符串
inline string PatchType2str(PatchType PatchType)
{
	if (PatchType == PatchType::DASHED) return "dashed";
	else if (PatchType == PatchType::GUIDE) return "guide";
	else if (PatchType == PatchType::ZEBRA) return "zebra";
	else if (PatchType == PatchType::SOLID) return "solid";
	else if (PatchType == PatchType::STOP) return "stop";
	else if (PatchType == PatchType::EMPTY) return "empty";
	else return "unknown";
}

/// @brief RoadInstancePatch类表示道路patch的实例，每个patch包含多个属性，例如ID、类型、关联帧ID、边界框的四个点和不确定性距离、线相关参数、原始点的坐标等。这个类的主要作用是存储和管理每个检测到的道路特征，并提供计算边界框高度、宽度和方向的方法。
class RoadInstancePatch
{
public:
	static long long next_id;				// 生成下一个实例的id
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
	//** ID/Class
	long long id;							// 实例的唯一标识符
	PatchType road_class;					// 表示道路类型的枚举变量
	map<PatchType, int> road_class_count;	// 存储不同道路类型数量的映射

	//** Flags
	bool line_valid;
	bool merged = false;
	bool frozen = false;
	bool valid_add_to_map = false;
	bool out_range = false;

	long long frame_id;						// 关联的帧id

	// Associated with RoadInstancePatchMap::timestamps and queued_poses. 关联RoadInstancePatchMap::timestamps和queued_poses
	// For patch-like instances -> linked_frames[0] 对于patch实例->linked_frames[0]
	// For line-like instances  -> linked_frames[i], i = 0, ..., line_points_metric.size() 对于线实例->linked_frames[i], i = 0, ..., line_points_metric.size()
	vector<vector<long long>> linked_frames; // 存储与该实例相关的帧id的向量

	//** Bounding box related
	// Main parameters for patch-like instances.
	// 
	//    p3----p2
	//    |      |
	//    |      |
	//    p0 --- p1
	//
	Eigen::Vector3d b_point[4];        // Image frame. 用于表示边界框的四个点，图像帧中的坐标
	Eigen::Vector3d b_point_metric[4]; // Body/map frame. 用于表示边界框的四个点，在地图帧中的坐标
	double b_unc_dist[4];              // Distance uncertainty. 边界框四个点的距离不确定性

	//** Line related
	// Main parameters for line-like instances.
	Eigen::VectorXd line_koef;																		// 用于表示线的参数
	vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> line_points;					// 线的点集，在图像帧中的坐标，aligned_allocator是为了保证内存对齐
	vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> line_points_metric;			// 线的点集，在地图帧中的坐标
	vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> line_points_uncertainty;		// 线点的不确定性矩阵

	//** Raw points related (image frame and metric frame)
	// General attributes.
	// 1) Image frame
	double top, left, width, height;											// 边界框的位置和尺寸
	Eigen::Vector3d mean;
	Eigen::Matrix3d cov;
	Eigen::Vector3d direction;
	double eigen_value[3];
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;			// 原始点的坐标

	// 2) Body/map frame
	Eigen::Vector3d mean_metric;												// 在地图帧中的均值
	vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points_metric;	// 原始点在地图帧中的坐标
	Eigen::Matrix3d mean_uncertainty;											// 平均不确定性
	double percept_distance = 10000;											// 感知距离


public:
	RoadInstancePatch()
	{
		id = next_id++;
		for (int ii = 0; ii < 4; ii++)
		{
			b_point[ii].setZero();
			b_point_metric[ii].setZero();
		}
	}

	// Height of bounding box.
	double h() const;

	// Width of bounding box.
	double w() const;

	// Direction of bounding box/line.
	Eigen::Vector3d d() const;

};

/// @brief RoadInstancePatchFrame类对应每一帧中的所有道路信息。它包含帧的唯一ID、时间戳、旋转矩阵和平移向量，以及存储不同类型patch的映射。这个类的主要功能是将图像帧中的道路特征转换为地图帧中的特征。
class RoadInstancePatchFrame
{
public:
	// 类的静态成员变量
	//   1.属于类本身的，而不是属于某个对象的。
	//   2.静态成员变量在类的所有对象之间共享，并且它的生命周期贯穿整个程序的运行过程。
	//   3.静态成员变量的声明使用 static 关键字，它只声明一次，且只在类的作用域内存在。
	//   4.静态成员变量的初始化不能在类的定义中进行，而是要在类的外部进行初始化。
	//   5.静态成员变量的初始化也必须在类定义之外的全局作用域中完成。
	static long long next_id;											// 生成每个RoadInstancePatchFrame对象的唯一id
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
	long long id;														// 每个RoadInstancePatchFrame对象的唯一id
	double time;														// 帧的时间戳
	Eigen::Matrix3d R; // R^world_body									// 旋转矩阵，表示世界坐标系到车体坐标系的变换
	Eigen::Vector3d t; // t^world_body									// 平移向量，表示世界坐标系到车体坐标系的变换
	map<PatchType, vector<shared_ptr<RoadInstancePatch>>> patches;		// 存储不同类型patch的映射，每种类型对应一个patch对象的共享指针向量
public:
	RoadInstancePatchFrame()
	{
		id = next_id++;	// 生成唯一id
	}

	// Calculate metric-scale properties of the patches.
	// Image frame -> body frame.
	int generateMetricPatches(const SensorConfig &config, const gv::IPMProcesser &ipm);
};

/// @brief RoadInstancePatchMap类用于将每一帧的信息存储并归纳到地图中。它包含存储不同类型patch的映射、参考向量、队列中的姿势、时间戳等。这个类提供了多种方法，例如添加帧、合并patch、清理地图、保存和加载地图、构建KD树进行地图匹配等。
class RoadInstancePatchMap
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	map<PatchType, vector<shared_ptr<RoadInstancePatch>>> patches;	// 存储不同类型patch的映射，每种类型对应一个patch对象的共享指针向量。
	Eigen::Vector3d ref;											// 参考向量

	map<long long, pair<Matrix3d, Vector3d>> queued_poses;			// 队列中的姿势
	map<long long, double> timestamps;
public:
	int addFrame(const RoadInstancePatchFrame & frame);

	// Merge patches in the current map.
	// mode 0 : incremental mapping; mode 1: map merging/map checking
	// For mode 0, Rwv and twv are used to determine active instances and freeze old instances.
	// For mode 1, Rwv and twv are useless. Just use dummy values. (To be improved
	int mergePatches(const SensorConfig &config, const int mode, const Eigen::Matrix3d &Rwv = Eigen::Matrix3d::Identity(), const Eigen::Vector3d & twv = Eigen::Vector3d::Zero());
	
	// Simply stacking the patches from two maps.
	// The patches wouldn't be merged.
	int mergeMap(const RoadInstancePatchMap& road_map_other);

	// Clear the map.
	int clearMap();

	// Unfreeze the pathes (for further merging).
	int unfreeze();
	
	// Integrity checking.
	int cleanMap();

	// Save/load functions.
	// Notice that only metric-scale properties are saved.
	int saveMapToFileBinaryRaw(string filename);
	int loadMapFromFileBinaryRaw(string filename);

	// Build KDtree for map matching.
	int buildKDTree();

	// Instance-level nearest matching.
	map<PatchType, vector<pair<int, int>>> mapMatch(const SensorConfig &config, 
		RoadInstancePatchFrame &frame, int mode = 0); // mode : 0(normal), 1(strict)
	
	// Line segment matching (for measurement construction).
	vector<pair<int, int>> getLineMatch(const SensorConfig &config, RoadInstancePatchFrame &frame, PatchType road_class,
		int frame_line_count, int map_line_count, int mode =0);

	// Geo-register the map elements based on linked frames.
	// Function 'mergePatches' should be called later for consistency.
	int geoRegister(const Trajectory& new_traj,
		vector<VisualizedInstance>& lines_vis);

public:
	map<long long, double> ignore_frame_ids; // frame_id - distance threshold
private:

};


/**
 * @brief					Merge the line instance cluster.
 * @param lines_in			The raw line cluser.
 * @param line_est			The merged line instance.
 *
 * @return					Success flag.
 */
extern int LineCluster2SingleLine(const PatchType road_class, const vector<shared_ptr<RoadInstancePatch>>& lines_in, shared_ptr<RoadInstancePatch>& line_est, Eigen::Matrix3d Rwv = Eigen::Matrix3d::Identity());

/**
 * @brief					Use the semantic IPM image to generate the road marking instances.
 * @param config			The raw line cluser.
 * @param ipm				IPM processor with up-to-date camera-ground parameters.
 * @param ipm_raw			RGB IPM image.
 * @param ipm_class			Semantic IPM image. (label = 0,1,2,3,4,5, other)
 *
 * @return					Success flag.
 */
extern RoadInstancePatchFrame generateInstancePatch(const SensorConfig& config, const gv::IPMProcesser& ipm, const cv::Mat& ipm_raw, const cv::Mat& ipm_class);

/**
 * @brief					Calculate the uncertainty of the element on the IPM image based on the pixel coordinates (uI, vI).
 * @param uI, vI			Pixel coordinates on the IPM image.
 *
 * @return					Uncertainty matrix.
 */
extern Matrix3d calcUncertainty(const gv::IPMProcesser& ipm, double uI, double vI);

extern int PointCloud2Curve2D(const vector<Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points, int dim, VectorXd& K);

extern int LabelClustering(const cv::Mat& ipm_class, cv::Mat& ipm_label, cv::Mat& stats, cv::Mat& centroids);