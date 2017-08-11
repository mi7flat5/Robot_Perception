#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
#from pcl_helper import *
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)
    # Statistical Outlier Filtering

    # Create outlier filter
    outlier_filter = pcl_cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(10)

    # Set threshold scale factor
    x = .001

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    filtered = outlier_filter.filter()

    # Voxel Grid Downsampling
    vox = filtered.make_voxel_grid_filter()
    LEAF_SIZE = .008

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # PassThrough Filter for z axis
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'

    passthrough.set_filter_field_name(filter_axis)
    axis_min = .608
    axis_max = 1.5
    passthrough.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough.filter()

    # Passthrough filter for y axis
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis2 = 'y'

    passthrough.set_filter_field_name(filter_axis2)
    axis_min = -0.45
    axis_max = 0.45
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()


    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()

    # Fit to plane
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = .01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()

    ec.set_ClusterTolerance(.05)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(2550)

    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)

    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    # Convert PCL data to ROS messages
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)


    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = extracted_outliers.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        roscloud = pcl_to_ros(pcl_cluster)

        # Extract histogram features and create feature vector
        chists = compute_color_histograms(roscloud, using_hsv=True)
        normals = get_normals(roscloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        features_scaled = scaler.transform(feature.reshape(1, -1))
        prediction = clf.predict(features_scaled)
        label = encoder.inverse_transform(prediction)[0]

        # Add it to detected_objects_labels list
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = roscloud
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)
    try:

        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = 3
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')

    foundObjects = {}
    for object in object_list:
        # Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(object.cloud).to_array()
        foundObjects[object.label] = np.mean(points_arr, axis=0)[:3]

    dbox = {}
    for i in range(len(dropbox_list_param)):
        dbox[dropbox_list_param[i]['group']]= [dropbox_list_param[i]['name'], np.array(dropbox_list_param[i]['position']).tolist()]



    dict_list = []
    # Loop through the pick list
    for i in range(len(object_list_param)):
        name = object_list_param[i]['name']
        group = object_list_param[i]['group']
        if foundObjects.has_key(name):
            # Parse parameters into individual variables
            object_name.data = name
            # Create 'pick_pose' for the object using centroid
            pick_pose.position.x = np.asscalar(foundObjects[name][0])
            pick_pose.position.y = np.asscalar(foundObjects[name][1])
            pick_pose.position.z = np.asscalar(foundObjects[name][2])
            # Assign the arm to be used for pick_place
            arm_name.data = dbox[group][0]
            # Create 'place_pose' for the object TODO: add random variation to avoid stacking
            place_pose.position.x = float(dbox[group][1][0])
            place_pose.position.y = float(dbox[group][1][1])
            place_pose.position.z = float(dbox[group][1][2])
            # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            yaml_dict = make_yaml_dict(test_scene_num, arm_name , object_name, pick_pose, place_pose)
            dict_list.append(yaml_dict)


        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        # Insert message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, arm_name , object_name, pick_pose, place_pose)

            print ("Response: ", resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # Output request parameters into output yaml file
    send_to_yaml('test4.yaml', dict_list)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model3.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
