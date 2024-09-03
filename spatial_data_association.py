from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import torch
import json, copy
from collections.abc import Iterable 
import math
import numpy as np
from radar_camera_fusion import my_custom_dbscan, visualize_radar_cluster_bbox
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import random
import matplotlib
matplotlib.use('Qt5Agg')
import time

class SensorFusion():

    def __init__(self, path_to_scenes, path_to_yolo):
        self.scene_list = sorted(list(scene for scene in Path(path_to_scenes).iterdir()))
        self.yolo_model = YOLO(path_to_yolo)
        self.json_data = []
        self.ax = None

    def get_image_pcd_lists(self, path_to_scene):
        path_to_images = str(path_to_scene) + "/camera_01/camera_01__data"
        path_to_pcd = str(path_to_scene) + "/radar_01/radar_01__data"

        self.image_list = sorted(list(image for image in Path(path_to_images).iterdir()))
        self.pcd_list = sorted(list(pcd for pcd in Path(path_to_pcd).iterdir()))

        calib_file_path = str(path_to_scene) + "/calibration.json"
        self.sensor_calib_dict = self.extract_calibration_parameters(calib_file_path)
        
        self.json_path = Path(str(path_to_scene) + "/associations.json")

        self.instances_counter = 0
        self.one_to_one_counter = 0
        self.many_to_one_counter = 0
        self.one_to_many_counter = 0
        self.many_to_many_counter = 0
        self.unassigned_counter = 0


    def extract_calibration_parameters(self, calibration_file):
        sensor_calibration_dict = {
        "camera_intrinsics": [],
        "camera_distcoeffs": [],
        "radar_to_camera": [],
        "radar_to_lidar": [],
        "lidar_to_ground": [],
        "camera_to_ground": []
        }

        with open(calibration_file, 'r') as f:
            data = json.load(f)

        for item in data['calibration']:
            if item['calibration'] == 'camera_01':
                sensor_calibration_dict['camera_intrinsics'] = item['k']
                sensor_calibration_dict['camera_distcoeffs'] = item['D']
            elif item['calibration'] == 'radar_01_to_camera_01':
                sensor_calibration_dict['radar_to_camera'] = item['T']
            elif item['calibration'] == 'radar_01_to_lidar_01':
                sensor_calibration_dict['radar_to_lidar'] = item['T']
            elif item['calibration'] == 'lidar_01_to_ground':
                sensor_calibration_dict['lidar_to_ground'] = item['T']
            elif item['calibration'] == 'camera_01_to_ground_homography':
                sensor_calibration_dict['camera_to_ground'] = item['T']

        return sensor_calibration_dict
    
    # Fetch Results from YOLO Predictions
    def class_box_generator_for_pred(self, img):
        split_part = str(img).split(r"INFRA-3DRC_scene")
        self.img_name = r"INFRA-3DRC_scene" + split_part[1]

        prediction_results = self.yolo_model.predict(img)

        for result in prediction_results:
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            detection = result.boxes.xyxy.cpu().numpy()
            self.instances_counter += 1

            # List of lists. Each list represents an object: [class, x1, y1, x2, y2, conf]
            self.bounding_boxes = np.column_stack((cls, detection, conf))

    def get_radar_dict(self, pcd):
        db_scan = my_custom_dbscan(eps1=0.1, eps2=0.5, min_samples=2)
        split_part = str(pcd).split(r"INFRA-3DRC_scene")
        self.pcd_name = r"INFRA-3DRC_scene" + split_part[1]
        
        # Dictionary of Lists. 'cluster' and 'noise'. Each list has 3 lists.
        # For 'cluster':
        # First list: [x_c, y_c, z_c]
        # Second list: [x_m, y_m, z_m]
        # Third list: [v_r]
        #
        # For 'noise':
        # First list: [x, y, z]
        # Third list: [v_r]
        self.radar_points = db_scan.process_pcd_files(pcd)

    # Radar Dictionary: on Image
    def radar_to_camera(self):
        T = self.sensor_calib_dict['radar_to_camera']
        K = self.sensor_calib_dict['camera_intrinsics']
        
        in_radar = self.radar_points
        self.radar_in_camera = {'cluster': [], 'noise': []}
        for key, value in in_radar.items():
            if key == 'cluster':
                for point in value:
                    if point:
                        updated_centroid = self.radar_to_camera_transformer(point[0], T, K)
                        updated_lowest_point = self.radar_to_camera_transformer(point[1], T, K)
                        updated_velocity = point[2]
                        updated_point = [list(updated_centroid), list(updated_lowest_point), list(updated_velocity)]

                        if key in self.radar_in_camera:
                            self.radar_in_camera[key].append(updated_point)
                        else:
                            print('no key exist')
            else:
                for point in value:
                    if point:
                        updated_centroid = updated_centroid = self.radar_to_camera_transformer(point[0], T, K)
                        updated_velocity = [point[1]]
                        updated_point = [list(updated_centroid), list(updated_velocity)]

                        if key in self.radar_in_camera:
                            self.radar_in_camera[key].append(updated_point)
                        else:
                            print('no key exist')
        
        # self.radar_in_camera:
        # Dictionary of Lists. 'cluster' and 'noise'. Each list has 3 lists.
        # For 'cluster':
        # First list: [x_c, y_c, z_c]
        # Second list: [x_m, y_m, z_m]
        # Third list: [v_r]
        #
        # For 'noise':
        # First list: [x, y, z]
        # Third list: [v_r]

    # def check_point(self, cluster, image, scale:bool=False):
    #     """
    #     Checks if centroid points of clusters are within bounding boxes 
    #     in the image plane. 
        
    #     Examples: 
    #     TODO: Add examples of computation for doc-test
    #     >>> check_point([], [])

    #     """ 
    #     bbox = image[1:5]
    #     cluster_centroid = cluster[0]
    #     if bbox[0] < cluster_centroid[0] < bbox[2] and bbox[1] < cluster_centroid[1] < bbox[3]:
    #         return 1
    #     return 0

    def check_point(self, cluster, image, scale:bool=False):
        """
        Checks if centroid points of clusters are within bounding boxes 
        in the image plane.
        """ 
        if bool: 
            _bbox = image[1:5]
            bbox = self.expand_bbox(_bbox, scale_ratio=1.2)
        else: 
            bbox =image[1:5]
        cluster_centroid = cluster[0]
        if bbox[0] < cluster_centroid[0] < bbox[2] and bbox[1] < cluster_centroid[1] < bbox[3]:
            return 1
        return 0
    
    def expand_bbox(self, box, scale_ratio=1.1):
        """
        Expand Bounding box, if flag is passed in the association matrix! 
        """
        # Calculate the width and height of the original box
        width = box[2] - box[0]     # x2 - x1
        height = box[3] - box[1]    # y2 - y1

        # Calculate the center of the original box
        center_x = box[0] + (width/2)
        center_y = box[1] + (height/2)

        # Calculate the increase in width and height
        new_width = width * scale_ratio
        new_height = height * scale_ratio

        # Calculate the new coordinates
        new_x1 = 0 if (center_x - new_width / 2) < 0 else (center_x - new_width / 2)
        new_y1 = 0 if (center_y - new_height / 2) < 0 else (center_y - new_height / 2)
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        return list([new_x1, new_y1, new_x2,new_y2])
    
    # def create_cluster_association_matrix(self):
    #     """
    #     Creates an association matrix by checking if cluster's fall within bounding boxes
    #     """
    #     self.cluster_association_matrix = np.zeros((len(self.radar_in_camera['cluster']), len(self.bounding_boxes)))

    #     for r_dx, cluster in enumerate(self.radar_in_camera['cluster']): 
    #         for i_dx, obj in enumerate(self.bounding_boxes):
    #             self.cluster_association_matrix[r_dx, i_dx] = self.check_point(cluster, obj)
    
    def create_cluster_association_matrix(self, scale_bbox:bool=False):
        """
        Creates an association matrix by checking if cluster's fall within bounding boxes
        """
        self.cluster_association_matrix = np.zeros((len(self.radar_in_camera['cluster']), len(self.bounding_boxes)))

        for r_dx, cluster in enumerate(self.radar_in_camera['cluster']): 
            for i_dx, obj in enumerate(self.bounding_boxes):
                self.cluster_association_matrix[r_dx, i_dx] = self.check_point(cluster, obj, scale=scale_bbox)

    def create_noise_association_matrix(self): 
        """
        Creates a noise association matrix by checking if the cluster's fall within bounding boxes
        """
        if not self.cluster_associations['unassigned_cases']['boxes']:
        # If there are no unassigned cases, exit the function
            self.noise_association_matrix = np.zeros((0, 0))
            return 
         
        self.noise_association_matrix = np.zeros((len(self.radar_in_camera['noise']),
                                                    self.cluster_association_matrix.shape[1]))
        for r_dx, noise in enumerate(self.radar_in_camera['noise']):
            for i_dx in self.cluster_associations['unassigned_cases']['boxes']: 
                self.noise_association_matrix[r_dx, i_dx] = self.check_point(noise, self.bounding_boxes[i_dx])

    def flatten_chain(self, matrix):
        def flatten(items):
            for item in items:
                if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                    yield from flatten(item)
                else:
                    yield item

        return list(flatten(matrix))
    

    def get_associations(self, noise:bool=False):
        """
        Checks and assigns different cases of radar-box data for spatial association 

        Examples: 
        >>> import pprint
        >>> matrix = np.array([
        ...     [1, 0, 1, 0, 0, 0],
        ...     [0, 1, 1, 0, 0, 0],
        ...     [1, 1, 0, 0, 0, 0],
        ...     [0, 0, 0, 1, 0, 0],
        ...     [0, 0, 0, 0, 1, 0],
        ...     [0, 0, 0, 0, 1, 0],
        ...     [0, 0, 0, 0, 0, 0]
        ... ])
        >>> pprint.pprint(rba.get_associations())
        {'many_radar_to_many_box': {'clusters': [0, 1, 2], 'boxes': [0, 1, 2]},
        'many_radar_to_one_box': {'clusters': [[4, 5]], 'boxes': [4]},
        'one_radar_to_many_box': {'clusters': [], 'boxes': []},
        'one_radar_to_one_box': {'clusters': [3], 'boxes': [3]},
        'unassigned_cases': {'clusters': [6], 'boxes': [5]}}
        """ 
        
        
        if noise == True:
            _matrix = self.noise_association_matrix
            self.noise_associations = {
            "many_radar_to_many_box" : {"clusters": [], "boxes": []}, 
            "many_radar_to_one_box"  : {"clusters": [], "boxes": []},
            "one_radar_to_many_box"  : {"clusters": [], "boxes": []}, 
            "one_radar_to_one_box"   : {"clusters": [], "boxes": []},
            'unassigned_cases'       : {"clusters": [], "boxes": []}
            }
            _association_mapping = self.noise_associations
            json_key = 'noise_assignments'
        else: 
            _matrix = self.cluster_association_matrix 
            self.cluster_associations = {
            "many_radar_to_many_box" : {"clusters": [], "boxes": []}, 
            "many_radar_to_one_box"  : {"clusters": [], "boxes": []},
            "one_radar_to_many_box"  : {"clusters": [], "boxes": []}, 
            "one_radar_to_one_box"   : {"clusters": [], "boxes": []},
            "unassigned_cases"       : {"clusters": [], "boxes": []}
            }
            json_key = 'cluster_assignments'
            _association_mapping = self.cluster_associations
            # Needs to be initialized here to avoid reinitialization
            self.final_associations = {'image': self.img_name, 'pcd': self.pcd_name, 'cluster_assignments': {'assigned bb': [], 'assigned radar': []},
                                                                                     'noise_assignments'  : {'assigned bb': [], 'assigned radar'   : []},
                                                                                     'unassigned'         : {'bb': [], 'clusters': [], 'noise': []}
            }

        # MANY TO MANY CHECKS
        # -------------------
        rows_with_multiple_truths = np.where(np.sum(_matrix, axis=1) > 1)[0] 
        columns_with_multiple_truths = np.where(np.sum(_matrix, axis=0) > 1)[0] 
        many_too_many_rows = set() 
        many_too_many_cols = set() 
        for r_id in range(_matrix.shape[0]):
            for c_id in range(_matrix.shape[1]):
                if r_id in rows_with_multiple_truths and c_id in columns_with_multiple_truths: 
                    if _matrix[r_id, c_id] == 1:
                        many_too_many_rows.add(r_id)
                        many_too_many_cols.add(c_id)

        _association_mapping['many_radar_to_many_box']["clusters"].extend(list(many_too_many_rows))
        _association_mapping['many_radar_to_many_box']["boxes"].extend(list(many_too_many_cols))

        # MANY TO ONE CHECKS 
        # ------------------
        for c in range(_matrix.shape[1]): 
            if c in columns_with_multiple_truths and c not in many_too_many_cols:
                associated_rows = np.where(_matrix[:, c] > 0)
                associated_rows = associated_rows[0].tolist()
                _association_mapping['many_radar_to_one_box']["clusters"].append((associated_rows))
                _association_mapping['many_radar_to_one_box']["boxes"].append([c])            

        # ONE TO MANY CHECKS
        # ------------------
        for r in range(_matrix.shape[0]): 
            if r in rows_with_multiple_truths and r not in many_too_many_rows:
                associated_cols = np.where(_matrix[r] > 0)
                associated_cols = associated_cols[0].tolist()
                _association_mapping['one_radar_to_many_box']["clusters"].append((r))
                _association_mapping['one_radar_to_many_box']["boxes"].append((associated_cols))

        # ONE TO ONE CHECKS
        # -----------------
        for i in range(_matrix.shape[0]):
            for j in range(_matrix.shape[1]):
                # print(i,j)
                if _matrix[i,j] == 1:
                    row_sum = sum(_matrix[i,:])
                    col_sum = sum(_matrix[:,j]) 
                    if row_sum == 1 and col_sum == 1:
                        _association_mapping['one_radar_to_one_box']["clusters"].append([i]) 
                        _association_mapping['one_radar_to_one_box']["boxes"].append([j]) 
                        # self.final_associations[json_key]['assigned radar'] += [self.radar_points['cluster'][i]] # Uncomment to output the actual coordinates
                        self.final_associations[json_key]['assigned radar'] += [i]
                        # self.final_associations[json_key]['assigned bb'] += self.bounding_boxes[j].tolist() # Uncomment to output the actual coordinates
                        self.final_associations[json_key]['assigned bb'] += [j]
                        self.one_to_one_counter += 1

        # NO MATCHES CHECKS
        # -----------------
        _association_mapping['unassigned_cases']['clusters'] = [
            r for r in range(_matrix.shape[0]) 
            if  r not in self.flatten_chain(_association_mapping['many_radar_to_many_box']["clusters"]) 
            and r not in self.flatten_chain(_association_mapping['one_radar_to_many_box']["clusters"] )
            and r not in self.flatten_chain(_association_mapping['many_radar_to_one_box']["clusters"])
            and r not in self.flatten_chain(_association_mapping['one_radar_to_one_box']["clusters"])
            # and r not in self.flatten_chain(self.cluster_associations['unassigned_cases']['clusters'])
        ]
        _association_mapping['unassigned_cases']['boxes'] = [
            c for c in range(_matrix.shape[1]) 
            if  c not in self.flatten_chain(_association_mapping['many_radar_to_many_box']["boxes"] )
            and c not in self.flatten_chain(_association_mapping['one_radar_to_many_box']["boxes"] )
            and c not in self.flatten_chain(_association_mapping['many_radar_to_one_box']["boxes"])
            and c not in self.flatten_chain(_association_mapping['one_radar_to_one_box']["boxes"])
            # and c not in self.flatten_chain(self.cluster_associations['unassigned_cases']['boxes'])
            and c not in self.flatten_chain(self.cluster_associations['many_radar_to_many_box']["boxes"] )
            and c not in self.flatten_chain(self.cluster_associations['one_radar_to_many_box']["boxes"] )
            and c not in self.flatten_chain(self.cluster_associations['many_radar_to_one_box']["boxes"])
            and c not in self.flatten_chain(self.cluster_associations['one_radar_to_one_box']["boxes"])
        ]

    # def find_and_group_similar_velocities(self, velocities, threshold=0.5, noise: bool = False)->list:
    #     """
    #     Finds and groups similar velocty points

    #     """
    #     grouped_points = []
    #     used_indices = set()
    #     if noise == True: 
    #         points = self.radar_points['noise']
    #         v_index = 1 # Due to different structure, the position of velocity will be different
    #     else: 
    #         points = self.radar_points['cluster']
    #         v_index = 2

    #     for i in velocities:
    #         if i in used_indices:
    #             continue
            
    #         current_group = [i]
    #         used_indices.add(i)

    #         for j in range(i + 1, len(velocities)):
    #             if j in used_indices:
    #                 continue
                
    #             if abs(points[i][0][2] - points[j][0][2]) <= threshold:
    #                 current_group.append(j)
    #                 used_indices.add(j)

    #         grouped_points.append(current_group)

    #     return grouped_points

    def find_and_group_similar_velocities(self, velocities, threshold=0.5, noise:bool=False):
        grouped_points = []
        used_indices = set()
        if noise == True: 
            points = self.radar_points['noise']
            v_index = 1 # Due to different structure, the position of velocity will be different
        else: 
            points = self.radar_points['cluster']
            v_index = 2

        for i in velocities:
            if i not in used_indices:
                current_group = [i]
                used_indices.add(i)
                for j in velocities[i+1:]:
                    if abs(points[i][v_index][0] - points[j][v_index][0]) <= threshold:
                        current_group.append(j)
                        used_indices.add(j)
                grouped_points.append(current_group)
        
        return grouped_points
    
    def merge_clusters(self, clusters, noise:bool=False):
        avg_centroid = np.empty((1,3))
        avg_lowest_point = np.empty((1,3))
        avg_velocity = 0
        if noise: 
            points = self.radar_points['noise']
            v_index = 1
        else: 
            points = self.radar_points['cluster']
            v_index = 2
        
        for cluster in clusters:
            # print(f"Cluster: {cluster}")
            # print(f"Points: {points}")
            avg_centroid += np.array(points[cluster][0])
            avg_lowest_point += np.array(points[cluster][1])
            avg_velocity += points[cluster][v_index][0] 
        
        avg_centroid /= len(clusters)
        avg_lowest_point /= len(clusters)
        avg_velocity /= len(clusters)

        merged_cluster = [avg_centroid.tolist()[0], avg_lowest_point.tolist()[0], [avg_velocity]]

        return merged_cluster
    
    def get_lowest_center_point(self, box):
        x_bot_mid = (box[0] + box[2]) / 2
        y_bot_mid = box[3]
        return np.array([[x_bot_mid, y_bot_mid]])
    
    def convert_radar_to_ground(self, points): 
        pass 
    
    # def calculate_merged_point(self, point_1, point_2, noise=False):
    #     avg_centroid += np.array(point_1)
    
   
    def get_closest_point_to_box(self, cluster_ids: list, box_id: int, noise: bool = False):
        self.radar_in_ground = self.radar_to_ground(self.radar_points)
        self.bounding_boxes_in_ground = self.homography(self.bounding_boxes)

        box_bot_mid_point = self.get_lowest_center_point(self.bounding_boxes_in_ground[box_id])

        ground_points = self.radar_in_ground['noise'] if noise else self.radar_in_ground['cluster']
        # print(f"Noise Cluster ID: {cluster_ids}")
        # print(f"Radar in ground: {ground_points}")

        closest_id = min(cluster_ids, key=lambda id: self.compute_distance(box_bot_mid_point, ground_points[id]))

        return closest_id
    
    def compute_distance(self, box_bot_mid_point, ground_point):
        point = ground_point[0] if isinstance(ground_point, list) else ground_point
        return np.linalg.norm(box_bot_mid_point - point.reshape((1, 3))[:, :2])
    
    def assess_many_radar_to_one_box(self, noise:bool=False):
        if noise == True:
            many_radar_to_one_box = self.noise_associations['many_radar_to_one_box'] # J Changed
            json_key = 'noise_assignments'
        else:  
            many_radar_to_one_box = self.cluster_associations['many_radar_to_one_box']
            json_key = 'cluster_assignments'
        # print()    
        # print("Initial many_clusters_to_one_box:")
        # print(many_radar_to_one_box)
        # print()    

        # index = 0
        # Create copies of the lists to iterate over
        boxes_copy = copy.deepcopy(many_radar_to_one_box['boxes'])
        clusters_copy = copy.deepcopy(many_radar_to_one_box['clusters'])

        # iterate through the individual cases in the frame
        for index, (box_id, cluster_set) in enumerate(zip(boxes_copy, clusters_copy)):
            alone_clusters = []
            merged_cluster = []
            merge_group = None

            # iterate through the associated cluster set and group clusters with similar velocities together 
            cluster_groups = self.find_and_group_similar_velocities(cluster_set, threshold=0.5, noise=noise)
            # print(f"Cluster groups: {cluster_groups}")
            for group in cluster_groups:
                # print(f"groups: {group}")
                if len(group) == 1:
                    alone_clusters.append(group[0])
                # Merge clusters with similar velocities (in the same group)
                elif len(group) > 1:
                    merge_group = copy.deepcopy(group)
                    merged_cluster.append(self.merge_clusters(group, noise=noise))
                    group_sorted = sorted(group, reverse=True)
                    # print(f"Group sorted: {group_sorted}")
                    # Remove the associated clusters
                    for i in group_sorted:
                        # print(f"i: {i}")
                        # print(f"Many radar to one box: {many_radar_to_one_box['clusters'][index]}")
                        many_radar_to_one_box['clusters'][index].remove(i)

            # Associate the merged clusters to the bounding box. 
            # TODO: Associate the closest merged_cluster, in case of many of them.
            if len(merged_cluster) > 0:
                # self.final_associations['bb'] += [self.bounding_boxes[box_id[0]].tolist()] # Uncomment to output the actual coordinates
                self.final_associations[json_key]['assigned bb'] += box_id
                # self.final_associations['clusters'] += [merged_cluster] # Uncomment to output the actual coordinates
                self.final_associations[json_key]['assigned radar'] += [merge_group]
                # Remove the associated box
                # print(f"Many radar to one bbox: {many_radar_to_one_box['boxes']}")
                many_radar_to_one_box['boxes'].remove(box_id)
                self.many_to_one_counter += 1
            # If there is no merged clusters, then associate the closest cluster to the box
            else:
                # Take the closest point to the box bottom center.
                closest_id = self.get_closest_point_to_box(alone_clusters, box_id[0], noise=noise)

                # self.final_associations['bb'] += [self.bounding_boxes[box_id[0]].tolist()] # Uncomment to output the actual coordinates
                self.final_associations[json_key]['assigned bb'] += box_id
                # self.final_associations['clusters'] += [self.radar_points['cluster'][closest_id]]  # Uncomment to output the actual coordinates
                self.final_associations[json_key]['assigned radar'] += [closest_id]

                # Remove the associated box and clusters
                # print(f"box_id: {box_id}")
                # print(f"Many radar to one bbox: {many_radar_to_one_box['boxes']}")
                many_radar_to_one_box['boxes'].remove(box_id)
                many_radar_to_one_box['clusters'][index].remove(closest_id)
                self.many_to_one_counter += 1

            # Sort the leftover clusters as unassigned
            for cluster in many_radar_to_one_box['clusters'][index]:
                # self.unass['clusters'] += [self.radar_points['cluster'][cluster]] # Uncomment to output the actual coordinates
                self.final_associations['unassigned']['clusters'] += [cluster]


    def handle_one_radar_to_many_box(self, noise:bool=False):
        """
        Handles the case of many bounding boxes associated with one radar
        
        """
        self.radar_in_ground = self.radar_to_ground(self.radar_points)
        self.bounding_boxes_in_ground = self.homography(self.bounding_boxes)
        # points = self.radar_in_ground['noise'] if noise else self.radar_in_ground['cluster']
        
        if noise: 
            r_indexes               = self.noise_associations['one_radar_to_many_box']['clusters'] 
            b_indexes               =  self.noise_associations['one_radar_to_many_box']['boxes']
            _association_mapping    = self.noise_associations
            points                  = self.radar_in_ground['noise']
            json_key = 'noise_assignments'

        else:
            r_indexes               = self.cluster_associations['one_radar_to_many_box']['clusters'] 
            b_indexes               = self.cluster_associations['one_radar_to_many_box']['boxes']
            _association_mapping    = self.cluster_associations 
            points                  = self.radar_in_ground['cluster'] # Take the ground by adding index [1]
            json_key = 'cluster_assignments'

        for r_id , b_id in  enumerate(b_indexes):
            box_ids = copy.deepcopy(b_id)
            print("initial b_indexes:")
            print(box_ids)
            bbox_1      = self.flatten_chain(self.get_lowest_center_point(self.bounding_boxes_in_ground[b_id[0]]))
            bbox_2      = self.flatten_chain(self.get_lowest_center_point(self.bounding_boxes_in_ground[b_id[1]]))
            radar_point = self.flatten_chain(points[r_id][1]) # Take closest point here
            distances = np.array([
            self.compute_eucledian(radar_point, bbox_1),
            self.compute_eucledian(radar_point, bbox_2)
        ])
            d_index = np.argmin(distances)

            self.final_associations[json_key]['assigned radar'].append(r_id)
            self.final_associations[json_key]['assigned bb'].append(b_id[d_index])
            box_ids.remove(b_id[d_index])
            self.one_to_many_counter += 1

            _association_mapping["one_radar_to_many_box"]["clusters"].remove(
                r_indexes[r_id]
            )

            _association_mapping["one_radar_to_many_box"]["boxes"].remove(
                b_id
            )

            self.final_associations['unassigned']['bb'] += box_ids

            print("final b_indexes:")
            print(box_ids)
        
        # print("AFTER _association_mapping")
        # print(_association_mapping)
        
    def compute_eucledian(self, point_1:list, point_2:list)->float: 
        return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)    

    
    def handle_many_radar_to_many_boxes(self, velocity_threshold = 0.5, distance_threshold=0.5, 
                                        noise:bool = False):
        """
        Handles many radar to many bounding box cases 
        """ 
        
        if noise: 
            radar_points_on_image   = self.radar_in_camera['noise']
            radar_points_on_ground  = self.radar_in_ground['noise']
            r_indexes               = self.noise_associations['many_radar_to_many_box']['clusters']
            b_indexes               = self.noise_associations['many_radar_to_many_box']['boxes']
            _association_mapping    = self.noise_associations
            json_key = 'noise_assignments'
        else:  
            radar_points_on_image   = self.radar_in_camera['cluster']
            r_indexes               = self.cluster_associations['many_radar_to_many_box']['clusters']
            b_indexes               = self.cluster_associations['many_radar_to_many_box']['boxes']
            _association_mapping    = self.cluster_associations      
            json_key = 'cluster_assignments'
        
        # 1: Merge clusters based on velocity 
        merged_cluster_indexes = self.find_and_group_similar_velocities(velocities=r_indexes, threshold=velocity_threshold, noise=noise) # (merged_cluster_indexes)
        # Each list of list - if inner if it has more than one - -then same velocity .
        copy_updated_clusters = self.find_and_group_similar_velocities(velocities=r_indexes, threshold=velocity_threshold, noise=noise) 
        
        # TODO: Merge inside of the list and note to put the co-ordinate to the ground  
        merged_clusters = [self.merge_clusters(clusters, noise=noise) for clusters in merged_cluster_indexes]
        
        # # 2: Distance based association
        if not b_indexes: 
            distance_matrix = np.zeros((0,0))
        else: 
            distance_matrix = np.zeros((len(merged_clusters), len(self.bounding_boxes)))
        # if unassigned_bboxes: # If cases still left
        for r_id, clusters in enumerate(merged_clusters):
            # print(f"Clusters: {clusters}")
            for c_id in b_indexes:
                # Assuming you get clusters on ground and in [x,y,z]
                distance_matrix[r_id, c_id] = self.check_distance_threshold(
                                                self.compute_eucledian(clusters[r_id], self.bounding_boxes_in_ground[c_id]), 
                                                distance_threshold=distance_threshold)
        
        # 2.1: Check for points that pass eucledian thresholds 
        # cols_with_passed_threshold = np.where(np.sum(distance_matrix, axis=0) == 1)[0]# Gather all the cols that passed the (0 < value <= threshold)  
        # print(f"Distance_Matrix: {distance_matrix}")
        # rows_with_passed_threshold = np.where(np.sum(distance_matrix, axis=1) == 1)[0]
        for c_id in range(distance_matrix.shape[1]):
            if np.sum(distance_matrix[:,c_id] == 1):
                _association_mapping['one_radar_to_one_box']['clusters'].extend([[i] for i in np.where(np.sum(distance_matrix[:,c_id] == 1))])
                self.final_associations[json_key]['assigned radar'].extend([[i] for i in np.where(np.sum(distance_matrix[:,c_id] == 1))])
                self.many_to_many_counter += 1
            else: 
                _association_mapping['one_radar_to_one_box']['clusters'].extend([[i] for i in np.where(np.min(distance_matrix[:,c_id]))]) 
                self.final_associations[json_key]['assigned radar'].extend([[i] for i in np.where(np.min(distance_matrix[:,c_id]))]) 
                self.many_to_many_counter += 1

    # def check_distance_

    def create_noise_association_matrix(self): 
        """
        Creates a noise association matrix by checking if the cluster's fall within bounding boxes
        """
        if not self.cluster_associations['unassigned_cases']['boxes']:
        # If there are no unassigned cases, exit the function
            self.noise_association_matrix = np.zeros((0, 0))
            return 
         
        self.noise_association_matrix = np.zeros((len(self.radar_in_camera['noise']),
                                                    self.cluster_association_matrix.shape[1]))
        for r_dx, noise in enumerate(self.radar_in_camera['noise']):
            for i_dx in self.cluster_associations['unassigned_cases']['boxes']: 
                self.noise_association_matrix[r_dx, i_dx] = self.check_point(noise, self.bounding_boxes[i_dx])

    # Calibration: Radar to Image Plane
    def radar_to_camera_transformer(self, radar_point, T, k):    
        n_p_array = np.array(radar_point).reshape(1,-1)
        transpose_RPA = np.transpose(n_p_array)

        new_array = np.vstack([transpose_RPA, np.ones((1, 1))])             
        product_1 = np.matmul(np.array(k), np.array(T))

        product_array = np.matmul(product_1, new_array)                      #[su, sv, s] but along column

        final_array = product_array / product_array [2]                      #[u, v, 1], along column

        u_v = np.delete(final_array, 2, axis = 0)                            #[u, v], along column      
        final_u_v = np.transpose(u_v)

        return final_u_v[0]

    def radar_to_ground(self, radar_dict):
        """Transforms the radar clusters and noise points coordinates of ONE frame from the Radar CS to the Ground CS

        Args:
            radar_dict (dictionary): The clustering algorithem output; noise/cluster keys with lists of 2 lists. Each point is a list of 2 lists; one for x,y,z position and one for the velocity.
            sensor_calibration_dict (dictionary): The dictionary of the calibration file

        Returns:
            dictionary: 2 keys: 'cluster' and 'noise'. Each key has a list of points. Each point has an np.array of [x,y,z] and a float value of the velocity.
        """
        radar_to_lidar = np.array(self.sensor_calib_dict['radar_to_lidar']) # Creates an array from the list
        radar_to_lidar = np.vstack((radar_to_lidar, [0, 0, 0, 1])) # Adds the homogeneous row 
        lidar_to_ground = np.array(self.sensor_calib_dict['lidar_to_ground'])
        lidar_to_ground = np.vstack((lidar_to_ground, [0, 0, 0, 1]))

        self.ground_radar = {'cluster': [], 'noise': []} # To save the radar points on the ground CS 

        for key, points in radar_dict.items():
            if key == 'cluster':
                for point in points:
                    if len(point):
                        centroid = np.append(np.array(point[0]), 1).reshape((4,1)) # Creates an array from the point list, adds 1, and reshapes to 4x1
                        ground_centroid = np.matmul(np.matmul(radar_to_lidar, lidar_to_ground), centroid) # Transforms the point to ground CS
                        lowest_point = np.append(np.array(point[1]), 1).reshape((4,1)) # Creates an array from the point list, adds 1, and reshapes to 4x1
                        ground_lowest_point = np.matmul(np.matmul(radar_to_lidar, lidar_to_ground), lowest_point) # Transforms the point to ground CS
                        ground_point = [ground_centroid[:-1], ground_lowest_point[:-1], point[1][0]] # Combine the point on ground CS with the velocity
                        # ground_point  =  [ground_lowest_point[:-1], ]
                        self.ground_radar['cluster'].append(ground_point)

            if key == 'noise':
                for point in points:
                    if len(point):
                        point = np.append(np.array(point[0]), 1).reshape((4,1)) # Creates an array from the point list, adds 1, and reshapes to 4x1
                        ground_point = np.matmul(np.matmul(radar_to_lidar, lidar_to_ground), point) # Transforms the point to ground CS
                        ground_point = [ground_point[:-1], point[1][0]] # Combine the point on ground CS with the velocity
                        self.ground_radar['noise'].append(ground_point)
        
        return self.ground_radar

    def homography(self, list_of_pred_boxes):
        """Transforms the prediction boxes coordinates of ONE frame from the Camera CS to the Ground CS

        Args:
            list_of_pred_boxes (np.array of np.arrays): the array of the prediction bounding boxes
            sensor_calibration_dict (dictionary): The dictionary of the calibration file

        Returns:
            List[Lists]: Each inner list has an np.array of the Ground coordinates of the BB [np.array[x1, y1, x2, y2], ...] and an integer of the object class
        """
        homo_matrix = np.array(self.sensor_calib_dict['camera_to_ground'])
        bb_ground = []

        for box in list_of_pred_boxes:
            x1y1 = np.array([[box[0]], 
                            [box[1]],
                            [1]]
            )

            x2y2 = np.array([[box[2]], 
                            [box[3]],
                            [1]]
            )

            hx1y1 = np.matmul(homo_matrix, x1y1)
            hx1y1 = hx1y1 / hx1y1[-1]
            hx1y1 = hx1y1[:2]
            hx1y1 = hx1y1.reshape((1,2))

            hx2y2 = np.matmul(homo_matrix, x2y2)
            hx2y2 = hx2y2 / hx2y2[-1]
            hx2y2 = hx2y2[:2]
            hx2y2 = hx2y2.reshape((1,2))

            ground_x1y1x2y2 = np.hstack((hx1y1, hx2y2))
            ground_object = [ground_x1y1x2y2, int(box[0])]
            ground_point_list = ground_x1y1x2y2.tolist() 

            # Flatten the list
            # ground_point_list = [item for sublist in ground_point_list for item in sublist if item != 1]

            bb_ground.append(ground_point_list[0])

        return bb_ground

    def get_contained_clusters(self, box, radar_on_ground):
        """Find the clusters that are inside of the input bounding box

        Args:
            box (List): The input bounding box
            radar_on_ground (Dictionary): The radar clusters and noise points

        Returns:
            List: The clusters contained inside the bounding box
        """
        contained_clusters = []

        for cluster in radar_on_ground['cluster']:
            if box[0][0] <= cluster[0].flatten()[0] <= box[0][2] and box[0][1] <= cluster[0].flatten()[1] <= box[0][3]:
                contained_clusters.append(cluster)

        return contained_clusters
    
    def append_json_data(self):
        self.json_data.append(self.final_associations)

    def create_json(self):
        self.unassigned_counter = self.instances_counter - (self.one_to_one_counter + self.many_to_one_counter + self.one_to_many_counter + self.many_to_many_counter)
        counters = {'instances_count': self.instances_counter,
                    'one_to_one_count': self.one_to_one_counter,
                    'many_to_one_count': self.many_to_one_counter,
                    'one_to_many_count': self.one_to_many_counter,
                    'many_to_many_count': self.many_to_many_counter,
                    'unassigned_count': self.unassigned_counter}
        self.json_data.append(counters)
        with self.json_path.open('w') as file:
            json.dump(self.json_data, file, indent=4)        
    
    #### VISUALIZATIONS #### 
    def visualize_radar_cluster_bbox(self, ax, radar_points: list, bbox_points: np.ndarray, image_path: str):
        """
        Visualize radar and bbox together for a single image
        Args:
            ax: Matplotlib axis object to plot on
            radar_points: List of lists of radar points [x_c, y_c, z_c], [x_min, y_min, z_min], [velocity]
            bbox_points: numpy array of bounding boxes [class, x1, y1, x2, y2, confidence]
            image_path: Path to the image file
        """
        img = plt.imread(image_path)
        ax.imshow(img)
        _radar_points = radar_points['cluster']
        noise_points = radar_points['noise'] 
        # Plot bounding boxes
        for bbox in bbox_points:
            class_id, x1, y1, x2, y2, confidence = bbox
            rectangle = Rectangle(
                xy=(x1, y1),
                width=x2 - x1,
                height=y2 - y1,
                color='red',
                fill=False,
                linewidth=0.5
            )
            ax.add_patch(rectangle)
            ax.text(x1, y1 - 10, f'{int(class_id)}: {confidence:.2f}', color='red', fontsize=6, weight='bold')

        # Plot radar points
        counter = 1
        for radar_set in _radar_points:
            x_c, y_c = radar_set[0]
            # print(radar_set)    
            # x_min, y_min = radar_set[1]
            # velocity = radar_set[2][0]

            # Plot center and min points
            ax.scatter(x_c, y_c, c='blue', s=50/counter, label='Centriod') # CAN CHANGE 
            # ax.scatter(x_min, y_min, c='green', s=50/counter, label=f'{counter}: Min Point')
            #TODO: Add Gray points 

            # Optionally, plot the z-coordinate as size or other attributes
            # ax.scatter(x_c, y_c, c='blue')  # Scale the size with z-coordinate
            # ax.scatter(x_min, y_min, c='green')  # Scale the size with z-coordinate

            # counter += 1    
        
        for radar_s in noise_points:
            # print(f"Noise: {radar_s}")
            x_c, y_c = radar_s[0]
            # x_min, y_min = radar_s[1]
            # velocity = radar_s[2][0]

            # Plot center and min points
            ax.scatter(x_c, y_c, c='gray', s=50/counter, label='Noise') # CAN CHANGE 
            # ax.scatter(x_min, y_min, c='green', s=50/counter, label=f'{counter}: Min Point')
            #TODO: Add Gray points 

            # Optionally, plot the z-coordinate as size or other attributes
            # ax.scatter(x_c, y_c, c='blue')  # Scale the size with z-coordinate
            # ax.scatter(x_min, y_min, c='green')  # Scale the size with z-coordinate

        ax.legend(loc='upper right')

    def rewrite_visualize_radar_cluster(self):
        pass 

    
    def updated_radar_cluster_association(self, ax, radar_points: dict, bbox_points: np.ndarray, image_path: str):
        img = plt.imread(image_path)
        ax.imshow(img)

        # Retrieve radar points and final associations
        index_final_cluster_radar_point = self.final_associations['cluster_assignments']['assigned radar']
        index_final_cluster_bbox_points = self.final_associations['cluster_assignments']['assigned bb']
        _radar_points = radar_points['cluster']

        index_final_noise_radar_point = self.final_associations['noise_assignments']['assigned radar']
        index_final_noise_bbox_points = self.final_associations['noise_assignments']['assigned bb']
        _noise_radar_points = radar_points['noise']

        total_radar_points = len(index_final_cluster_radar_point) + len(index_final_noise_radar_point) 
        total_box_points =  len(index_final_cluster_bbox_points) + len(index_final_noise_bbox_points)
        
    
        # Generate a list of unique colors
        cmap = plt.get_cmap('plasma')
        radar_colors = [cmap(i) for i in np.linspace(0, 1, total_radar_points)]
        box_colors   = [cmap(i) for i in np.linspace(0, 1, total_box_points)]

        color_index = 0

        # Plot cluster radar points
        for i, idx in enumerate(index_final_cluster_radar_point):
            # if color_index >= len(colors):
            #     break
            radar_color = radar_colors[i]
            color_index += 1
            if isinstance(idx, int):
                radar_set = _radar_points[idx]
                x_c, y_c = radar_set[0]
                ax.scatter(x_c, y_c, color=radar_color, s=50, label='Centroid')

            elif isinstance(idx, list):
                merged_radar_point = self.merge_clusters(self.flatten_chain(index_final_cluster_radar_point))
                xc, yc = self.radar_to_camera_transformer(
                    merged_radar_point[0],
                    T=self.sensor_calib_dict['radar_to_camera'],
                    k=self.sensor_calib_dict['camera_intrinsics']
                )
                ax.scatter(xc, yc, color=radar_color, s=50)
        
        # Plot noise bounding boxes
        for i, idx in enumerate(index_final_cluster_bbox_points):
            # if color_index >= len(box_colors):
            #     break
            radar_color = box_colors[i]
            color_index += 1
            class_id, x1, y1, x2, y2, confidence = bbox_points[idx]
            rectangle = Rectangle(
                xy=(x1, y1),
                width=x2 - x1,
                height=y2 - y1,
                edgecolor=radar_color,
                fill=False,
                linewidth=0.5
            )
            ax.add_patch(rectangle)
            ax.text(x1, y1 - 10, f'{int(class_id)}: {confidence:.2f}', color=radar_color, fontsize=6, weight='bold')

        # Plot noise radar points
        for i, idx in enumerate(index_final_noise_radar_point):
            # if color_index >= len(colors):
            #     break
            radar_color = radar_colors[i]
            color_index += 1
            if isinstance(idx, int):
                radar_set = _noise_radar_points[idx]
                x_c, y_c = radar_set[0]
                ax.scatter(x_c, y_c, color=radar_color, s=50, label='Noise')

            elif isinstance(idx, list):
                merged_radar_point = self.merge_clusters(self.flatten_chain(index_final_noise_radar_point), noise=True)
                xc, yc = self.radar_to_camera_transformer(
                    merged_radar_point[0],
                    T=self.sensor_calib_dict['radar_to_camera'],
                    k=self.sensor_calib_dict['camera_intrinsics']
                )
                ax.scatter(xc, yc, color=radar_color, s=50)

        # Plot noise bounding boxes
        for i, idx in enumerate(index_final_noise_bbox_points):
            # if color_index >= len(colors):
            #     break
            radar_color = box_colors[i]
            color_index += 1
            class_id, x1, y1, x2, y2, confidence = bbox_points[idx]
            rectangle = Rectangle(
                xy=(x1, y1),
                width=x2 - x1,
                height=y2 - y1,
                edgecolor=radar_color,
                fill=False,
                linewidth=0.5
            )
            ax.add_patch(rectangle)
            ax.text(x1, y1 - 10, f'{int(class_id)}: {confidence:.2f}', color=radar_color, fontsize=6, weight='bold')

        ax.set_title('Associated Radar Points and Bounding Boxes')
        ax.legend(loc='upper right')

    def _update(self, frame):
        # Update your data or plot here based on frame number
        img = self.image_list[frame]
        pcd = self.pcd_list[frame]
        # Example: update radar_in_camera, bounding_boxes based on img, pcd or other logic
        self.class_box_generator_for_pred(img)
        self.get_radar_dict(pcd)
        self.radar_to_camera()
        self.radar_to_ground(self.radar_points)

        # Perform other necessary computations or updates here
        self.create_cluster_association_matrix(scale_bbox=True)
        self.get_associations()
        self.assess_many_radar_to_one_box()
        self.handle_one_radar_to_many_box()
        self.create_noise_association_matrix()
        self.get_associations(noise=True)
        self.assess_many_radar_to_one_box(noise=True)
        self.handle_one_radar_to_many_box(noise=True)
        print(f"Final Cluster Associations:\n{self.final_associations['cluster_assignments']}\n")
        print(f"Final Noise Associations:\n{self.final_associations['noise_assignments']}\n")
        print(f"Final Unassigned:\n{self.final_associations['unassigned']}\n")
        # Clear the axis and update the plot using the specified plot_function
        if self.ax is None:
            self.ax = plt.gca()
        self.ax.clear()
        self.updated_radar_cluster_association(self.ax, self.radar_in_camera, self.bounding_boxes, img)
        # self.visualize_radar_cluster_bbox(self.ax, self.radar_in_camera, self.bounding_boxes, img)
        self.ax.legend(loc='upper right')
        self.ax.set_title('After Association: Radar Points and Bounding Boxes')
        # self.ax.set_title('Before Association: Radar Points and Bounding Boxes')


    def update_animation(self):
        fig, self.ax = plt.subplots()
        ani = animation.FuncAnimation(fig, self._update, frames=len(self.image_list), interval=1, repeat=True)
        plt.show()

    def project_stuff_to_ground(self, bounding_box, radar_points_in_ground):
        # boxes_to_ground = self.homography([bounding_box])
        fig, ax = plt.subplots()

        # PLOT BOUNDING BOX: 
        for bbox in bounding_box:
            print(f"Bounding Box: {bbox}")
            x1, y1, x2, y2 = bbox
            rectangle = Rectangle(
                xy=(x1, y1),
                width=x2 - x1,
                height=y2 - y1,
                color='red',
                fill=False,
                linewidth=0.5
            )

            ax.add_patch(rectangle)
            # ax.text(x1, y1 - 10, f'{int(class_id)}: {confidence:.2f}', color='red', fontsize=6, weight='bold')
        
        # Plot radar points 
        for radar_set in radar_points_in_ground['cluster']:
            print(f"Radar set: {radar_set[0]}")
            x_c, y_c, _ = self.flatten_chain(radar_set[0])
            # print(radar_set)    
            x_min, y_min, _ = self.flatten_chain(radar_set[1])
            # velocity = radar_set[2][0]

            # Plot center and min points
            ax.scatter(x_c, y_c, c='blue', label="Centroid") # CAN CHANGE

        ax.legend(loc='upper right')
        plt.show()


def main()->dict:
    path_to_scenes =r'D:\Sensor_Fusion\ss_project\data\INFRA-3DRC-Dataset'
    path_to_scene = r"D:\Sensor_Fusion\ss_project\data\INFRA-3DRC-Dataset\INFRA-3DRC_scene-06"
    path_to_yolo =  r"D:\Sensor_Fusion\ss_project\fusion\best.pt"
    
    sf = SensorFusion(path_to_scenes, path_to_yolo)
    # # # ALL SCENES 
    # for scene in sf.scene_list:
    #     sf.get_image_pcd_lists(scene)
    #     for img, pcd in zip(sf.image_list, sf.pcd_list):
    #         sf.class_box_generator_for_pred(img)
    #         sf.get_radar_dict(pcd)
    #         sf.radar_to_camera()
    #         # print(f"Radar to Camera : {sf.radar_in_camera}")
    #         # print(f"Radar to Ground : {sf.radar_to_ground(sf.radar_points)}")
    #         # print(f"Camera to Ground: {sf.homography(sf.bounding_boxes)}")
        
    #         # DO STUFF FOR CLUSTERS FIRST 
    #         sf.create_cluster_association_matrix(scale_bbox=True)
    #         # print(f"Cluster Association Matrix: \n{sf.cluster_association_matrix}\n")

    #         sf.get_associations()
    #         # print(f"Cluster Associations: \n{sf.cluster_associations}\n")

    #         sf.assess_many_radar_to_one_box()
    #         sf.handle_one_radar_to_many_box()
    #         # sf.handle_many_radar_to_many_boxes()
            
    #         # DO STUFF FOR NOISE NEXT
    #         sf.create_noise_association_matrix()
    #         # print(f"Noise Association Matrix: \n{sf.noise_association_matrix}\n")

    #         sf.get_associations(noise=True)
    #         # print(f"Noise Association: \n{sf.noise_associations}\n")

    #         sf.assess_many_radar_to_one_box(noise=True)
    #         sf.handle_one_radar_to_many_box(noise=True)
    #         # sf.handle_many_radar_to_many_boxes(noise=True)

            
    #         print(f"Final Cluster Associations:\n{sf.final_associations['cluster_assignments']}\n")
    #         print(f"Final Noise Associations:\n{sf.final_associations['noise_assignments']}\n")
    #         print(f"Final Unassigned:\n{sf.final_associations['unassigned']}\n")
    #         sf.update_animation()

    #         # TODO: Add visualization of ground plane
    #     # sf.create_json()
    
    # PARTICULAR SCENE 
    # path_to_scene = r"C:\Users\alhasan\Documents\Python Scripts\datass\INFRA-3DRC-Dataset\INFRA-3DRC_scene-22"
    sf.get_image_pcd_lists(path_to_scene)
    frame = 0
    for img, pcd in zip(sf.image_list, sf.pcd_list):
        sf.class_box_generator_for_pred(img)
        sf.get_radar_dict(pcd)
        sf.radar_to_camera()
        # print(f"Radar to Camera : {sf.radar_in_camera}")
        # print(f"Radar to Ground : {sf.radar_to_ground(sf.radar_points)}")
        # print(f"Camera to Ground: {sf.homography(sf.bounding_boxes)}")
    
        # DO STUFF FOR CLUSTERS FIRST 
        sf.create_cluster_association_matrix(scale_bbox=True)
        # print(f"Cluster Association Matrix: \n{sf.cluster_association_matrix}\n")

        sf.get_associations()
        # print(f"Cluster Associations: \n{sf.cluster_associations}\n")

        sf.assess_many_radar_to_one_box()
        sf.handle_one_radar_to_many_box()
        # sf.handle_many_radar_to_many_boxes()
        
        # DO STUFF FOR NOISE NEXT
        sf.create_noise_association_matrix()
        # print(f"Noise Association Matrix: \n{sf.noise_association_matrix}\n")

        sf.get_associations(noise=True)
        # print(f"Noise Association: \n{sf.noise_associations}\n")

        sf.assess_many_radar_to_one_box(noise=True)
        sf.handle_one_radar_to_many_box(noise=True)
        # sf.handle_many_radar_to_many_boxes(noise=True)

        
        print(f"Final Cluster Associations:\n{sf.final_associations['cluster_assignments']}\n")
        print(f"Final Noise Associations:\n{sf.final_associations['noise_assignments']}\n")
        print(f"Final Unassigned:\n{sf.final_associations['unassigned']}\n")

    #     sf.append_json_data()

    #     # sf.project_stuff_to_ground(sf.bounding_boxes, sf.radar_in_ground)
    #     # fig, sf.ax = plt.subplots()
    #     # sf._update(frame, self.updated_radar_cluster_association)
    #     # ani = animation.FuncAnimation(fig, sf._update, frames=len(sf.image_list), interval=1, repeat=False)
    #     # layout = QVBoxLayout(central_widget)
    #     # TODO: COMMENT OUT FOR ANIMATION 
        sf.update_animation()
    #     # plt.show()    
    #     # # visualize_radar_cluster_bbox(sf.radar_in_camera['cluster'], sf.bounding_boxes, img)

    # sf.create_json()

if __name__ == '__main__':
    main()

# TODO: Add the below case assessment functions:
# For one to many: check the velocities of clusters and merge the ones with same velocity
# For many to one: shift to ground and assign the closest cluster point (on x) to the lowest bounding box point
# For many to many: merge the clusters with the same velocity, shift to ground, assign depending on distance
# When merging based on velocities keep a distance threshold in mind


## ALL FUNCTIONS AND THEIR DESCRIPTIONS - INPUTS - OUTPUTS 
# TRANSFORMATIONS 
# ---------------
# radar_to_camera() 
# radar_to_ground() 
# camera_to_ground() 

# CASE FILTER
# ------------
# noise_associaition_matrix() 
# cluster_association_matrix() 

# TODO: 
# -----------
# Bounding Box scaling 
# Many-to-Many 
# One -to-Many
# Many-to-One
# add scaling bounding boxes
# add noise related clusters separately in the final dict, else, how do you know 
# TODO: Add the bounding box scaling feature --can be scaled by 20%     
# UPDATE THE EPSILON EQUATION TO NON_LINEAR FUNCTION 

# TODO:
# - Everything unassigned is added S
# - Many-to-many J 
# - test-everything works
# - remove stuff --once assigned J 
# - visual for before after J 
# - stats S 
# - Ability to run per frame / per scene / entire directory S  
# - expanding bounding boxes J