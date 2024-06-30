import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 将Matplotlib后端设置为非交互式模式


class Visualizer:
    def __init__(self, R_gt, t_gt, R_pr, t_pr, img_path, mesh_path):
        '''
        R_gt: 3x3，真实的旋转矩阵
        t_gt: 3x1，真实的平移向量
        R_pr: 3x3，预测的旋转矩阵
        t_pr: 3x1，预测的平移向量
        img_path: 图片的路径
        mesh_path: mesh(.ply)的路径
        '''
        
        self.img_path = img_path
        self.mesh_path = mesh_path 
        
        self.Rt_gt = np.concatenate((R_gt, t_gt), axis=1)  # [3, 4]  
        self.Rt_pr = np.concatenate((R_pr, t_pr), axis=1)  # [3, 4] 
        
        # 图片的宽和高
        self.im_width = 640
        self.im_height = 480
        
        # 相机内参(intrinsic_calibration)
        self.fx = 572.4114
        self.fy = 573.5704
        self.cx = 325.2611
        self.cy = 242.0489
        self.K = np.array([[self.fx, 0.0    , self.cx ], 
                           [0.0    , self.fy, self.cy ],
                           [0.0    , 0.0    , 1.0     ]])
        
        # 代表立方体的12条边，其中每个元素为两个顶点的索引 
        self.edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
        
        # 获取顶点
        self.vertices = self.get_meshply_vertices(mesh_path)  # [顶点数量, 3]  其中每一行代表一个顶点，每一列代表xyz坐标    
        
        
    def get_meshply_vertices(self, mesh_path):
        '''
        mesh_path: mesh(.ply)的路径
        '''
        vertices = []
        vertex_mode = False
        nb_vertices = 0
        idx = 0
        
        # 打开mesh_path文件
        with open(mesh_path, 'r') as mesh_file:
            for line in mesh_file:
                elements = line.split()
                if vertex_mode:
                    vertices.append([float(i) for i in elements[:3]])
                    idx += 1
                    if idx == nb_vertices:
                        break
                elif elements[0] == 'element':
                        if elements[1] == 'vertex':
                            nb_vertices = int(elements[2])
                elif elements[0] == 'end_header':
                        vertex_mode = True
                        
        return vertices
    
    def get_3D_corners(self, vertices):   
        min_x = np.min(vertices[0,:])
        max_x = np.max(vertices[0,:])
        min_y = np.min(vertices[1,:])
        max_y = np.max(vertices[1,:])
        min_z = np.min(vertices[2,:])
        max_z = np.max(vertices[2,:])
        corners = np.array([[min_x, min_y, min_z],
                            [min_x, min_y, max_z],
                            [min_x, max_y, min_z],
                            [min_x, max_y, max_z],
                            [max_x, min_y, min_z],
                            [max_x, min_y, max_z],
                            [max_x, max_y, min_z],
                            [max_x, max_y, max_z]])
        
        corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
        return corners   # [4, 8]  其中每一列代表一个顶点，前三行代表xyz坐标，最后一行代表齐次坐标
    
    def compute_projection(self, points_3D, transformation, intrinsic_calibration):
        '''
        这个函数用于计算3D点在图像上的投影
        '''
        projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
        camera_projection = (intrinsic_calibration.dot(transformation)).dot(points_3D)
        projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
        projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
        
        return projections_2d
    
    # def visualize(self):
    #     vertices  = np.c_[np.array(self.vertices), np.ones((len(self.vertices), 1))].transpose()  # [4, 顶点数量]  其中每一列代表一个顶点，前三行代表xyz坐标，最后一行代表齐次坐标
    #     corners3D = self.get_3D_corners(vertices)  # [4, 8]  其中每一列代表一个顶点，前三行代表xyz坐标，最后一行代表齐次坐标
    #     proj_corners_gt = np.transpose(self.compute_projection(corners3D, self.Rt_gt, self.K)) 
    #     proj_corners_pr = np.transpose(self.compute_projection(corners3D, self.Rt_pr, self.K))
        
    #     image = Image.open(self.img_path)
    #     img = np.array(image)
    #     plt.xlim((0, self.im_width))
    #     plt.ylim((0, self.im_height))
    #     plt.imshow(resize(img, (self.im_height, self.im_width)))
        
    #     for edge in self.edges_corners:
    #         plt.plot(proj_corners_gt[edge, 0], proj_corners_gt[edge, 1], color='r', linewidth=2.0)
    #         plt.plot(proj_corners_pr[edge, 0], proj_corners_pr[edge, 1], color='w', linewidth=2.0)
            
    #     plt.gca().invert_yaxis()
    #     plt.savefig('/home/xietao/fanxuan/FFB6D-master/your_image_filename.png')
    #     plt.show()
    def visualize(self):
        vertices = np.c_[np.array(self.vertices), np.ones((len(self.vertices), 1))].transpose()  # [4, 顶点数量]  其中每一列代表一个顶点，前三行代表xyz坐标，最后一行代表齐次坐标
        corners3D = self.get_3D_corners(vertices)  # [4, 8]  其中每一列代表一个顶点，前三行代表xyz坐标，最后一行代表齐次坐标
        proj_corners_gt = np.transpose(self.compute_projection(corners3D, self.Rt_gt, self.K)) 
        proj_corners_pr = np.transpose(self.compute_projection(corners3D, self.Rt_pr, self.K))
        
        image = Image.open(self.img_path)
        img = np.array(image)
        plt.xlim((0, self.im_width))
        plt.ylim((0, self.im_height))
        plt.imshow(resize(img, (self.im_height, self.im_width)))
        plt.axis('off')
        
        for edge in self.edges_corners:
            plt.plot(proj_corners_gt[edge, 0], proj_corners_gt[edge, 1], color='r', linewidth=2.0)
            plt.plot(proj_corners_pr[edge, 0], proj_corners_pr[edge, 1], color='w', linewidth=2.0)
         
        plt.gca().invert_yaxis()
        plt.savefig('/home/xietao/fanxuan/FFB6D-master/ffb6d/train_log/linemod/visual_class/our/driller_1178.png', bbox_inches='tight', pad_inches=0)
        
        

  
 
if __name__ == '__main__':
    R_gt = np.array([0.98548597, -0.00825023, -0.16955499, -0.13048200, -0.67573500, -0.72550398, -0.10858900, 0.73709798, -0.66700399]).reshape(3, 3)
    t_gt = np.array([27.85431770, -110.12161613, 1023.44225463]).reshape(3, 1)
    R_pr = np.array([0.98649596, -0.00625023, -0.16755487, -0.13048200, -0.65573500, -0.71550398, -0.10858922, 0.74709798, -0.65700399]).reshape(3, 3)
    t_pr = np.array([26.83431680, -110.22161723, 1022.44225463]).reshape(3, 1)
    img_path = '/home/xietao/xt_dataset/linemod/Linemod_preprocessed/data/08/rgb/0000.png'
    mesh_path = '/home/xietao/xt_dataset/linemod/Linemod_preprocessed/models/obj_08.ply'
    
    visualizer = Visualizer(R_gt, t_gt, R_pr, t_pr, img_path, mesh_path)
    visualizer.visualize()
                             
                                  
        


