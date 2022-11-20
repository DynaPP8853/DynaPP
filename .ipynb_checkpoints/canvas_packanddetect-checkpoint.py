# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from utils.general import box_inter, box_iou
import copy
import time

# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import cv2
import sys

def check_intersect(boxA, boxB):
    ''' 두개의 box가 서로 교집합을 가지는지 check한다.
    args :
        boxA : boxA 좌표 [xmin,ymin,xmax,ymax 순]
        boxB : boxB 좌표
    '''
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea > 0:
        return True
    else:
        return False

def canvas(original_img, boxes, where_old_image=None,where_new_image=None, prev_out=None, background = 0.3):
    if where_old_image==None:

        where_old_image, groups_groups = patch_construction(original_img, boxes, prev_out=prev_out, background=background)
        new_bboxes = where_old_image
    elif len(where_old_image)==1:
        x, y, z, w = where_old_image[0,1], where_old_image[0,3], where_old_image[0,0], where_old_image[0,2]
        return original_img[:,:,x:y, z:w], where_old_image, None, None
    else:
        w_max, h_max = torch.max(where_new_image[:,2:4], dim=0)[0]
        img_comb = torch.full((original_img.size(0), original_img.size(1), h_max, w_max), 0.).to(original_img.device).half()  # base image with 4 tiles
        for i in range(len(where_old_image)):
            img_comb[:,:,where_new_image[i,1]:where_new_image[i,3], where_new_image[i,0]:where_new_image[i,2]] = original_img[:,:, where_old_image[i,1]:where_old_image[i,3],where_old_image[i,0]:where_old_image[i,2]]
        return img_comb, where_old_image, where_new_image, None
    
     
    '''pack and detect 에서 pack을 구현한다.
    args :
        dets : inference_detector 추론의 결과이다.
        cf_jpg_path : current frame의 경로이다. (jpeg확장자)
    return:
        pack을 진행하여 새로 만든 이미지 patch halfxhalf를 반환 (numpy array)
        pack이 불가능한 경우에는 False를 반환
    '''

 
    width_max = 0
    height_max = 0
    # new_bboxes = new_bboxes.detach().cpu().numpy()
    full = original_img.size(3)
    half = original_img.size(3)//2
     
    # pack procedure
    # find the largest dimension of all bounding boxes

    width_max = np.max(new_bboxes[:,2]-new_bboxes[:,0])
    height_max = np.max(new_bboxes[:,3]-new_bboxes[:,1])
    
    if max(height_max,width_max) > half:
        return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
    
    if height_max > width_max:
        extend = "horizontal"
    else:
        extend = "vertical"
    
    patch_bboxes = []
    patch = torch.zeros([1,3,half,half]).cuda().half()
#     print(f"extend : {extend}, len(new_bboxes): {len(new_bboxes)}\nnew_bboxes : {new_bboxes}")
    # largest dimension이 height인 경우 가로로 extend한다.
    if extend == 'horizontal':
        
        # height_max, width_max 모두 half을 넘지 않기 때문에 box가 하나라면 Pack이 가능한다.
        if len(new_bboxes) == 1:
            xmin = int(new_bboxes[0][0])
            ymin = int(new_bboxes[0][1]) 
            xmax = int(new_bboxes[0][2])
            ymax = int(new_bboxes[0][3])
#             w = xmax - xmin
#             h = ymax - ymin
            ##########################scl extend here ##########################
            # 1. 1개뿐이니 halfxhalf 만족
            # 2. center를 기준으로 +- half//2로 늘리는데 바운딩박스가 초과하면 밀어서 조정  
            
            center_x = (xmin+xmax)//2
            center_y = (ymin+ymax)//2
            xmin = center_x-half//2
            ymin = center_y-half//2
            xmax = center_x+half//2
            ymax = center_y+half//2 # half//2씩으로 확장
            
            if xmin<0:
                xmax = half
                xmin=0
            if xmax>full:
                xmin = half
                xmax=full
            if ymin<0:
                ymax = half           # 모퉁이 부분은 밀어서 0-half
                ymin=0
            if ymax>full:
                ymin = half
                ymax= full            # 모퉁이 부분은 밀어서 half-full
            ##################################################################
            patch[:,:,:half, :half] = original_img[:,:,ymin:ymax, xmin:xmax]

            return patch, torch.tensor([[xmin,ymin,xmax,ymax]]).cuda(), None, None
        
        # horizontal하게 두 box를 배치하였을 때 half이 넘으면 안된다.
        elif len(new_bboxes) == 2:
            box1 = list(map(int, new_bboxes[0]))
            box2 = list(map(int, new_bboxes[1]))
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            if w1 + w2 > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]

            ##########################scl extend here ########################## [0]: xmin [1]: ymin [2]: xmax [3]: ymax
            
            center_x1 = (box1[0]+box1[2])//2
            center_y1 = (box1[1]+box1[3])//2
            center_x2 = (box2[0]+box2[2])//2
            center_y2 = (box2[1]+box2[3])//2  #중심찾기
            
            # 확장 가능한 extra 양 찾기
            extra_w_whole = half - (w1 + w2)   # 가로로 두개가 배치하니 둘다 빼줌
            extra_h_left_whole = half - h1     # 세로로는 한개가 배치되니
            extra_h_right_whole = half - h2    
        
            extra_w = make_extra(extra_w_whole, box_num=2) # 얼만큼 가로로 확장할 수 있는지 [box1 왼, box1 오, box2 왼, box2 오]
            extra_h_left = make_extra(extra_h_left_whole, box_num=1)  # 얼만큼 세로로 확장할 수 있는지 [box1 위, box1 아], left란 뜻은                                                                                                                           이미지의 왼쪽에 위치하기에
            extra_h_right = make_extra(extra_h_right_whole, box_num=1) # 위와 같이 box2
            extra_h = extra_h_left+extra_h_right

            
            # 변화된 bbox
            box1[0],box1[2], box2[0], box2[2] = greedy_intersect_extend(extra_w, [box1[0],box1[2], box2[0], box2[2]],\
                                                                       [box1[1],box1[3], box2[1], box2[3]], full=full) 
            box1[1],box1[3], box2[1], box2[3] = greedy_intersect_extend(extra_h,[box1[1],box1[3], box2[1], box2[3]],\
                                                                       [box1[0],box1[2], box2[0], box2[2]], full=full)
                
            w1 = box1[2]-box1[0] # 다시 계산
            w2 = box2[2]-box2[0]
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            #################################################################################################3
            try:
                patch[:,:,:h1,:w1] = original_img[:,:,box1[1]:box1[3], box1[0]:box1[2]]
                patch[:,:,:h2,w1:w1+w2] = original_img[:,:,box2[1]:box2[3], box2[0]:box2[2]]
            except:
                
                print(f"{extend} {len(new_bboxes)}")
                print(f"h : {h} w : {w}")
                print(f"original_img : {original_img.shape}")
                print(f"af_bboxes : {af_bboxes}")
                print(f"error : {sys.exc_info()}")
                print(f"original_img.shape : {original_img.shape}")
                print(f"box1 : {box1}")
                print(f"box2 : {box2}")
                print(f"w1 : {w1} h1 : {h1}")
                print(f"w2 : {w2} h2 : {h2}")
                
                
            where_new_image = torch.tensor([[0,0,w1,h1],
                                [w1,0,w1+w2,h2]])
            where_old_image = torch.tensor([[box1[0],box1[1],box1[2],box1[3]],
                                [box2[0],box2[1],box2[2],box2[3]]])
            return patch, where_old_image.cuda(), where_new_image.cuda(), None
         
        # height가 가장 큰 box를 배치하고 나머지 두 box를 세로로 배치한다.
        elif len(new_bboxes) == 3:
            # sort by height descending order
            sorted_bboxes = sorted(new_bboxes, key=lambda k: (k[3]-k[1]), reverse = True)
#             print(f"sorted_bboxes : {sorted_bboxes}")
            box1 = list(map(int, sorted_bboxes[0]))
            box2 = list(map(int, sorted_bboxes[1]))
            box3 = list(map(int, sorted_bboxes[2]))
            # height가 가장 큰 box의 width와 나머지 중 width가 큰 box의 width의 합이 half을 넘으면 안된다.
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            if  w1 + max(w2,w3) > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None

            # heigh가 가장 큰 box외에 나머지 두 box의 height의 합이 half을 넘으면 안된다.
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
            if h2 + h3 > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            
            #########################scl extend here #############################################
            # 확장 가능한 extra 양 찾기
            extra_w_whole = half - (w1 + max(w2,w3)) # 가로로 box1과 max(box2,box3)의 길이가 제일 기므로
            extra_h_left_whole = half - h1 # 세로로는 혼자
            extra_h_right_whole = half - (h2 + h3) # 세로로 box2,box3
            
            # more는 max보다 작은 게 있을경우 max만큼 추가 확장이 가능하니 그부분만큼의 range를 더 열어주는 것 
            extra_w = make_extra(extra_w_whole, box_num=2)
            extra_w = extra_w + extra_w[2:4] # [box1 왼, box1 오, box2 왼, box2 오,box3 왼, box3 오]
            more_w = make_extra(abs(w2-w3), box_num=1) # 둘의 차이만큼 더 작은 쪽에 확장가능

            if w2>w3:
                extra_w[4] += more_w[0]
                extra_w[5] += more_w[1]     # box 3에 더해줌
            else:
                extra_w[2] += more_w[0]
                extra_w[3] += more_w[1]     # box 2에 더해줌
                
            extra_h_left = make_extra(extra_h_left_whole, box_num=1) # box1 세로 느는 양
            extra_h_right = make_extra(extra_h_right_whole, box_num=2) #box 2&3 세로 느는 양
            extra_h = extra_h_left+extra_h_right
            
            # 변화된 bbox
            box1[0], box1[2], box2[0], box2[2], box3[0], box3[2] = \
                        greedy_intersect_extend(extra_w,  [box1[0], box1[2], box2[0], box2[2], box3[0], box3[2]],\
                                               [box1[1], box1[3], box2[1], box2[3], box3[1], box3[3]], full=full)
            box1[1], box1[3], box2[1], box2[3], box3[1], box3[3] = \
                        greedy_intersect_extend(extra_h,  [box1[1], box1[3], box2[1], box2[3], box3[1], box3[3]],\
                                                [box1[0], box1[2], box2[0], box2[2], box3[0], box3[2]], full=full)
            
    
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
#             print(f"box1 : {box1}, box2 : {box2}, box3 : {box3}")
            #################################################################################################3
            try:
                patch[:,:,:h1, :w1] = original_img[:,:,box1[1]:box1[3], box1[0]:box1[2]]
                patch[:,:,:h2, w1:w1+w2] = original_img[:,:,box2[1]:box2[3], box2[0]:box2[2]]
                patch[:,:,h2:h2+h3, w1:w1+w3] = original_img[:,:,box3[1]:box3[3], box3[0]:box3[2]]
            except:
                
                print(f"{extend} {len(new_bboxes)}")
                print(f"h : {h} w : {w}")
                print(f"original_img : {original_img.shape}")
                print(f"af_bboxes : {af_bboxes}")
                print(f"error : {sys.exc_info()}")
                print(f"original_img.shape : {original_img.shape}")
                print(f"box1 : {box1}")
                print(f"box2 : {box2}")
                print(f"box3 : {box3}")
                a = input()
#                 print(f"w1 : {w1} h1 : {h1}")
#                 print(f"w2 : {w2} h2 : {h2}")
#                 a = input()
            where_old_image = torch.tensor([[box1[0],box1[1],box1[2],box1[3]],
                                [box2[0],box2[1],box2[2],box2[3]],
                                [box3[0],box3[1],box3[2],box3[3]]])

            where_new_image = torch.tensor([[0,0,w1,h1],
                                [w1,0,w1+w2,h2],
                                [w1,h2,w1+w3,h2+h3]])
            

            return patch, where_old_image.cuda(), where_new_image.cuda(), None
        
        # height가 큰 순서대로 1,3번째, 2,4번째 box들을 같은 column에 배치한다.
        elif len(new_bboxes) == 4:
            sorted_bboxes = sorted(new_bboxes, key=lambda k: (k[3]-k[1]), reverse = True)
            box1 = list(map(int, sorted_bboxes[0]))
            box2 = list(map(int, sorted_bboxes[1]))
            box3 = list(map(int, sorted_bboxes[2]))
            box4 = list(map(int, sorted_bboxes[3]))
            
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
            h4 = box4[3]-box4[1]
            
            if h1 + h3 > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None

            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            w4 = box4[2]-box4[0]
            
            if max(w1,w3)+max(w2,w4) > half:  # 아래 배치할 때 확장된 것을 고려해 식을 바꿈 (height 순서가 바뀔수 있다)
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None

            ################################## scl extend here #########################################
            extra_w_whole = half - (max(w1,w3) + max(w2,w4)) # 가로로 배치할때 세로 나열 묶음중 큰 값을 뺀다
            extra_h_left_whole = half - (h1+h3)              # box1,3 세로로 배치
            extra_h_right_whole = half - (h2+h4)             # box2,4 세로로 배치
            
            # 가로 확장 계산
            extra_w = make_extra(extra_w_whole, box_num=2)
            extra_w = extra_w+extra_w
            more_w_left = make_extra(abs(w1-w3), box_num=1)
            more_w_right = make_extra(abs(w2-w4), box_num=1)
            
            # extra로 더 묶음 중 작은 부분에게 더 늘릴 수 있는 부분 추가
            if w1>w3:
                extra_w[4] += more_w_left[0]
                extra_w[5] += more_w_left[1]
            else:
                extra_w[0] += more_w_left[0]
                extra_w[1] += more_w_left[1]
             
            if w2>w4:
                extra_w[6] += more_w_right[0]
                extra_w[7] += more_w_right[1]
            else:
                extra_w[2] += more_w_right[0]
                extra_w[3] += more_w_right[1]
                
            # 세로 확장 계산
            extra_h_left = make_extra(extra_h_left_whole, box_num=2)
            extra_h_right = make_extra(extra_h_right_whole, box_num=2)
            extra_h = extra_h_left[0:2]+extra_h_right[0:2]+extra_h_left[2:4]+extra_h_right[2:4]
            
            
            # 새로운 bbox
            box1[0], box1[2], box2[0], box2[2], box3[0], box3[2], box4[0], box4[2] = \
                        greedy_intersect_extend(extra_w,  [box1[0], box1[2], box2[0], box2[2], box3[0], box3[2], box4[0], box4[2]],\
                                                [box1[1], box1[3], box2[1], box2[3], box3[1], box3[3], box4[1], box4[3]], full=full)
            box1[1], box1[3], box2[1], box2[3], box3[1], box3[3], box4[1], box4[3] = \
                        greedy_intersect_extend(extra_h,  [box1[1], box1[3], box2[1], box2[3], box3[1], box3[3], box4[1], box4[3]],\
                                               [box1[0], box1[2], box2[0], box2[2], box3[0], box3[2], box4[0], box4[2]], full=full)
                        
            
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
            h4 = box4[3]-box4[1]  
            
            
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            w4 = box4[2]-box4[0]
            #################################################################
            
            
            patch[:,:,:h1, :w1] = original_img[:,:,box1[1]:box1[3], box1[0]:box1[2]]
            patch[:,:,h1:h1+h3, :w3] = original_img[:,:,box3[1]:box3[3], box3[0]:box3[2]]
            patch[:,:,:h2, max(w1,w3):max(w1,w3)+w2] = original_img[:,:,box2[1]:box2[3], box2[0]:box2[2]]
            patch[:,:,h2:h2+h4, max(w1,w3):max(w1,w3)+w4] = original_img[:,:,box4[1]:box4[3], box4[0]:box4[2]]
            # (height 순서가 바뀔수 있다), 
            # 이때 box1바로 옆 box2, box3 바로 옆 box4를 배치시엔 문제 생길수있다 
            # 따라서 가로방향으로 box1,box3의 max포인트 지점부터 box2,box4가 배치됨
            
            where_new_image = torch.tensor([[0,0,w1,h1],
                                [max(w1,w3),0,max(w1,w3)+w2,h2],
                                [0,h1,w3,h1+h3],
                                [max(w1,w3),h2,max(w1,w3)+w4,h2+h4]])
            
            where_old_image = torch.tensor([[box1[0],box1[1],box1[2],box1[3]],
                                [box2[0],box2[1],box2[2],box2[3]],
                                [box3[0],box3[1],box3[2],box3[3]],
                                [box4[0],box4[1],box4[2],box4[3]]])


            return patch, where_old_image.cuda(), where_new_image.cuda(), None
        # more than 5 boxes
        else:
            return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None



    elif extend == 'vertical':
        if len(new_bboxes) == 1:
            # height_max, width_max 모두 half을 넘지 않기 때문에 box가 하나라면 Pack이 가능하다.
            xmin = int(new_bboxes[0][0])
            ymin = int(new_bboxes[0][1])
            xmax = int(new_bboxes[0][2])
            ymax = int(new_bboxes[0][3])
#             w = xmax - xmin
#             h = ymax - ymin
            ##########################scl extend here ##########################
            # 1. 1개뿐이니 halfxhalf 만족
            # 2. center를 잡고 늘리는데 모퉁이라 바운딩박스가 초과하면 밀어서 조정            
            center_x = (xmin+xmax)//2
            center_y = (ymin+ymax)//2
            xmin = center_x-half//2
            ymin = center_y-half//2
            xmax = center_x+half//2
            ymax = center_y+half//2
            if xmin<0:
                xmax = half
                xmin=0
            if xmax>full:
                xmin = half
                xmax=full
            if ymin<0:
                ymax = half
                ymin=0
            if ymax>full:
                ymin = half
                ymax= full            
            ##################################################################
            patch[:,:,:half, :half] = original_img[:,:,ymin:ymax, xmin:xmax]

    
            return patch, torch.tensor([[xmin,ymin,xmax,ymax]]).cuda(), None, None
        
        # 두 row에 각각의 box를 배치한다. 
        elif len(new_bboxes) == 2:
            box1 = list(map(int, new_bboxes[0]))
            box2 = list(map(int, new_bboxes[1]))
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            # 높이의 합이 half이 넘으면 안된다.
            if h1 + h2 > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]

            ####################### scl extend here ##############################
            center_x1 = (box1[0]+box1[2])//2
            center_y1 = (box1[1]+box1[3])//2
            center_x2 = (box2[0]+box2[2])//2
            center_y2 = (box2[1]+box2[3])//2
            
            extra_h_whole = half - (h1 + h2)
            extra_w_up_whole = half - w1
            extra_w_down_whole = half - w2
        
            extra_h = make_extra(extra_h_whole, box_num=2)
            extra_w_up = make_extra(extra_w_up_whole, box_num=1)
            extra_w_down = make_extra(extra_w_down_whole, box_num=1)
            extra_w = extra_w_up+extra_w_down


            box1[1],box1[3], box2[1], box2[3] = greedy_intersect_extend(extra_h,[box1[1],box1[3], box2[1], box2[3]],\
                                                                        [box1[0],box1[2], box2[0], box2[2]], full=full)            
            box1[0],box1[2], box2[0], box2[2] = greedy_intersect_extend(extra_w, [box1[0],box1[2], box2[0], box2[2]],\
                                                                       [box1[1],box1[3], box2[1], box2[3]], full=full)
                
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
              
            ###################################################################33
            patch[:,:,:h1,:w1] = original_img[:,:,box1[1]:box1[3], box1[0]:box1[2]]
            patch[:,:,h1:h1+h2,:w2] = original_img[:,:,box2[1]:box2[3], box2[0]:box2[2]]
            
            where_new_image = torch.tensor([[0,0,w1,h1],
                                [0,h1,w2,h1+h2]])
            where_old_image =  torch.tensor([[box1[0],box1[1],box1[2],box1[3]],
                                [box2[0],box2[1],box2[2],box2[3]]])
             

            return patch, where_old_image.cuda(), where_new_image.cuda(), None
        
        # 가장 가로길이가 긴 box를 한 row에 배치하고 나머지 두 box를 다른 row에 배치한다.
        elif len(new_bboxes) == 3:
            # sort by width
            sorted_bboxes = sorted(new_bboxes, key=lambda k: (k[2]-k[0]), reverse = True)
            box1 = list(map(int, sorted_bboxes[0]))
            box2 = list(map(int, sorted_bboxes[1]))
            box3 = list(map(int, sorted_bboxes[2]))
            # width가 가장 큰 box와 나머지 중 height가 가장 큰 box의 height의 합이 half을 넘으면 안된다.
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
            if h1 + max(h2,h3) > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            # width가 가장 큰 box외에 나머지 두 box의 width의 합이 half을 넘으면 안된다.
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            if w2 + w3 > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            
            box1 = list(map(int, sorted_bboxes[0]))
            box2 = list(map(int, sorted_bboxes[1]))
            box3 = list(map(int, sorted_bboxes[2]))
            
            ################### scl extend here ###########################
            extra_h_whole = half - (h1 + max(h2,h3))
            extra_w_up_whole = half - w1
            extra_w_down_whole = half - (w2 + w3)
        
            extra_h = make_extra(extra_h_whole, box_num=2)
            extra_h = extra_h + extra_h[2:4]
            more_h = make_extra(abs(h2-h3), box_num=1)
            
            if h2>h3:
                extra_h[4] += more_h[0]
                extra_h[5] += more_h[1]
            else:
                extra_h[2] += more_h[0]
                extra_h[3] += more_h[1]     
                   
            extra_w_up = make_extra(extra_w_up_whole, box_num=1)
            extra_w_down = make_extra(extra_w_down_whole, box_num=2)
            extra_w = extra_w_up+extra_w_down
            

            box1[1], box1[3], box2[1], box2[3], box3[1], box3[3] = \
                        greedy_intersect_extend(extra_h,  [box1[1], box1[3], box2[1], box2[3], box3[1], box3[3]],
                                               [box1[0], box1[2], box2[0], box2[2], box3[0], box3[2]], full=full)            
            box1[0], box1[2], box2[0], box2[2], box3[0], box3[2] = \
                        greedy_intersect_extend(extra_w,  [box1[0], box1[2], box2[0], box2[2], box3[0], box3[2]],\
                                               [box1[1], box1[3], box2[1], box2[3], box3[1], box3[3]], full=full)

            
    
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]

            #################################################################################################3            
            patch[:,:,:h1,:w1] = original_img[:,:,box1[1]:box1[3], box1[0]:box1[2]]
            patch[:,:,h1:h1+h2,:w2] = original_img[:,:,box2[1]:box2[3], box2[0]:box2[2]]
            patch[:,:,h1:h1+h3, w2:w2+w3] = original_img[:,:,box3[1]:box3[3], box3[0]:box3[2]]
            
            where_new_image = torch.tensor([[0,0,w1,h1],
                                [0,h1,w2,h1+h2],
                                [w2,h1,w2+w3,h1+h3]])

            where_old_image = torch.tensor([[box1[0],box1[1],box1[2],box1[3]],
                                [box2[0],box2[1],box2[2],box2[3]],
                                [box3[0],box3[1],box3[2],box3[3]]])


            return patch, where_old_image.cuda(), where_new_image.cuda(), None
        
        # width가 큰 순서대로 1,3번째, 2,4번째 box들을 같은 row에 배치한다.
        elif len(new_bboxes) == 4:
            sorted_bboxes = sorted(new_bboxes, key=lambda k: (k[2]-k[0]), reverse = True)
            box1 = list(map(int, sorted_bboxes[0]))
            box2 = list(map(int, sorted_bboxes[1]))
            box3 = list(map(int, sorted_bboxes[2]))
            box4 = list(map(int, sorted_bboxes[3]))
            # width 기준으로 정렬하였을 때 1,3번째로 width가 큰 box, 2,4번째로 width가 큰 box가 같은 row에 배치된다.
            # 이 때 두 width의 합이 half을 넘으면 안된다.
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            w4 = box4[2]-box4[0]
            if w1 + w3 > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            # width 기준으로 정렬하였을 때 1,3번째로 width가 큰 box, 2,4번째로 width가 큰 box가 같은 row에 배치된다.
            # 이 때 두 row의 height의 합이 half이 넘으면 안된다.
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
            h4 = box4[3]-box4[1]
            if max(h1,h3)+max(h2,h4) > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            
            ################################## scl extend here #########################################
            extra_h_whole = half - (max(h1,h3) + max(h2,h4))
            extra_w_up_whole = half - (w1+w3)
            extra_w_down_whole = half - (w2+w4)
            
            
            extra_h = make_extra(extra_h_whole, box_num=2)
            extra_h = extra_h+extra_h
            more_h_up = make_extra(abs(h1-h3), box_num=1)
            more_h_down = make_extra(abs(h2-h4), box_num=1)
            if h1>h3:
                extra_h[4] += more_h_up[0]
                extra_h[5] += more_h_up[1]
            else:
                extra_h[0] += more_h_up[0]
                extra_h[1] += more_h_up[1]
             
            if h2>h4:
                extra_h[6] += more_h_down[0]
                extra_h[7] += more_h_down[1]
            else:
                extra_h[2] += more_h_down[0]
                extra_h[3] += more_h_down[1]
                
                
            extra_w_up = make_extra(extra_w_up_whole, box_num=2)
            extra_w_down = make_extra(extra_w_down_whole, box_num=2)
            extra_w = extra_w_up[0:2]+extra_w_down[0:2]+extra_w_up[2:4]+extra_w_down[2:4]
            
            
            box1[1], box1[3], box2[1], box2[3], box3[1], box3[3], box4[1], box4[3] = \
                        greedy_intersect_extend(extra_h,  [box1[1], box1[3], box2[1], box2[3], box3[1], box3[3], box4[1], box4[3]],\
                                               [box1[0], box1[2], box2[0], box2[2], box3[0], box3[2], box4[0], box4[2]], full=full)
                                    
            box1[0], box1[2], box2[0], box2[2], box3[0], box3[2], box4[0], box4[2] = \
                        greedy_intersect_extend(extra_w,  [box1[0], box1[2], box2[0], box2[2], box3[0], box3[2], box4[0], box4[2]],\
                                               [box1[1], box1[3], box2[1], box2[3], box3[1], box3[3], box4[1], box4[3]], full=full)

            
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
            h4 = box4[3]-box4[1]  
            
            
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            w4 = box4[2]-box4[0]
            ####################################################################3
            
            patch[:,:,:h1, :w1] = original_img[:,:,box1[1]:box1[3], box1[0]:box1[2]]
            patch[:,:,:h3, w1:w1+w3] = original_img[:,:,box3[1]:box3[3], box3[0]:box3[2]]
            patch[:,:,max(h1,h3):max(h1,h3)+h2, :w2] = original_img[:,:,box2[1]:box2[3], box2[0]:box2[2]]
            patch[:,:,max(h1,h3):max(h1,h3)+h4, w2:w2+w4] = original_img[:,:,box4[1]:box4[3], box4[0]:box4[2]]
         
            where_new_image = torch.tensor([[0,0,w1,h1],
                                [0,max(h1,h3),w2,max(h1,h3)+h2],
                                [w1,0,w1+w3,h3],
                                [w2,max(h1,h3),w2+w4,max(h1,h3)+h4]])
            where_old_image = torch.tensor([[box1[0],box1[1],box1[2],box1[3]],
                                [box2[0],box2[1],box2[2],box2[3]],
                                [box3[0],box3[1],box3[2],box3[3]],
                                [box4[0],box4[1],box4[2],box4[3]]])
            

            return patch, where_old_image.cuda(), where_new_image.cuda(), None
        # box가 5개 이상일 때
        else:
            return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None


def intersect_extend(extra, box_first_small,box_first_big, box_second_small, box_second_big):
    
    limitation_dic = {'first_min_lim': box_first_small, 'first_max_lim': (box_first_big+box_second_small)//2-box_first_big, \
            'second_min_lim': box_second_small - (box_second_small+box_first_big)//2, 'second_max_lim': full-box_second_big}

    limitation_dic = {k: v for k, v in sorted(limitation_dic.items(), key=lambda item: item[1], reverse=False)}

    add_to_others = 0
    for i, key in enumerate(limitation_dic):
        value = limitation_dic[key]
        extra[i] += add_to_others//(4-i)
        add_to_others -= add_to_others//(4-i)
        if value >= extra[i]:
            value = extra[i]
        else:
            add_to_others += extra[i] - value
        limitation_dic[key] = value

    box_first_small -= limitation_dic['first_min_lim']
    box_first_big += limitation_dic['first_max_lim']
    box_second_small-= limitation_dic['second_min_lim']
    box_second_big += limitation_dic['second_max_lim']
    return box_first_small,box_first_big, box_second_small, box_second_big

def check_intersect_possibility(boxes_other_oreientation):
    '''box가 intersect인지 체크 (0: no intersect, 1 : intersect)
    args :
        boxes_other_oreientation : 다른방향의 박스포인트
        '''
    #안겹친다고 가정
    box_possible = [0]*(len(boxes_other_oreientation)//2)
    
    #겹칠확률 존재시 1
    for i in range(len(boxes_other_oreientation)//2):
        for j in range(len(boxes_other_oreientation)//2):
            if j!=i:
                if min(boxes_other_oreientation[2*j+1],boxes_other_oreientation[2*i+1])-\
                                                    max(boxes_other_oreientation[2*i],boxes_other_oreientation[2*j])>0:
                    box_possible[i] = 1
    return box_possible

def greedy_intersect_extend(extra, boxes, boxes_other_oreientation, full=960):
    '''pack and detect 에서 greedy한 extend의 변화된 bbox를 return한다.
    args :
        extra : boudning box 포인트마다의 확장 가능값
        boxes : [box1 xmin, box1 xmax, box2 xmin, box2 xmax, box3 xmin, box3 xmax ...]
        또는 [box1 ymin, box1 ymax, box2 ymin, box2 ymax, box3 ymin, box3 ymax ...]
        boxes_other_oreientation : 확장하려는 방향외에 다른방향의 포인트를 받아와서 이방향으로 확장 시 겹칠여부를 조사한다.
    return:
        확장을 진행하여 변화된 box의 포인트들을 반환
    '''
    
    box_intersect = check_intersect_possibility(boxes_other_oreientation) #겹칠여부반환
    box_dic = {}        # box point (boxes)
    extra_dic = {}      # 확장 가능한 값들 (extra)
    box_intersect_dic = {}
    # box_n번째_(min or max) ----> ex ) box2_0 : 2번째 box min값
    for  i in range(len(boxes)):
        box_dic[f'box{i//2}_{i%2}'] = boxes[i] 
        extra_dic[f'box{i//2}_{i%2}'] = extra[i]
        box_intersect_dic[f'box{i//2}_{i%2}'] = box_intersect[i//2]

    # 들어온 box끼리의 중심 순서 파악
    centers = []
    for  i in range(len(boxes)//2):
        centers.append((boxes[2*i]+boxes[2*i+1])//2)
    
    #위 순서대로 box_dic을 순서 change ---> ex) [box2,box1,box3] 순이라면 [box2 xmin, box2 xmax, box1 xmin, box1 xmax, box3 xmin, box3 xmax]
    center_order = {}
    for  i in range(len(centers)):
        center_order[f'{i}'] = centers[i]
    center_order = {k: v for k, v in sorted(center_order.items(), key=lambda item: item[1], reverse=False)}
    new_box_dic = {}
    for key in center_order:
        new_box_dic[f'box{int(key)}_0'] = box_dic[f'box{int(key)}_0']
        new_box_dic[f'box{int(key)}_1'] = box_dic[f'box{int(key)}_1']
    
    # 비교할 값 만들기, 즉 확장했을때 intersect 또는 이미지를 이탈할시엔 비교값을 복귀 (비교값은 box간의 중심과 0,full으로 이루어짐)
    # [0, (newbox_1,newbox_2), (newbox_2,newbox_3), ..., full]
    new_box_dic_values = list(new_box_dic.values())
    compares = []
    for index in range(len(new_box_dic_values)):
        if index ==0:
            compares.append(0)
        elif index %2==0:
            val = (new_box_dic_values[index]+new_box_dic_values[index-1])//2
            compares.append(val)
            compares.append(val)
    compares.append(full)

    # 겹칠 확률이 없는 박스에 대해선 이미지를 초과하는지만 체크하고 확장
    # 비교값이 확장방향을 역행한다면 그대로 납두기
    # 아니라면 비교값과 확장값을 비교해서 intersect 하면 비교값, 아니면 확장값
    for i,key in enumerate(new_box_dic):
        value = new_box_dic[key]
        move = extra_dic[key]
        if i %2 ==0:     
            if box_intersect_dic[key]==0:
                value = max(0,value-move)
                new_box_dic[key] = value                 
            else:
                if compares[i]<value:
                    value = max(compares[i],value-move)
                    new_box_dic[key] = value   
        else:
            if box_intersect_dic[key]==0:
                value = min(full,value+move)
                new_box_dic[key] = value                 
            else:
                if compares[i]>value:
                    value = min(compares[i],value+move)
                    new_box_dic[key] = value                          
              
    # 새로운 값으로 삽입
    for  i in range(len(boxes)):
        boxes[i] = new_box_dic[f'box{i//2}_{i%2}']

    return boxes

def clamp_to_zero(*args):
    new = []
    for i in args:
        new.append(max(0,i))
    return new

def clamp_to_max(*args):
    new = []
    for i in args:
        new.append(min(full,i))
    return new

def no_intersect_extend(extra, box_first_small,box_first_big, box_second_small, box_second_big):
    
    box_first_small -= extra[0]
    box_first_big += extra[1]
    box_second_small -= extra[2]
    box_second_big += extra[3]
    if box_first_small<0:
        box_first_big -= box_first_small
        box_first_small = 0
    if box_first_big>full:
        box_first_small += (full-box_first_big)
        box_first_big = full
    if box_second_small<0:
        box_second_big -= box_second_small
        box_second_small = 0
    if box_second_big>full:
        box_second_small += (full-box_second_big[2])
        box_second_big = full
    return box_first_small,box_first_big, box_second_small, box_second_big


def make_extra(extra_whole, box_num=2):
    '''확장 값들을 반환한다
    args :
        extra_whole : half 이미지에서 확장 가능 남은 총 pixel
        box_num : 원하는 방향의 나열되는 box의 개수
    return:
        box 개수의 두배만큼(boxmin 확장, boxmax 확장) 늘릴 양을 작은값부터 반환해준다
        ex ) box_num=2, extra_whole = 45 -->> [11,11,11,12]
    '''
    extra = []
    for i in range(box_num*2):
        numb = int(np.ceil(extra_whole/(box_num*2-i)))
        extra.insert(0, numb)
        extra_whole -= numb
    return extra


def patch_construction(img, boxes, prev_out=None, background=0.3):
    # Sort by width ratio
    for_sort_labels = boxes.clone()
    min_len, max_len = max(img.size(3), img.size(2))//33, max(img.size(3), img.size(2))//16
    for_sort_labels = expand_box(for_sort_labels, expand = background, thres_len= (min_len, max_len), x_size = img.size(3), y_size = img.size(2), prev_out=prev_out)
    for_sort_labels[:,[0,2]].clamp_(min=0, max=img.size(3))
    for_sort_labels[:,[1,3]].clamp_(min=0, max=img.size(2))
    inter_matrix = ((box_inter(for_sort_labels, for_sort_labels))*1).cpu().numpy()
    for_sort_labels = for_sort_labels.cpu().numpy()
    groups = [(i+np.nonzero(inter_matrix[i,i:])[0]).tolist() for i in range(len(inter_matrix))]
    
    finish =0
    while finish != len(groups)-1:
        group = groups[finish]
        to_add =[]
        for idx, check_group in enumerate(groups[finish+1:]):
            for num in group:
                if num in check_group:
                    to_add.append(finish+1+idx)
        to_add = list(set(to_add))
        if len(to_add) ==0:
            finish+=1
        else:
            for i in sorted(to_add, reverse=True):
                groups[finish].extend(copy.deepcopy(groups[i]))
                del groups[i]
            groups[finish] = list(set(groups[finish]))


    box_to_crop = [np.concatenate([np.min(for_sort_labels[group,0:2],axis=0, keepdims=True), np.max(for_sort_labels[group,2:4],axis=0, keepdims=True)], axis=1) for group in groups]
    new_box = np.concatenate(box_to_crop, axis=0)
    while True:
        temp = torch.from_numpy(new_box)
        inter_matrix = ((box_inter(temp, temp))*1).numpy()
        if np.sum(inter_matrix) == len(inter_matrix):
            break
        else:
            groups = [(i+np.nonzero(inter_matrix[i,i:])[0]).tolist() for i in range(len(inter_matrix))]
            finish =0
            while finish != len(groups)-1:
                group = groups[finish]
                to_add =[]
                for idx, check_group in enumerate(groups[finish+1:]):
                    for num in group:
                        if num in check_group:
                            to_add.append(finish+1+idx)
                to_add = list(set(to_add))
                if len(to_add) ==0:
                    finish+=1
                else:
                    for i in sorted(to_add, reverse=True):
                        groups[finish].extend(copy.deepcopy(groups[i]))
                        del groups[i]
                    groups[finish] = list(set(groups[finish]))
            box_to_crop = [np.concatenate([np.min(new_box[group,0:2],axis=0, keepdims=True), np.max(new_box[group,2:4],axis=0, keepdims=True)], axis=1) for group in groups]
            new_box = np.concatenate(box_to_crop, axis=0)


    # 묶을 그룹별로 정리
    new_box[:, [1,3]] = np.clip(new_box[:, [1,3]], 0, img.size(2))
    new_box[:, [0,2]] = np.clip(new_box[:, [0,2]], 0, img.size(3))
    return new_box.astype(np.int32), groups


def expand_box(boxes, expand , thres_len, x_size, y_size, prev_out):
    length_x = (boxes[:,2:3] - boxes[:,0:1])*expand/2
    length_y =  (boxes[:,3:4] - boxes[:,1:2])*expand/2
    
    boxes[:,0:1],boxes[:,1:2],boxes[:,2:3],boxes[:,3:4] = boxes[:,0:1]-length_x,boxes[:,1:2]-length_y,boxes[:,2:3]+length_x,boxes[:,3:4]+length_y
    return boxes


def check_intersect(boxA, boxB):
    ''' 두개의 box가 서로 교집합을 가지는지 check한다.
    args :
        boxA : boxA 좌표 [xmin,ymin,xmax,ymax 순]
        boxB : boxB 좌표
    '''
    xA = max(float(boxA[0].item()), float(boxB[0].item()))
    yA = max(float(boxA[1].item()), float(boxB[1].item()))
    xB = min(float(boxA[2].item()), float(boxB[2].item()))
    yB = min(float(boxA[3].item()), float(boxB[3].item()))
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea > 0:
        return True
    else:
        return False


def make_newboxes(boxes):
    box_num = len(boxes)
    while True:
        groups = {}
        for i in range(0,box_num):
            new = True
            for key, value in groups.items():
                if i in value:
                    new = False
                    save_key = key
            if new:
                save_key = i
                groups[i] = [i]
                
            for j in range(i+1, box_num):
                if check_intersect(boxes[i], boxes[j]):
                    groups[save_key].append(j)

        # find enclosing bounding box around the union bounding boxes in each connected component
        new_bboxes = []
        for idxs in groups.values():
            idxs = list(set(idxs))
            xmin = torch.min(boxes[idxs,0])
            xmax = torch.max(boxes[idxs,2])
            ymin = torch.min(boxes[idxs,1])
            ymax = torch.max(boxes[idxs,3])
            enclosing_bbox = torch.LongTensor([xmin, ymin, xmax, ymax]).cuda()
            new_bboxes.append(enclosing_bbox)
        new_bboxes = torch.vstack(new_bboxes)
        # check new bboxes intersect
        intersect = False
        box_num = len(new_bboxes)
        for i in range(0, box_num):
            for j in range(i+1, box_num):
                if check_intersect(new_bboxes[i], new_bboxes[j]):
                    intersect = True

        # if there are intersecting boxes in new_bboxes, build enclosing box again
        if intersect:
            boxes = new_bboxes
        else:
            break
    return new_bboxes
