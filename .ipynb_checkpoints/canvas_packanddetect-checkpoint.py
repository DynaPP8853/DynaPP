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

def patch_construction(img, boxes, prev_out=None, background=0.3):
    '''     
    <Description>
    This method is a patch construction.
    
    <input> 
    img : image
    boxes : bounding boxes
    prev_out : previous bounding boxes
    background : d% background
    
    <output>
    new_box : patches
    groups : groups of each contains bounding bax in the patches
    '''    
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


    # ?????? ???????????? ??????
    new_box[:, [1,3]] = np.clip(new_box[:, [1,3]], 0, img.size(2))
    new_box[:, [0,2]] = np.clip(new_box[:, [0,2]], 0, img.size(3))
    return new_box.astype(np.int32), groups


def canvas(original_img, boxes, where_old_image=None,where_new_image=None, prev_out=None, background = 0.3):
    '''     
    <Description>
    This method is forming Pack and Detect canvas. 
    
    <input> 
    original_img : image
    boxes : bounding boxes
    where_old_image : patch locations in the image
    where_new_image : patch locations in the canvas
    prev_out : previous bounding boxes
    background : d% background
    
    <output>
    packed frame. 
    patch locations in the image, 
    patch locations in the canvas, 
    groups of each contains bounding bax in the patches
    '''    
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
    
     
    '''pack and detect ?????? pack??? ????????????.
    args :
        dets : inference_detector ????????? ????????????.
        cf_jpg_path : current frame??? ????????????. (jpeg?????????)
    return:
        pack??? ???????????? ?????? ?????? ????????? patch halfxhalf??? ?????? (numpy array)
        pack??? ???????????? ???????????? False??? ??????
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
    # largest dimension??? height??? ?????? ????????? extend??????.
    if extend == 'horizontal':
        
        # height_max, width_max ?????? half??? ?????? ?????? ????????? box??? ???????????? Pack??? ????????????.
        if len(new_bboxes) == 1:
            xmin = int(new_bboxes[0][0])
            ymin = int(new_bboxes[0][1]) 
            xmax = int(new_bboxes[0][2])
            ymax = int(new_bboxes[0][3])
#             w = xmax - xmin
#             h = ymax - ymin
            ##########################scl extend here ##########################
            # 1. 1???????????? halfxhalf ??????
            # 2. center??? ???????????? +- half//2??? ???????????? ?????????????????? ???????????? ????????? ??????  
            
            center_x = (xmin+xmax)//2
            center_y = (ymin+ymax)//2
            xmin = center_x-half//2
            ymin = center_y-half//2
            xmax = center_x+half//2
            ymax = center_y+half//2 # half//2????????? ??????
            
            if xmin<0:
                xmax = half
                xmin=0
            if xmax>full:
                xmin = half
                xmax=full
            if ymin<0:
                ymax = half           # ????????? ????????? ????????? 0-half
                ymin=0
            if ymax>full:
                ymin = half
                ymax= full            # ????????? ????????? ????????? half-full
            ##################################################################
            patch[:,:,:half, :half] = original_img[:,:,ymin:ymax, xmin:xmax]

            return patch, torch.tensor([[xmin,ymin,xmax,ymax]]).cuda(), None, None
        
        # horizontal?????? ??? box??? ??????????????? ??? half??? ????????? ?????????.
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
            center_y2 = (box2[1]+box2[3])//2  #????????????
            
            # ?????? ????????? extra ??? ??????
            extra_w_whole = half - (w1 + w2)   # ????????? ????????? ???????????? ?????? ??????
            extra_h_left_whole = half - h1     # ???????????? ????????? ????????????
            extra_h_right_whole = half - h2    
        
            extra_w = make_extra(extra_w_whole, box_num=2) # ????????? ????????? ????????? ??? ????????? [box1 ???, box1 ???, box2 ???, box2 ???]
            extra_h_left = make_extra(extra_h_left_whole, box_num=1)  # ????????? ????????? ????????? ??? ????????? [box1 ???, box1 ???], left??? ??????                                                                                                                           ???????????? ????????? ???????????????
            extra_h_right = make_extra(extra_h_right_whole, box_num=1) # ?????? ?????? box2
            extra_h = extra_h_left+extra_h_right

            
            # ????????? bbox
            box1[0],box1[2], box2[0], box2[2] = greedy_intersect_extend(extra_w, [box1[0],box1[2], box2[0], box2[2]],\
                                                                       [box1[1],box1[3], box2[1], box2[3]], full=full) 
            box1[1],box1[3], box2[1], box2[3] = greedy_intersect_extend(extra_h,[box1[1],box1[3], box2[1], box2[3]],\
                                                                       [box1[0],box1[2], box2[0], box2[2]], full=full)
                
            w1 = box1[2]-box1[0] # ?????? ??????
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
         
        # height??? ?????? ??? box??? ???????????? ????????? ??? box??? ????????? ????????????.
        elif len(new_bboxes) == 3:
            # sort by height descending order
            sorted_bboxes = sorted(new_bboxes, key=lambda k: (k[3]-k[1]), reverse = True)
#             print(f"sorted_bboxes : {sorted_bboxes}")
            box1 = list(map(int, sorted_bboxes[0]))
            box2 = list(map(int, sorted_bboxes[1]))
            box3 = list(map(int, sorted_bboxes[2]))
            # height??? ?????? ??? box??? width??? ????????? ??? width??? ??? box??? width??? ?????? half??? ????????? ?????????.
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            if  w1 + max(w2,w3) > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None

            # heigh??? ?????? ??? box?????? ????????? ??? box??? height??? ?????? half??? ????????? ?????????.
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
            if h2 + h3 > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            
            #########################scl extend here #############################################
            # ?????? ????????? extra ??? ??????
            extra_w_whole = half - (w1 + max(w2,w3)) # ????????? box1??? max(box2,box3)??? ????????? ?????? ?????????
            extra_h_left_whole = half - h1 # ???????????? ??????
            extra_h_right_whole = half - (h2 + h3) # ????????? box2,box3
            
            # more??? max?????? ?????? ??? ???????????? max?????? ?????? ????????? ???????????? ?????????????????? range??? ??? ???????????? ??? 
            extra_w = make_extra(extra_w_whole, box_num=2)
            extra_w = extra_w + extra_w[2:4] # [box1 ???, box1 ???, box2 ???, box2 ???,box3 ???, box3 ???]
            more_w = make_extra(abs(w2-w3), box_num=1) # ?????? ???????????? ??? ?????? ?????? ????????????

            if w2>w3:
                extra_w[4] += more_w[0]
                extra_w[5] += more_w[1]     # box 3??? ?????????
            else:
                extra_w[2] += more_w[0]
                extra_w[3] += more_w[1]     # box 2??? ?????????
                
            extra_h_left = make_extra(extra_h_left_whole, box_num=1) # box1 ?????? ?????? ???
            extra_h_right = make_extra(extra_h_right_whole, box_num=2) #box 2&3 ?????? ?????? ???
            extra_h = extra_h_left+extra_h_right
            
            # ????????? bbox
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
        
        # height??? ??? ???????????? 1,3??????, 2,4?????? box?????? ?????? column??? ????????????.
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
            
            if max(w1,w3)+max(w2,w4) > half:  # ?????? ????????? ??? ????????? ?????? ????????? ?????? ?????? (height ????????? ????????? ??????)
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None

            ################################## scl extend here #########################################
            extra_w_whole = half - (max(w1,w3) + max(w2,w4)) # ????????? ???????????? ?????? ?????? ????????? ??? ?????? ??????
            extra_h_left_whole = half - (h1+h3)              # box1,3 ????????? ??????
            extra_h_right_whole = half - (h2+h4)             # box2,4 ????????? ??????
            
            # ?????? ?????? ??????
            extra_w = make_extra(extra_w_whole, box_num=2)
            extra_w = extra_w+extra_w
            more_w_left = make_extra(abs(w1-w3), box_num=1)
            more_w_right = make_extra(abs(w2-w4), box_num=1)
            
            # extra??? ??? ?????? ??? ?????? ???????????? ??? ?????? ??? ?????? ?????? ??????
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
                
            # ?????? ?????? ??????
            extra_h_left = make_extra(extra_h_left_whole, box_num=2)
            extra_h_right = make_extra(extra_h_right_whole, box_num=2)
            extra_h = extra_h_left[0:2]+extra_h_right[0:2]+extra_h_left[2:4]+extra_h_right[2:4]
            
            
            # ????????? bbox
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
            # (height ????????? ????????? ??????), 
            # ?????? box1?????? ??? box2, box3 ?????? ??? box4??? ???????????? ?????? ??????????????? 
            # ????????? ?????????????????? box1,box3??? max????????? ???????????? box2,box4??? ?????????
            
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
            # height_max, width_max ?????? half??? ?????? ?????? ????????? box??? ???????????? Pack??? ????????????.
            xmin = int(new_bboxes[0][0])
            ymin = int(new_bboxes[0][1])
            xmax = int(new_bboxes[0][2])
            ymax = int(new_bboxes[0][3])
#             w = xmax - xmin
#             h = ymax - ymin
            ##########################scl extend here ##########################
            # 1. 1???????????? halfxhalf ??????
            # 2. center??? ?????? ???????????? ???????????? ?????????????????? ???????????? ????????? ??????            
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
        
        # ??? row??? ????????? box??? ????????????. 
        elif len(new_bboxes) == 2:
            box1 = list(map(int, new_bboxes[0]))
            box2 = list(map(int, new_bboxes[1]))
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            # ????????? ?????? half??? ????????? ?????????.
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
        
        # ?????? ??????????????? ??? box??? ??? row??? ???????????? ????????? ??? box??? ?????? row??? ????????????.
        elif len(new_bboxes) == 3:
            # sort by width
            sorted_bboxes = sorted(new_bboxes, key=lambda k: (k[2]-k[0]), reverse = True)
            box1 = list(map(int, sorted_bboxes[0]))
            box2 = list(map(int, sorted_bboxes[1]))
            box3 = list(map(int, sorted_bboxes[2]))
            # width??? ?????? ??? box??? ????????? ??? height??? ?????? ??? box??? height??? ?????? half??? ????????? ?????????.
            h1 = box1[3]-box1[1]
            h2 = box2[3]-box2[1]
            h3 = box3[3]-box3[1]
            if h1 + max(h2,h3) > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            # width??? ?????? ??? box?????? ????????? ??? box??? width??? ?????? half??? ????????? ?????????.
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
        
        # width??? ??? ???????????? 1,3??????, 2,4?????? box?????? ?????? row??? ????????????.
        elif len(new_bboxes) == 4:
            sorted_bboxes = sorted(new_bboxes, key=lambda k: (k[2]-k[0]), reverse = True)
            box1 = list(map(int, sorted_bboxes[0]))
            box2 = list(map(int, sorted_bboxes[1]))
            box3 = list(map(int, sorted_bboxes[2]))
            box4 = list(map(int, sorted_bboxes[3]))
            # width ???????????? ??????????????? ??? 1,3????????? width??? ??? box, 2,4????????? width??? ??? box??? ?????? row??? ????????????.
            # ??? ??? ??? width??? ?????? half??? ????????? ?????????.
            w1 = box1[2]-box1[0]
            w2 = box2[2]-box2[0]
            w3 = box3[2]-box3[0]
            w4 = box4[2]-box4[0]
            if w1 + w3 > half:
                return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None
            # width ???????????? ??????????????? ??? 1,3????????? width??? ??? box, 2,4????????? width??? ??? box??? ?????? row??? ????????????.
            # ??? ??? ??? row??? height??? ?????? half??? ????????? ?????????.
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
        # box??? 5??? ????????? ???
        else:
            return original_img, torch.tensor([[0,0,full,full]]).cuda(), None, None


def check_intersect_possibility(boxes_other_oreientation):
    '''     
    <Description>
    Check whether boxex intersect
    
    <input> 
    boxes_other_oreientation : boxes with other orientation
    
    <output>
    box_possible : (0: no intersect, 1 : intersect)
    '''    
    #??????????????? ??????
    box_possible = [0]*(len(boxes_other_oreientation)//2)
    
    #???????????? ????????? 1
    for i in range(len(boxes_other_oreientation)//2):
        for j in range(len(boxes_other_oreientation)//2):
            if j!=i:
                if min(boxes_other_oreientation[2*j+1],boxes_other_oreientation[2*i+1])-\
                                                    max(boxes_other_oreientation[2*i],boxes_other_oreientation[2*j])>0:
                    box_possible[i] = 1
    return box_possible

def greedy_intersect_extend(extra, boxes, boxes_other_oreientation, full=960):
    '''pack and detect ?????? greedy??? extend??? ????????? bbox??? return??????.
    args :
        extra : boudning box ?????????????????? ?????? ?????????
        boxes : [box1 xmin, box1 xmax, box2 xmin, box2 xmax, box3 xmin, box3 xmax ...]
        ?????? [box1 ymin, box1 ymax, box2 ymin, box2 ymax, box3 ymin, box3 ymax ...]
        boxes_other_oreientation : ??????????????? ???????????? ??????????????? ???????????? ???????????? ??????????????? ?????? ??? ??????????????? ????????????.
    return:
        ????????? ???????????? ????????? box??? ??????????????? ??????
    '''

    
    box_intersect = check_intersect_possibility(boxes_other_oreientation) #??????????????????
    box_dic = {}        # box point (boxes)
    extra_dic = {}      # ?????? ????????? ?????? (extra)
    box_intersect_dic = {}
    # box_n??????_(min or max) ----> ex ) box2_0 : 2?????? box min???
    for  i in range(len(boxes)):
        box_dic[f'box{i//2}_{i%2}'] = boxes[i] 
        extra_dic[f'box{i//2}_{i%2}'] = extra[i]
        box_intersect_dic[f'box{i//2}_{i%2}'] = box_intersect[i//2]

    # ????????? box????????? ?????? ?????? ??????
    centers = []
    for  i in range(len(boxes)//2):
        centers.append((boxes[2*i]+boxes[2*i+1])//2)
    
    #??? ???????????? box_dic??? ?????? change ---> ex) [box2,box1,box3] ???????????? [box2 xmin, box2 xmax, box1 xmin, box1 xmax, box3 xmin, box3 xmax]
    center_order = {}
    for  i in range(len(centers)):
        center_order[f'{i}'] = centers[i]
    center_order = {k: v for k, v in sorted(center_order.items(), key=lambda item: item[1], reverse=False)}
    new_box_dic = {}
    for key in center_order:
        new_box_dic[f'box{int(key)}_0'] = box_dic[f'box{int(key)}_0']
        new_box_dic[f'box{int(key)}_1'] = box_dic[f'box{int(key)}_1']
    
    # ????????? ??? ?????????, ??? ??????????????? intersect ?????? ???????????? ??????????????? ???????????? ?????? (???????????? box?????? ????????? 0,full?????? ????????????)
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

    # ?????? ????????? ?????? ????????? ????????? ???????????? ?????????????????? ???????????? ??????
    # ???????????? ??????????????? ??????????????? ????????? ?????????
    # ???????????? ???????????? ???????????? ???????????? intersect ?????? ?????????, ????????? ?????????
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
              
    # ????????? ????????? ??????
    for  i in range(len(boxes)):
        boxes[i] = new_box_dic[f'box{i//2}_{i%2}']

    return boxes

def make_extra(extra_whole, box_num=2):
    '''?????? ????????? ????????????
    args :
        extra_whole : half ??????????????? ?????? ?????? ?????? ??? pixel
        box_num : ????????? ????????? ???????????? box??? ??????
    return:
        box ????????? ????????????(boxmin ??????, boxmax ??????) ?????? ?????? ??????????????? ???????????????
        ex ) box_num=2, extra_whole = 45 -->> [11,11,11,12]
    '''
    extra = []
    for i in range(box_num*2):
        numb = int(np.ceil(extra_whole/(box_num*2-i)))
        extra.insert(0, numb)
        extra_whole -= numb
    return extra


def expand_box(boxes, expand , thres_len, x_size, y_size, prev_out):
    length_x = (boxes[:,2:3] - boxes[:,0:1])*expand/2
    length_y =  (boxes[:,3:4] - boxes[:,1:2])*expand/2
    
    boxes[:,0:1],boxes[:,1:2],boxes[:,2:3],boxes[:,3:4] = boxes[:,0:1]-length_x,boxes[:,1:2]-length_y,boxes[:,2:3]+length_x,boxes[:,3:4]+length_y
    return boxes

# +
# def intersect_extend(extra, box_first_small,box_first_big, box_second_small, box_second_big):
    
#     limitation_dic = {'first_min_lim': box_first_small, 'first_max_lim': (box_first_big+box_second_small)//2-box_first_big, \
#             'second_min_lim': box_second_small - (box_second_small+box_first_big)//2, 'second_max_lim': full-box_second_big}

#     limitation_dic = {k: v for k, v in sorted(limitation_dic.items(), key=lambda item: item[1], reverse=False)}

#     add_to_others = 0
#     for i, key in enumerate(limitation_dic):
#         value = limitation_dic[key]
#         extra[i] += add_to_others//(4-i)
#         add_to_others -= add_to_others//(4-i)
#         if value >= extra[i]:
#             value = extra[i]
#         else:
#             add_to_others += extra[i] - value
#         limitation_dic[key] = value

#     box_first_small -= limitation_dic['first_min_lim']
#     box_first_big += limitation_dic['first_max_lim']
#     box_second_small-= limitation_dic['second_min_lim']
#     box_second_big += limitation_dic['second_max_lim']
#     return box_first_small,box_first_big, box_second_small, box_second_big

# +
# def no_intersect_extend(extra, box_first_small,box_first_big, box_second_small, box_second_big):
    
#     box_first_small -= extra[0]
#     box_first_big += extra[1]
#     box_second_small -= extra[2]
#     box_second_big += extra[3]
#     if box_first_small<0:
#         box_first_big -= box_first_small
#         box_first_small = 0
#     if box_first_big>full:
#         box_first_small += (full-box_first_big)
#         box_first_big = full
#     if box_second_small<0:
#         box_second_big -= box_second_small
#         box_second_small = 0
#     if box_second_big>full:
#         box_second_small += (full-box_second_big[2])
#         box_second_big = full
#     return box_first_small,box_first_big, box_second_small, box_second_big

# +
# def make_newboxes(boxes):
#     box_num = len(boxes)
#     while True:
#         groups = {}
#         for i in range(0,box_num):
#             new = True
#             for key, value in groups.items():
#                 if i in value:
#                     new = False
#                     save_key = key
#             if new:
#                 save_key = i
#                 groups[i] = [i]
                
#             for j in range(i+1, box_num):
#                 if check_intersect(boxes[i], boxes[j]):
#                     groups[save_key].append(j)

#         # find enclosing bounding box around the union bounding boxes in each connected component
#         new_bboxes = []
#         for idxs in groups.values():
#             idxs = list(set(idxs))
#             xmin = torch.min(boxes[idxs,0])
#             xmax = torch.max(boxes[idxs,2])
#             ymin = torch.min(boxes[idxs,1])
#             ymax = torch.max(boxes[idxs,3])
#             enclosing_bbox = torch.LongTensor([xmin, ymin, xmax, ymax]).cuda()
#             new_bboxes.append(enclosing_bbox)
#         new_bboxes = torch.vstack(new_bboxes)
#         # check new bboxes intersect
#         intersect = False
#         box_num = len(new_bboxes)
#         for i in range(0, box_num):
#             for j in range(i+1, box_num):
#                 if check_intersect(new_bboxes[i], new_bboxes[j]):
#                     intersect = True

#         # if there are intersecting boxes in new_bboxes, build enclosing box again
#         if intersect:
#             boxes = new_bboxes
#         else:
#             break
#     return new_bboxes

# +
# def check_intersect(boxA, boxB):
#     ''' ????????? box??? ?????? ???????????? ???????????? check??????.
#     args :
#         boxA : boxA ?????? [xmin,ymin,xmax,ymax ???]
#         boxB : boxB ??????
#     '''
#     xA = max(float(boxA[0]), float(boxB[0]))
#     yA = max(float(boxA[1]), float(boxB[1]))
#     xB = min(float(boxA[2]), float(boxB[2]))
#     yB = min(float(boxA[3]), float(boxB[3]))
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#     if interArea > 0:
#         return True
#     else:
#         return False
