# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 19:17:26 2021

@author: ludov
"""

# import time
# import os
import numpy as np
# import tensorflow as tf
import cv2
#pour masques
import imutils

#--- Initialisation Variables ---#
font = cv2.FONT_HERSHEY_COMPLEX
color_black=(0,0,0)
color_blue=(255,0,0)
color_green=(0,255,0)
color_cyan=(255,255,0)
color_orange=(0,255,255)
color_red=(0, 0, 255)
cap = cv2.VideoCapture("DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4") # On récupère la vidéo
width=int(cap.get(3))
himag=int(cap.get(4))
marge = 100#px

#--- Initialisation Tracking ---#
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 1.0)


#--- FILTRE BACKGROUND ---#

#Filtre Hue
H_h_min,H_s_min,H_v_min = 19, 0, 0#vals test: 19, 0, 0
H_h_max,H_s_max,H_v_max = 179, 255, 255#vals test: 179, 255, 255
lower_H = np.array([H_h_min,H_s_min,H_v_min])
upper_H = np.array([H_h_max,H_s_max,H_v_max])

#Filtre Sat
S_h_min,S_s_min,S_v_min = 0, 81, 0#vals test: 19, 0, 0
S_h_max,S_s_max,S_v_max = 255, 240, 255#vals test: 179, 255, 255
lower_S = np.array([S_h_min,S_s_min,S_v_min])
upper_S = np.array([S_h_max,S_s_max,S_v_max])

#Filtre Val
V_h_min,V_s_min,V_v_min = 0, 0, 10#vals test: 19, 0, 0
V_h_max,V_s_max,V_v_max = 255, 255, 162#vals test: 179, 255, 255
lower_V = np.array([V_h_min,V_s_min,V_v_min])
upper_V = np.array([V_h_max,V_s_max,V_v_max])

#Filtre déchets:
lower_D = np.array([0,134,0])
upper_D = np.array([138,255,49])

#Filtre ombres:
lower_O = np.array([5,0,5])
upper_O = np.array([55,40,30])  
nbr_tot_cars = 0
s_test = 0
L_counts = []


for test_count in range(1): #pour effectuer des tests unitaires
    
    kernel1 = np.ones((5, 5), np.uint8)
    counts_cars =[]
    cap = cv2.VideoCapture("DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4") # On récupère la vidéo
    width=int(cap.get(3))
    himag=int(cap.get(4))
    list_ROI_tracking = []
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        success,frame = cap.read()
        if success:
            img = frame.copy()
            
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #données à traiter img[159:][:]
            
            #background color substraction
            mask_H = cv2.inRange(imgHSV,lower_H,upper_H)
            mask_S = cv2.inRange(imgHSV,lower_S,upper_S)
            mask_V = cv2.inRange(imgHSV,lower_V,upper_V)
            
            mask_HSV = cv2.add(cv2.add(mask_H, mask_S), mask_V)
            
            imgResult_1 = cv2.bitwise_and(img,img,mask=mask_HSV)
            
            #filter quality upgrades:
            #dechets
            mask_D = cv2.inRange(imgHSV,lower_D,upper_D)
            mask_HSV_2 = cv2.absdiff(mask_HSV, mask_D)
            #ombres
            mask_O = cv2.inRange(imgHSV,lower_O,upper_O)
            mask_HSV_3 = cv2.absdiff(mask_HSV_2, mask_O)
            
            
            for i in range(2):
                if i == 0:
                    mask_HSV_4=cv2.erode(mask_HSV_3, kernel1)
                    mask_HSV_4=cv2.dilate(mask_HSV_4, kernel1)
                else:
                    mask_HSV_4=cv2.erode(mask_HSV_4, kernel1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
                    
                    # mask_HSV_4=cv2.dilate(mask_HSV_4, kernel1)
                    mask_HSV_4=cv2.dilate(mask_HSV_4, kernel)
            imgResult_2 = cv2.bitwise_and(img,img,mask=mask_HSV_4)
            if test_count == 1:
                filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                # Applying cv2.filter2D function on our Logo image
                imgResult_2=cv2.filter2D(imgResult_2,-1,filter)
    
            
            contours_2 = cv2.findContours(mask_HSV_4, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours_2 = imutils.grab_contours(contours_2)
            
            areas_contours_2 = []
            
            contours_2_final = []
            
            
            list_M_f = []
            
            # DETECTION
            nbr_added = 0
            for cr in range(len(contours_2)):
                area = cv2.contourArea(contours_2[cr])
                areas_contours_2.append(area)
                if area > 10000 and area < 35000:
                    valid_contour = contours_2[cr]
                    contours_2_final.append(valid_contour)

                    
                    #calcul barycentre
                    M_f = cv2.moments(valid_contour)
                    cx_f = int(M_f["m10"]/M_f["m00"])
                    cy_f = int(M_f["m01"]/M_f["m00"])
                    
                    if not list_M_f:
                        if np.min(valid_contour.T[1]) > 160 and cy_f > 160:
                            list_M_f.append(np.array((cx_f,cy_f)))
                            #drawing center circle and rectangle
                            cv2.circle(imgResult_2, (cx_f, cy_f), 15, color_orange, -1)
                            cv2.putText(imgResult_2, str(area), ( cx_f, cy_f), font, 1, color_blue)
                            ROI = (min([x.T[0] for x in valid_contour])[0], min([x.T[1] for x in valid_contour])[0], 
                                   max([x.T[0] for x in valid_contour])[0], max([x.T[1] for x in valid_contour])[0])      
                            ROI_tracking = (ROI[0], ROI[1], ROI[2]- ROI[0], ROI[3] - ROI[1])
                            #tracking
                            list_ROI_tracking.append(ROI_tracking)
                            nbr_added += 1
                            cv2.rectangle(imgResult_2, ROI[:2], ROI[2:], color_green, 2)
                            
                            #sur image origine
                            cv2.rectangle(img, ROI[:2], ROI[2:], color_green, 2)
                            
                        
                    if list_M_f:
                        if not sum([int(np.linalg.norm(np.array((elem[0] + elem[2]//2, elem[1] + elem[3]//2)) - np.array((cx_f,cy_f))) < marge) for elem in list_ROI_tracking]):    
                            if not sum([int(np.linalg.norm(elem - np.array((cx_f,cy_f))) < marge) for elem in list_M_f]):
                                if not sum([elem[0] < cx_f and cx_f < elem[0] + elem[2] and elem[1] < cy_f and cy_f < elem[1] + elem[3] for elem in list_ROI_tracking]):
                                    ROI = (min([x.T[0] for x in valid_contour])[0], min([x.T[1] for x in valid_contour])[0], 
                                           max([x.T[0] for x in valid_contour])[0], max([x.T[1] for x in valid_contour])[0])
                                    
                                    if not sum([ROI[0] < elem[0] + elem[2]//2 and elem[0] + elem[2]//2 < ROI[2] and ROI[1] < elem[1] + elem[3]//2 and elem[1] + elem[3]//2 < ROI[3] for elem in list_ROI_tracking]):
                                    
                                    
                                        if cy_f > 160: #au-delà on est dans la zone du texte
                                            list_M_f.append(np.array((cx_f,cy_f)))
                                            
                                            #drawing center circle and rectangle
                                            cv2.circle(imgResult_2, (cx_f, cy_f), 15, color_orange, -1)
                                            cv2.putText(imgResult_2, str(area), ( cx_f, cy_f), font, 1, color_blue)
                                            
                                            ROI_tracking = (ROI[0], ROI[1], ROI[2]- ROI[0], ROI[3] - ROI[1])
                                            list_ROI_tracking.append(ROI_tracking)
                                            nbr_added += 1
                                            #tracking 
                                            
                                            cv2.rectangle(imgResult_2, ROI[:2], ROI[2:], color_green, 2)
                                        
                                            #sur image origine
                                            cv2.rectangle(img, ROI[:2], ROI[2:], color_green, 2)
                                                   
                                
            #update tracking:
            for i in range(len(list_ROI_tracking)):
                _, list_ROI_tracking[i]=cv2.meanShift(mask_HSV_4, list_ROI_tracking[i], term_criteria)
                
            #suppress out of screen boxes, suppress overlapping:
            nbr_suppressed = 0
            update_list_ROI_tracking = []
            for i in range(len(list_ROI_tracking)):
                if list_ROI_tracking[i][1] > 160 and cv2.countNonZero(mask_HSV_4[list_ROI_tracking[i][1]:list_ROI_tracking[i][1]+list_ROI_tracking[i][3],list_ROI_tracking[i][0]:list_ROI_tracking[i][0]+list_ROI_tracking[i][2]]) > 10000 :
                    must_be_zero = 0
                    for j in range(i+1, len(list_ROI_tracking)):
                        if list_ROI_tracking[i][0] < list_ROI_tracking[j][0] + list_ROI_tracking[j][2] // 2 and list_ROI_tracking[j][0] + list_ROI_tracking[j][2] // 2 < list_ROI_tracking[i][0] + list_ROI_tracking[i][2] and list_ROI_tracking[i][1] < list_ROI_tracking[j][1] + list_ROI_tracking[j][3] // 2 and list_ROI_tracking[j][1] + list_ROI_tracking[j][3] // 2 < list_ROI_tracking[i][1] + list_ROI_tracking[i][3]:
                            must_be_zero = 1
                            break
                        
                    if must_be_zero == 0:
                        update_list_ROI_tracking.append(list_ROI_tracking[i])
                    else:
                        nbr_suppressed += 1
                else:
                    nbr_suppressed += 1
            list_ROI_tracking = update_list_ROI_tracking
             
            #drawing boxes
            for i in range(len(list_ROI_tracking)):
                cv2.rectangle(img,(list_ROI_tracking[i][0],list_ROI_tracking[i][1]), 
                                  (list_ROI_tracking[i][0] + list_ROI_tracking[i][2],list_ROI_tracking[i][1] + list_ROI_tracking[i][3]), 
                                  color_green, 2)
                cv2.putText(img, str(np.product(list_ROI_tracking[i][-2:])), list_ROI_tracking[i][:2], font, 1, (255,100,0), 4)
                
                cv2.rectangle(imgResult_2,(list_ROI_tracking[i][0],list_ROI_tracking[i][1]), 
                                  (list_ROI_tracking[i][0] + list_ROI_tracking[i][2],list_ROI_tracking[i][1] + list_ROI_tracking[i][3]), 
                                  color_green, 2)
            counts_cars.append(len(list_ROI_tracking))
            print(counts_cars[-1])

            #Multi tracking
            nbr_new_cars = max(0,nbr_added - nbr_suppressed)
            nbr_tot_cars += nbr_new_cars
            
            # if not s_test:
            #     s_test = len(list_ROI_tracking)
            
            # for i in range(len(counts_cars)-1):
            #     s_test += counts_cars[i+1] - counts_cars[i]
            
            key2=cv2.waitKey(1)
            if key2 == ord('q'):
                break
            if key2 == ord('p'):
                cv2.waitKey()
             
    
            #Nbre de voitures détectées sur image2:
            cv2.putText(img, str(nbr_tot_cars), ( 1650, 140), font, 5, color_red,3)
            cv2.putText(img, "Ludovic DUMONT", ( 770, 95), font, 2, color_black,3)
            
            # cv2.imshow("Output_m1",imgResult_1)
            # cv2.imshow("img", cv2.resize(img, (width//2, himag//2)))
            cv2.imshow("img", img)
            
            # cv2.imshow("imgResult_2",cv2.resize(imgResult_2, (width//2, himag//2)))
            
                
            
    
    cap.release()
    cv2.destroyAllWindows()
    L_counts.append(counts_cars)
    
#optimisation manuelle:
if len(L_counts)==2:
    
    if L_counts[0] == L_counts[1]:
        print("aucun changement dans le nombre de voitures détectées, impact paramètre neutre")
    elif np.average(L_counts[0]) > np.average(L_counts[1]):
        print("nombre de voitures détectées inférieur par rapport au premier passage, impact paramètre négatif")
    else:
        print("nombre de voitures détectées supérieur par rapport au premier passage, impact paramètre positif")
    