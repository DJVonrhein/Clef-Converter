#Takes an image as input. Yields a modified image (nonsense if error)

import datetime
import cv2 as cv
import numpy as np
from math import ceil, floor
import sys
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmin
import pytesseract
from PIL import Image 
from pdf2image import convert_from_path




#place solid white rectangle to hide template matches
def erase_template_match(template, img, threshold, message=""):
    count = 0
    res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED) 
    
    # Store the coordinates of matched area in a numpy array 
    loc = np.where(res >= threshold) 

    # Erase matched regions by covering with a solid white rectangle  
    for pt in zip(*loc[::-1]): 
        cv.rectangle(img, (pt[0] - 3, pt[1] - 1), (pt[0] + (int)(template.shape[1]) + 1, pt[1] + (int)(template.shape[0]) + 1), (255, 255, 255), cv.FILLED) 
        count += 1
    # if(count > 0):
    #     print(message)

'''not actually used in conversion, only a developing tool'''
#place thin black rectangle to show template matches
def show_template_match(template, img, threshold):
    res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED) 
    
    # # Store the coordinates of matched area in a numpy array 
    loc = np.where(res >= threshold) 

    # # Erase matched regions by covering with a solid white rectangle  
    for pt in zip(*loc[::-1]): 
        # staff_removed[pt[0]]
        cv.rectangle(img, (pt[0] - 1, pt[1] - 1), (pt[0] + (int)(template.shape[1]) + 1, pt[1] + (int)(template.shape[0]) + 1), (0, 0, 0), 1) 

#same as above, but with the added condition that template does not intersect a note head. notes_img is binary mask
def get_notehead_mask(img, note_height):
    img_cpy = img.copy()
    half_note_head_template = cv.imread('../templates/halfnote.png', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height)
    up_width = (int)((up_height /half_note_head_template.shape[0] ) * half_note_head_template.shape[1])
    half_note_head_template = cv.resize(half_note_head_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)

    quarter_note_head_template = cv.imread('../templates/quarternote.png', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height)
    up_width = (int)((up_height /quarter_note_head_template.shape[0] ) * quarter_note_head_template.shape[1])
    quarter_note_head_template = cv.resize(quarter_note_head_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)

   
    erase_template_match(template=half_note_head_template, img=img_cpy, threshold=0.55)#consider hiding flats prior
    erase_template_match(template=quarter_note_head_template, img=img_cpy, threshold=0.7)

    #return diff between img and img_cpy as binary mask
    notehead_mask = np.zeros(img.shape, np.uint8)
    notehead_mask[img != img_cpy] = 255
    return notehead_mask

#idea: since bar lines look like note stems, only erase template matches not bordering noteheads
#TODO: ignore matches outside of staff
def erase_bar_lines(img, noteheads_mask, threshold, note_height, is_piano):
    img_before = img.copy() 

    bar_line_template = cv.imread('../templates/barline.png', cv.IMREAD_GRAYSCALE) 
    # print("old bar line size:", bar_line_template.shape)
    up_height = (int)(note_height * 4.1)
    if (is_piano):
        up_height = (int)(up_height * 3)
    up_width = (int)((up_height /bar_line_template.shape[0] ) * bar_line_template.shape[1])
    bar_line_template = cv.resize(bar_line_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    # print("new bar line size:", bar_line_template.shape)    
    
    res = cv.matchTemplate(img, bar_line_template, cv.TM_CCOEFF_NORMED) 
    horizontal_tol = note_height * 0.2
    vertical_tol = note_height 
    
    # # Store the coordinates of matched area in a numpy array 
    loc = np.where(res >= threshold) 

    noteheads_color = cv.cvtColor(noteheads_mask, cv.COLOR_GRAY2RGB)

    #TODO: shift a tighter area
    for pt in zip(*loc[::-1]):
        #is a bar line (not a stem) because it doesn't intersect a note
        if( 255 not in noteheads_mask[(int)(pt[1] - vertical_tol):(int)(pt[1] + bar_line_template.shape[0] + vertical_tol ), (int)(pt[0] - horizontal_tol):(int)(pt[0] + bar_line_template.shape[1] + horizontal_tol )]):
            cv.rectangle(img, ((int)(pt[0]  ), (int)(pt[1] - vertical_tol)), ((int)(pt[0] + bar_line_template.shape[1] ), (int)(pt[1] + bar_line_template.shape[0] + vertical_tol)), (255, 255, 255), cv.FILLED) 
            cv.rectangle(noteheads_color, ((int)(pt[0]  - 1), (int)(pt[1] - vertical_tol)), ((int)(pt[0] + bar_line_template.shape[1] + 1), (int)(pt[1] + bar_line_template.shape[0] + vertical_tol)), (255, 0, 0), cv.FILLED) 
    
    bar_lines_mask = np.zeros(img.shape, np.uint8)
    indices = np.where(img_before != img)
    bar_lines_mask[indices] = 255

    bar_lines_mask[img_before>180] = 0
    return bar_lines_mask
    
        
#idea: hide half and full rests during preprocessing
#TODO: main does not yet utilize this feature
def erase_long_rests(img, noteheads_mask, note_height, threshold):
    #The same template matches both half and full rests
    half_full_rest_template = cv.imread('../templates/rests/halffullrests.png', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 0.7)
    up_width = (int)((up_height /half_full_rest_template.shape[0] ) * half_full_rest_template.shape[1])
    # print("up height, width =", up_height, up_width)
    half_full_rest_template = cv.resize(half_full_rest_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    res = cv.matchTemplate(img, half_full_rest_template, cv.TM_CCOEFF_NORMED) 
    
    # # Store the coordinates of matched area in a numpy array 
    loc = np.where(res >= threshold) 

    # cv.imwrite(sys.argv[3], noteheads_mask)
    # # Erase matched regions by covering with a solid white rectangle only if notehead mask isnt white in region 
    # for pt in zip(*loc[::-1]): 
    #     is_longrest = True
    #     for i in range(pt[0] - 2, pt[0] + (int)(half_full_rest_template.shape[1]) ):
    #         for j in range(pt[1] - 2, pt[1] + (int)(half_full_rest_template.shape[0]) ):
    #             if(noteheads_mask[j,i] == 255):
    #                 is_longrest = False
    #                 break
    #     # if(noteheads_mask[(pt[0] - 2):(pt[0] + (int)(template.shape[1]) + 2), (pt[0] - 1):(pt[0] + (int)(template.shape[1] + 1))].all(0)):
    #     if(is_longrest == True):
    #         cv.rectangle(img, (pt[0] - 1, pt[1] - 1), (pt[0] + (int)(half_full_rest_template.shape[1]) + 1, pt[1] + (int)(half_full_rest_template.shape[0]) + 1), (255, 255, 255), cv.FILLED) 

#hide the double bar lines during preprocessing.
def erase_double_bar_lines(img, note_height, threshold, is_piano):
    double_bar_line_template = cv.imread('../templates/doublebarline.png', cv.IMREAD_GRAYSCALE) 
    # print("old double bar line size:", double_bar_line_template.shape)
    up_height = (int)(note_height * 4.1)
    if(is_piano):
        up_height = (int)(up_height * 3)
    up_width = (int)((up_height /double_bar_line_template.shape[0] ) * double_bar_line_template.shape[1])
    double_bar_line_template = cv.resize(double_bar_line_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    # print("new double bar line size:", double_bar_line_template.shape)    #ONLY WORKS FOR BIGGER BASSES
    # cv.imwrite(intermediate_output_file_name, double_bar_line_template)
    img_before = img.copy()
    erase_template_match(template=double_bar_line_template, img=img, threshold=threshold)#0.8 for takefive.png
    
    double_bar_lines_mask = np.zeros(img.shape, np.uint8)
    indices = np.where(img_before != img)
    double_bar_lines_mask[indices] = 255
    double_bar_lines_mask[img_before>200] = 0
    return double_bar_lines_mask

#hide the time signature during preprocessing. Templates of every possible digit (1 - 9) are searched, some needing looser thresholds in order to be matched
def erase_time_signatures(img, note_height, threshold):
    img_before = img.copy()

    #1
    one_template = cv.imread('../templates/time_signature/time1.png', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height*2)    #SAME HEIGHT FOR ALL DIGITS
    up_width = (int)((up_height / one_template.shape[0] ) * one_template.shape[1])
    one_template = cv.resize(one_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=one_template,threshold=0.85, message="erased 1s")# notes get seen as 1s easily

    #2
    two_template = cv.imread('../templates/time_signature/time2.png', cv.IMREAD_GRAYSCALE) 
    up_width = (int)((up_height / two_template.shape[0] ) * two_template.shape[1])
    two_template = cv.resize(two_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=two_template,threshold=threshold, message="erased 2s")

    #3
    three_template = cv.imread('../templates/time_signature/time3.png', cv.IMREAD_GRAYSCALE) 
    up_width = (int)((up_height / three_template.shape[0] ) * three_template.shape[1])
    three_template = cv.resize(three_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=three_template,threshold=threshold, message="erased 3s")

    #4
    four_template = cv.imread('../templates/time_signature/time4.png', cv.IMREAD_GRAYSCALE) 
    up_width = (int)((up_height / four_template.shape[0] ) * four_template.shape[1])
    four_template = cv.resize(four_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=four_template,threshold=threshold, message="erased 4s")

    #5
    five_template = cv.imread('../templates/time_signature/time5.png', cv.IMREAD_GRAYSCALE) 
    up_width = (int)((up_height / five_template.shape[0] ) * five_template.shape[1])
    five_template = cv.resize(five_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=five_template,threshold=threshold, message="erased 5s")

    #6
    six_template = cv.imread('../templates/time_signature/time6.png', cv.IMREAD_GRAYSCALE) 
    up_width = (int)((up_height / six_template.shape[0] ) * six_template.shape[1])
    six_template = cv.resize(six_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=six_template,threshold=threshold, message="erased 6s")

    #7
    seven_template = cv.imread('../templates/time_signature/time7.png', cv.IMREAD_GRAYSCALE) 
    up_width = (int)((up_height / seven_template.shape[0] ) * seven_template.shape[1])
    seven_template = cv.resize(seven_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=seven_template,threshold=threshold, message="erased 7s")

    #8
    eight_template = cv.imread('../templates/time_signature/time8.png', cv.IMREAD_GRAYSCALE) 
    up_width = (int)((up_height / eight_template.shape[0] ) * eight_template.shape[1])
    eight_template = cv.resize(eight_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=eight_template,threshold=threshold, message="erased 8s")

    #9
    nine_template = cv.imread('../templates/time_signature/time9.png', cv.IMREAD_GRAYSCALE) 
    up_width = (int)((up_height / nine_template.shape[0] ) * nine_template.shape[1])
    nine_template = cv.resize(nine_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    erase_template_match(img=img,template=nine_template,threshold=threshold, message="erased 9s")

    indices = np.where(img_before != img)
    time_signatures_mask = np.zeros(img.shape, np.uint8)
    time_signatures_mask[indices] = 255
    time_signatures_mask[img_before>200] = 0
    return time_signatures_mask

#idea: erase quarter, eighth and sixteenth rests earlier than the other types of rests
def erase_short_rests(img, note_height, threshold):

    img_before = img.copy()

    quarter_rest_template = cv.imread('../templates/rests/quarterrest.png', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 2.8)
    up_width = (int)((up_height /quarter_rest_template.shape[0] ) * quarter_rest_template.shape[1])
    quarter_rest_template = cv.resize(quarter_rest_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)

    #TODO: ERASE FORTES PRIOR, diagonal dilation to recover 8th and 16th rests
    #quarter rests need looser threshold
    
    erase_template_match(template=quarter_rest_template, img=img, threshold=threshold + 0.12)

    eighth_rest_template = cv.imread('../templates/rests/eighthrest.png', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 2.15)
    up_width = (int)((up_height /eighth_rest_template.shape[0] ) * eighth_rest_template.shape[1])
    eighth_rest_template = cv.resize(eighth_rest_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)

    erase_template_match(template=eighth_rest_template, img=img, threshold=threshold)

    sixteenth_rest_template = cv.imread('../templates/rests/sixteenthrest.png', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 2.15)
    up_width = (int)((up_height /sixteenth_rest_template.shape[0] ) * sixteenth_rest_template.shape[1])
    sixteenth_rest_template = cv.resize(sixteenth_rest_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    
    #TODO: fix "cutoff" appearance for 16th rests
    erase_template_match(template=sixteenth_rest_template, img=img, threshold=threshold)

    short_rests_mask = np.zeros(img.shape, np.uint8)

    #TODO: consider if logic is optimal
    indices = np.where(img_before != img) 
    # indices = np.where(img_before + 20 > img) 
    short_rests_mask[indices] = 255
    # cv.imwrite(sys.argv[3], short_rests_mask)

    #TODO: base it off of a real variable (staff upper threshold)
    short_rests_mask[img_before>180] = 0    #180 is reasonable for MuseScore and modern qualilty images.

    return short_rests_mask
    

#use pytesseract to detect title, composer, or other words. 
def erase_words_no_resize(img, threshold, note_height):
    img_before = img.copy()

    results = pytesseract.image_to_data(img, config='--psm 11', output_type='dict')
    for i in range(len(results["text"])):
        # take coords of the bounding box for every result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # take confidence of the text
        conf = int(results["conf"][i])
        
        # Cover all text with a white rectangle only if confident
        if conf > threshold: 
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)

    words_mask = np.zeros(img.shape, np.uint8)
    indices = np.where(img != img_before)
    words_mask[indices] = 255
    return words_mask

def erase_words(img, threshold, note_height):
    original_shape = img.shape
    
    #TODO: immense speedup if we can avoid this upsizing
    up_height = (int)(2200)
    up_width = (int)(1700)
    # up_width = (int)((up_height /half_note_head_template.shape[0] ) * half_note_head_template.shape[1])
    img_enlarged = cv.resize(img, (up_width, up_height), interpolation=cv.INTER_CUBIC)
    img_enlarged_cpy = img_enlarged.copy()

    img_enlarged_blurred = cv.medianBlur(img_enlarged, 3)
    results = pytesseract.image_to_data(img_enlarged_blurred, config='--psm 11', output_type='dict')

    for i in range(len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # Extract the confidence of the text
        conf = int(results["conf"][i])
        
        if conf > threshold: 
            # Cover the text with a white rectangle
            cv.rectangle(img_enlarged, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # img = cv.resize(img, (img.shape[1], img_before.shape[0]), interpolation=cv.INTER_LINEAR)
    words_mask = np.zeros(img_enlarged.shape, np.uint8)
    indices = np.where(img_enlarged != img_enlarged_cpy)
    words_mask[indices] = 255
    words_mask = cv.resize(words_mask, (original_shape[1], original_shape[0]), interpolation=cv.INTER_LINEAR)
    return words_mask

#TODO: repeat for the smaller, inline clefs present in some pieces, after a downsize
def erase_bass_clefs(img, note_height, desired_clef, threshold, new_clefs):


    img_before = img.copy()
    
    bass_clef_template = cv.imread('../templates/bassclef.jpg', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 3.7)
    up_width = (int)((up_height /bass_clef_template.shape[0] ) * bass_clef_template.shape[1])
    bass_clef_template = cv.resize(bass_clef_template, (up_width, up_height), interpolation=cv.INTER_LINEAR) 

    # Perform match operations. 
    res = cv.matchTemplate(img, bass_clef_template, cv.TM_CCOEFF_NORMED) 
    
    # # Store the coordinates of matched area in a numpy array 
    loc = np.where(res >= threshold) 

    treble_clef_template = cv.imread('../templates/trebleclef.jpg', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 7)
    up_width = (int)((up_height /treble_clef_template.shape[0] ) * treble_clef_template.shape[1])
    treble_clef_template = cv.resize(treble_clef_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    for pt in zip(*loc[::-1]): 
        cv.rectangle(img, (pt[0] - 2, pt[1] - 2), (pt[0] + (int)(bass_clef_template.shape[1]) + 2, pt[1] + (int)(bass_clef_template.shape[0]) + 2), (255, 255, 255), cv.FILLED)
        if(desired_clef == "treble"):
            new_clefs[(pt[1] - (int)(note_height)):(pt[1] + (int)(treble_clef_template.shape[0]) - (int)(note_height)), pt[0]:(pt[0] + (int)(treble_clef_template.shape[1]) )]= treble_clef_template
        
    bass_clefs_mask = np.zeros(img.shape, np.uint8)
    indices = np.where(img_before != img)
    bass_clefs_mask[indices] = 255
    bass_clefs_mask[img_before>180] = 0
    return bass_clefs_mask

#TODO: repeat for the smaller, inline clefs present in some pieces, after a downsize
def erase_treble_clefs(img, note_height, desired_clef, threshold, new_clefs):
    img_before = img.copy()

    treble_clef_template = cv.imread('../templates/trebleclef.jpg', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 7)
    up_width = (int)((up_height /treble_clef_template.shape[0] ) * treble_clef_template.shape[1])
    treble_clef_template = cv.resize(treble_clef_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    # print("new treble clef size:", treble_clef_template.shape)  #TODO: ONLY WORKS FOR BIGGER TREBLES

    # Perform match operations. 
    res = cv.matchTemplate(img, treble_clef_template, cv.TM_CCOEFF_NORMED) 
    # # Store the coordinates of matched area in a numpy array 
    loc = np.where(res >= threshold) 


    bass_clef_template = cv.imread('../templates/bassclef.jpg', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 3.7)
    up_width = (int)((up_height /bass_clef_template.shape[0] ) * bass_clef_template.shape[1])
    bass_clef_template = cv.resize(bass_clef_template, (up_width, up_height), interpolation=cv.INTER_LINEAR) 
    # Erase matched regions by covering with a solid white rectangle  
    for pt in zip(*loc[::-1]): 
        cv.rectangle(img, (pt[0] - 2, pt[1] - 2), (pt[0] + (int)(treble_clef_template.shape[1]) + 2, pt[1] + (int)(treble_clef_template.shape[0]) + 2), (255, 255, 255), cv.FILLED) 
        if(desired_clef == "bass"):
            new_clefs[pt[1] + (int)(1.3*note_height):pt[1] + (int)(bass_clef_template.shape[0]) + (int)(1.3*note_height), pt[0]:pt[0] + (int)(bass_clef_template.shape[1])] = bass_clef_template
        
        
    treble_clefs_mask = np.zeros(img.shape, np.uint8)
    indices = np.where(img_before != img)
    treble_clefs_mask[indices] = 255
    treble_clefs_mask[img_before>180] = 0
    return treble_clefs_mask


#expands clef areas vertically.  Only accounts for treble and bass (green and blue, respectively). This function is somewhat of a bottleneck with its python for loops.
def add_clef_areas(line_separators_array, music_lines, trebles_mask, basses_mask):

    #downsize to 1% the input size for faster execution of for loops
    music_lines_shrunk = cv.resize(music_lines, (music_lines.shape[1]//10, music_lines.shape[0]//10), interpolation = cv.INTER_AREA)
    rows, cols, channels = music_lines_shrunk.shape

    #eliminate new gray areas present after aveeraging in resize()
    #TODO: can resize() not do this?
    music_lines_shrunk[music_lines_shrunk > 0] = 255


    trebles_mask_shrunk = cv.resize(trebles_mask, (trebles_mask.shape[1]//10, trebles_mask.shape[0]//10))
    basses_mask_shrunk = cv.resize(basses_mask, (basses_mask.shape[1]//10, basses_mask.shape[0]//10))
    
    
    white_indices = np.where(music_lines_shrunk == (255,255,255))
    white_indices = white_indices[:-1]
    music_lines_shrunk[white_indices] = (0,0,0)
    
    #Step 1: expand the clef areas vertically
    line_separators_shrunk_array = line_separators_array.copy()
    for i in range( line_separators_shrunk_array.shape[0]):
        line_separators_shrunk_array[i] = line_separators_shrunk_array[i]//10

    prev_line = line_separators_shrunk_array[0]
    for i in range (1, line_separators_shrunk_array.shape[0]):
        
        # row = line_separators_array[i] + (line_separators_array[i + 1] - line_separators_array[i])//2
        row = line_separators_shrunk_array[i]
        trebles_mask_fit_to_line = trebles_mask_shrunk[prev_line:row, :] 
        #clef becomes tall white rectangle 
        trebles_mask_fit_to_line = cv.dilate(trebles_mask_fit_to_line, cv.getStructuringElement(cv.MORPH_RECT, (1,3)), iterations=rows//10)
        trebles_mask_shrunk[prev_line:row, :] = trebles_mask_fit_to_line

        basses_mask_fit_to_line = basses_mask_shrunk[prev_line:row, :] 
        #clef becomes tall white rectangle
        basses_mask_fit_to_line = cv.dilate(basses_mask_fit_to_line, cv.getStructuringElement(cv.MORPH_RECT, (1,3)), iterations=rows//10)
        basses_mask_shrunk[prev_line:row, :] = basses_mask_fit_to_line
        # curr_line_trebles[row:,:] = 0
        # line_boundaries_mask[row, :] = 0
        prev_line = row

    music_lines_shrunk[trebles_mask_shrunk==255] = (0,255,0)
    music_lines_shrunk[basses_mask_shrunk==255] = (255,0,0)
    

    #Step 2: expand the clef areas horizontally
    # print("music_lines[0,0,:]:",music_lines[0,0,:])
    black =  np.array([0,0,0], np.uint8)
    for i in range(rows):
        prev_color = black
        for j in range(cols):
            if(np.array_equal(music_lines_shrunk[i,j], black)):
                music_lines_shrunk[i,j] = prev_color
            elif(not np.array_equal(music_lines_shrunk[i,j], prev_color)):
                prev_color = music_lines_shrunk[i,j]
   
    
        
    music_lines = cv.resize(music_lines_shrunk, (music_lines.shape[1], music_lines.shape[0]))
    for i in line_separators_array:
        music_lines[i,:] = (0,0,255)
    return music_lines

#hides the braces during preprocessing (piano pieces only)
def erase_braces(img, threshold, note_height):
    img_before = img.copy()

    brace_template = cv.imread('../templates/brace.png', cv.IMREAD_GRAYSCALE) 
    up_height = (int)(note_height * 14)
    up_width = (int)((up_height /brace_template.shape[0] ) * brace_template.shape[1])
    brace_template = cv.resize(brace_template, (up_width, up_height), interpolation=cv.INTER_LINEAR)

    # get template matches 
    res = cv.matchTemplate(img, brace_template, cv.TM_CCOEFF_NORMED) 
    
    # Store the coordinates of confident matches in a numpy array 
    loc = np.where(res >= threshold) 

    # Erase matched regions by covering with a solid white rectangle  
    for pt in zip(*loc[::-1]): 
        cv.rectangle(img, (pt[0] - 2, pt[1] - 2), (pt[0] + (int)(brace_template.shape[1]) + 2, pt[1] + (int)(brace_template.shape[0]) + 2), (255, 255, 255), cv.FILLED) 

    braces_mask = np.zeros(img.shape, np.uint8)
    indices = np.where(img_before != img)
    braces_mask[indices] = 255
    braces_mask[img_before>180] = 0
    return braces_mask
#Estimates the distance between one staff line and the next. requires that music has no extra or missing lines staff lines. 
#TODO: for loops potential bottleneck
def get_average_note_height(staff_binary, gap_heights):
    prev = 255
    rows, cols = staff_binary.shape

    #TODO: consider optimality 
    tol = rows / 50 
    line_starts = [] #list of row heights for each staff line (only their topmost pixel)
    for i in range(rows):
        curr = staff_binary.item(i,cols//2 +1)
        if (curr == 0 and prev == 255): #start of black zone
            line_starts.append(i)
        prev = curr
    
    total_sum = 0
    divisor = 0
    for i in range(1, len(line_starts)):
        new_gap = line_starts[i] - line_starts[i-1]
        if(new_gap < tol):
            total_sum += new_gap
            divisor += 1

    average = total_sum  / divisor
    return average


# takes binary image of entire sheet.  Returns list of heights chosen as boundaries btwn staffs 
#TODO: fails if screenshot's left margin is nonexistent@
def segment_staffs(img): 
    
    rows, cols = img.shape
    print('enlarged shape: ', img.shape)
    
    half_width_structure = cv.getStructuringElement(cv.MORPH_RECT, (cols//2,1))
    staff_gaps = cv.erode(img, half_width_structure, cols//2)

    gap_heights = [0]    # will store middle (height) of each gap
    tol = rows // 10    # tolerance to prevent adding gaps that aren't tall enough
    curr = 0
    is_white = True
    for i in range(rows):
        curr = staff_gaps.item(i,cols//2 +1)
        if (curr== 0 and is_white): #start of black zone
            is_white = False
            if(i > gap_heights[-1] + tol):
                gap_heights.append(i)
        elif (curr==255 and not is_white ): #start of white zone
            is_white = True
            if(i > gap_heights[-1] + tol):
                gap_heights.append(i)

    assert len(gap_heights) > 1, "Error finding any gaps between the staff. Proceed if image is one line"
   
    return gap_heights

#put white between the white staff lines. Used for deciphering a clef's territory
def get_music_lines_mask(staff_binary, note_height):
    staff_binary = cv.bitwise_not(staff_binary)
    rows, cols = staff_binary.shape
    #sanitize margins by blacking out the top and bottom 
    y = 0
    while(staff_binary[y, cols//2] == 255):
        y += 1
    staff_binary[0:y, :] = 0
    y = rows - 1
    while(staff_binary[y, cols//2] == 255):
        y -= 1
    staff_binary[y:rows, :] = 0


    music_lines_mask = staff_binary.copy()

    line_start = -1
    line_end = -1
    prev = -1
    tol = rows // 100
    #idea: when curr_distance > note_height (+ some tolerance), this is a new line
    for i in range(rows):
        # print(staff_binary[i, cols//2])
        if(staff_binary[i, cols//2] == 255):
            #case 1: top of a new line
        
            if(i - prev > (int)(note_height + tol)):
                
                # line_end = i
                if(line_start > -1 and line_end > -1):
                    # print("whited ", line_start, " -> ", line_end)
                    music_lines_mask[line_start:line_end, :] = 255
                line_start = i
                
            #case 2: not new "line" BEcause we are still somewhere in between some ledger lines
            else:
                line_end = i
            prev = i
    
    music_lines_mask[line_start:line_end, :] = 255

    return music_lines_mask


def isolate_staff(img, line_width):
    #histogram of intensity
    # histr = cv.calcHist([img],[0],None,[256],[0,255]) 
    # plt.plot(histr)
    # plt.title("Original Music Intensity Counts")
    # plt.savefig(sys.argv[3])

    #TODO: pick the intensity just before the right peaks of histr plot
    staff_upper_threshold = 235 #takefive.png
    # staff_thresh_intensity = 220 #obsessed.png (musescore)

    ret,arbitrary_threshold = cv.threshold(img,staff_upper_threshold,255,cv.THRESH_BINARY) # we normally use otsu thresholding, but use this so fewer grays will be turned to white
    ret2,otsu = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) # otsu is generally used as the main binary form of image

    height, width = img.shape
    #Step 1 : get just the staff
    select_staff_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,1))  #kernel for eroding thin white lines - first, will be used to get just the staff
    horizStruct = cv.getStructuringElement(cv.MORPH_RECT, (width // 30,1))  #kernel for eroding thin white lines - first, will be used to get just the staff

    staff_shortened = cv.erode(cv.bitwise_not(arbitrary_threshold),select_staff_kernel,iterations = width // 6) # Note: image will be blank if iterations is too high. Must be less than staff length divided by (I believe) 3       Perfect before our Hough transform
    staff_lengthened = cv.dilate(staff_shortened, select_staff_kernel, iterations = width // 2 ) # Note: can be done infinitely without affecting future steps; we expect the staff to be too long anyway after this step

  
    return cv.bitwise_not(staff_lengthened)







def isolate_notes(bw):
    # Create the images that will use to extract the horizontal and vertical lines
    # show_wait_destroy("bw", bw)

    horizontal = bw.copy()
    vertical = bw.copy()
    # Specify size on horizontal axis
    cols = horizontal.shape[1] 
    horizontalsize = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontalsize,1))
    # Apply morphology operations
    horizontal = cv.erode( horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    # show_wait_destroy("horizontal", horizontal)
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows  // 30
    # print('rows, cols =', rows, cols)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, ( 1,verticalsize))
    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    # show_wait_destroy("vertical", vertical)
    # Inverse vertical image
    cv.bitwise_not(vertical, vertical)
    # show_wait_destroy("vertical_bit", vertical)
     # Extract edges and smooth image according to the logic
     # 1. extract edges
     # 2. dilate(edges)
     # 3. src.copyTo(smooth)
     # 4. blur smooth img
     # 5. smooth.copyTo(src, edges)

    # Step 1
    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2);
    # show_wait_destroy("edges", edges);
    # Step 2
    kernel = np.ones((2,2), dtype=np.uint8);
    edges = cv.dilate(edges, kernel);
    # show_wait_destroy("dilate", edges);
    # Step 3
    smooth = vertical.copy();
    # Step 4
    smooth = cv.blur(smooth, (2, 2));
    # Step 5
    # smooth.copyTo(vertical, edges); #C++
    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]
    # Show final result
    # show_wait_destroy("smooth - final", vertical);
    return vertical
    # cv.waitKey(0)

#"combine" the ink from 2 images. Essentially returns min(img1, img2)
def get_combined_ink(img1, img2):
    assert img1.shape == img2.shape and len(img1.shape)==2, "Tried to combine ink from images of different size"
    rows, cols = img1.shape
    for i in range(rows):
        for j in range(cols):
            img1[i, j] = min(img1[i,j], img2[i,j])
    return img1

#gets ink from img1 not in img2. gets min(grayscale1, grayscale2)  where they are different, and white where they are the same
def get_removed_ink(img1, img2):
    assert img1.shape == img2.shape and len(img1.shape)==2, "Tried to remove ink from images of different size"
    rows, cols = img1.shape
    for i in range(rows):
        for j in range(cols):
            if img1[i,j] != img2[i,j]:
                img1[i, j] = min(img1[i,j], img2[i,j])
            else:
                img1[i, j] = 255
    return img1

#move all notes in treble areas (where music_lines_mask is green) down a space 
#TODO: add new ledger lines where they are needed
def treble_to_bass(staff, notes, note_height, music_lines_mask):  
    ignored_notes = np.where(music_lines_mask != [0,255, 0])[:-1]
    treble_notes = notes.copy()
    treble_notes[ignored_notes] = 255
    non_treble_notes = notes.copy()
    non_treble_notes[non_treble_notes==treble_notes] = 255

    M = np.float32([[1, 0, 0], [0, 1,  note_height]])
    shifted = cv.warpAffine(treble_notes, M, (notes.shape[1], notes.shape[0]))
    #pad the top (now black) back to white
    shifted[0:(int)(note_height) + 1, :] = 255  
    

    treble_converted = get_combined_ink(shifted, staff)
    return get_combined_ink(treble_converted, non_treble_notes)

#move all notes in bass areas (where music_lines_mask is red) up a space 
#TODO: add new ledger lines where they are needed
def bass_to_treble(staff, notes, note_height, music_lines_mask):  
    ignored_notes = np.where(music_lines_mask != [255, 0, 0])[:-1]
    bass_notes = notes.copy()
    bass_notes[ignored_notes] = 255
    non_bass_notes = notes.copy()
    non_bass_notes[non_bass_notes==bass_notes] = 255

    M = np.float32([[1, 0, 0], [0, 1,  note_height * -1]])
    shifted = cv.warpAffine(bass_notes, M, (notes.shape[1], notes.shape[0]))
    #pad the bottom (now black) back to white
    shifted[shifted.shape[0] - (int)(note_height):, :] = 255  

    bass_converted = get_combined_ink(shifted, staff)
    return get_combined_ink(shifted, non_bass_notes)

#return an array of rows where color changed, including row 0 and row -1
def get_line_separators_array(music_lines_mask, music_lines):
    prev_color = 0
    rows, cols = music_lines_mask.shape
    arr = np.empty([44], np.uint16)
    arr[0] = 0
    curr_arr_idx = 1
    #arr will contain all the row indices that transition from black to white or white to black
    for i in range(rows):
        if(music_lines_mask[i, cols//2] != prev_color):
            assert curr_arr_idx < 42, "Image was detected to have over 20 lines! It likely won't convert well, aborting program"
            arr[curr_arr_idx] = i
            curr_arr_idx += 1
        prev_color = music_lines_mask[i, cols//2]

    assert curr_arr_idx < 42, "Image was detected to have over 20 lines! It likely won't convert well, aborting program"
    arr[curr_arr_idx] = rows - 1
    curr_arr_idx += 1
    arr.resize(curr_arr_idx)
    assert arr.shape[0] % 2 == 0, "Failed to distinguish the boundaries between lines. Got an odd number of line_separators"

    #row divisions will contain all the row indices that bisect the lines
    row_divisions = np.empty([20],np.uint16)
    curr_row = 0
    for i in range (0, arr.shape[0], 2):
        row = arr[i] + (arr[i + 1] - arr[i])//2
        # music_lines[row, :] = (0, 0, 255)
        # line_boundaries_mask[row, :] = 0
        row_divisions[curr_row] = row
        curr_row += 1
    assert curr_row > 1, "Failed to idenify boundaries for a single line of music"
    row_divisions.resize(curr_row)
    return row_divisions




   


#simply returns 100 for now, works well for musescore
#TODO: decide a threshold value based on the histograms of the staff. idea: return the bin just before the right side climb.
def estimate_staff_upper_threshold(grayscale_staff):
    #histogram of intensity
    # staff_histogram = cv.calcHist(images=[grayscale_staff],channels=[0],mask=None, histSize=[256],ranges=[0, 255]) 
    # plt.plot(staff_histogram)
    # plt.title("Staff Intensity Counts")
    # plt.savefig(sys.argv[3])
    # hist_arr = staff_histogram.reshape(staff_histogram.shape[0])
    # staff_avg_intensity = hist_arr.argmax() # the mode of the intensity histogram
    
    # num_medium_pixels_in_staff = hist_arr_4bin.sum()
    #threshold the staff at the first pixel frequency of 5% that is lighter than the symbols tend to be
    # hist_arr_4bin[:20] = 0 
    # valid_intensities = hist_arr[75:]
    # staff_upper_threshold = 115 #takefive.png
    staff_upper_threshold = 100 #obsessed.png (musescore)

    # assert staff_upper_threshold != 0, "Failed to threshold the staff for clean symbol extraction, breaking error"
    # print('staff_avg_intensity:', staff_avg_intensity)
    # print('num medium pixels in staff:', num_medium_pixels_in_staff)
    # print('staff_upper_threshold:', staff_upper_threshold)
    # hist_arr.flatten()
    # plt.clf()

    return staff_upper_threshold

# #TODO: decide a threshold value based on the histograms of the staff. idea: return the bin just before the right side climb.
# def get_staff_thresh(histogram, min ):
#     # local_minima = argrelextrema(histogram, np.less)
#     # local_minima = argrelmin(histogram)
#     # print("local_minima indices:",argrelextrema(histogram, np.less))

#     # print("local_minima:",histogram[argrelextrema(histogram, np.less)[0]])
    
#     max_idx = histogram.argmax()
#     max_int = histogram[max_idx]
#     local_minima = argrelmin(data=histogram, order=3)
#     print ("max_int:",max_int)
#     print("local_minima indices before:",local_minima)

#     print("local_minima before:",histogram[local_minima[0]])
#     def filter_smalls(x):
#         print("x:", x)
#         # return x > max / 10
#     local_minima = np.where(filter_smalls, local_minima, False)
#     print("local_minima indices after:",local_minima)

#     print("local_minima after:",histogram[local_minima[0]])
#     # for i in range(histogram.shape[0]):
#     #     if(histogram[i] < min):
#     #         min = histogram[i]
#     # print("min : ", min)
#     # if(min < 10):
#     #     min = 10

#     for i in range(max_idx, 0, -1):
#         if( histogram[i] < int(max_int / 18)):
#             print('staff_threshold :', i)
#             # return i
#     return 115


def convert_clef(img, desired_clef):
    if(len(img.shape)>2):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    assert img is not None, "convert_clef's img was None. Check with os.path.exists()"
    assert desired_clef == "bass" or desired_clef == "treble", "desired_clef should be bass or treble, got %s" % (desired_clef)

    is_piano = False    #if braces are template matched, this becomes True 
    
    #breaks segment_staffs
    ''' 
    BEGIN ENLARGE IMAGE FOR ACCURATE TEMPLATE MATCH
    '''
    print('Pre-processing started\n')
    start_time = datetime.datetime.now()
    print("img shape: ", img.shape)
    if(img.shape[0] < 2200):
        up_height = 2200
        up_width = (int)(up_height / img.shape[0]  * img.shape[1])
        img = cv.resize(img, (up_width, up_height), interpolation=cv.INTER_LINEAR)
    if(img.shape[1] < 1700):
        up_width = 1700
        up_height = (int)(up_width / img.shape[1]  * img.shape[0])
        img = cv.resize(img, (up_width, up_height), interpolation=cv.INTER_LINEAR)

    ''' 
    FINISH ENLARGE IMAGE FOR ACCURATE TEMPLATE MATCH
    '''

    rows, cols = img.shape
    grayscale_inv = cv.bitwise_not(img)
    img_bin_inv = cv.adaptiveThreshold(grayscale_inv,  255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    img_bin = cv.bitwise_not(img_bin_inv)


    '''
    ESTIMATE THE STAFF LINES AND REMOVE THEM
    '''

    gap_heights = segment_staffs(img= img_bin)
    line_width = gap_heights[1] - gap_heights[0] #size between gaps is the staff width  (line width)
    complete_staff_binary = isolate_staff(img=img.copy(),line_width=line_width) #modifies grayscale!!!!
    staff_mask = cv.bitwise_not(complete_staff_binary)

    grayscale_staff = img.copy()
    grayscale_staff[staff_mask==0] = 255

    staff_upper_threshold = estimate_staff_upper_threshold(grayscale_staff)

    #histograms coulld provide more accurate staff_upper_thresh in the future
    # staff_histogram = cv.calcHist(images=[grayscale_staff],channels=[0],mask=None, histSize=[256],ranges=[0, 255]) 
    # plt.plot(staff_histogram)
    # plt.title("Staff Intensity Counts")
    # plt.savefig(sys.argv[3])
    # hist_arr = staff_histogram.reshape(staff_histogram.shape[0])
    # staff_avg_intensity = hist_arr.argmax() # the mode of the intensity histogram
    
  
    #staff_threshed contains only the dark symbol overlaps
    ret, staff_threshed = cv.threshold(grayscale_staff, staff_upper_threshold, 255, cv.THRESH_BINARY)

    #don't just erase the entire staff before finding symbols. Erase the staff where it isnt as dark as the symbols! Using staff_threshed
    staff_no_overlaps = grayscale_staff.copy()
    staff_no_overlaps[staff_threshed==0] = 255

    
    staff_removed = img.copy()
    staff_removed[staff_no_overlaps<255] = 255

    
    #crucial for template sizing
    note_height = get_average_note_height(staff_binary=complete_staff_binary, gap_heights=gap_heights) #important for resizing our templates 
    print("average note height: ", note_height)

    '''
    END ESTIMATE STAFF
    '''


    '''
    BEGIN TEMPLATE MATCHING
    '''
    #we will template match staff_removed for static symbols which won't be translated vertically 

    #stores all symbols that we are going to erase through pattern matching
    #TODO: just make this a binary mask and get real values later
    static_symbols_mask = np.zeros(staff_removed.shape, np.uint8)
    staff_removed_copy = staff_removed.copy()

    #: dilate staff_removed (inverted) to recover staff overlap
    ret,staff_removed_bin = cv.threshold(staff_removed,200,255,cv.THRESH_BINARY) # wider range of grays will be kept as black than in adaptive thresholding
    staff_removed_bin_dilated = cv.bitwise_not(cv.dilate(cv.bitwise_not(staff_removed_bin), cv.getStructuringElement(cv.MORPH_RECT, (1, 3)), 1))
    light_symbol_overlaps_mask = np.zeros(staff_removed.shape, np.uint8)
    light_symbol_overlaps_mask[staff_removed_bin_dilated != staff_removed_bin] = 255
    img[light_symbol_overlaps_mask == 0] = 255
    staff_removed = get_combined_ink(staff_removed, img)

    new_clefs = np.full(staff_removed.shape, [255], np.uint8)
    trebles_mask = erase_treble_clefs(img=staff_removed, note_height=note_height, threshold=0.55, desired_clef=desired_clef, new_clefs=new_clefs)
    basses_mask = erase_bass_clefs(img=staff_removed, note_height=note_height, threshold=0.55, desired_clef=desired_clef, new_clefs=new_clefs)
    # cv.imwrite(intermediate_output_file_name,new_clefs)
    if(desired_clef == "treble"):
        static_symbols_mask = cv.bitwise_or(static_symbols_mask, trebles_mask)
    elif(desired_clef == "bass"):
        static_symbols_mask = cv.bitwise_or(static_symbols_mask, basses_mask)

    
    
    music_lines_mask = get_music_lines_mask(staff_binary=complete_staff_binary, note_height=note_height)
    music_lines = cv.cvtColor(music_lines_mask, cv.COLOR_GRAY2RGB)

    #draw red lines to divide the staff lines, return their row indices
    line_separators_array = get_line_separators_array(music_lines_mask, music_lines)

    music_lines = add_clef_areas(line_separators_array.copy(), music_lines, trebles_mask, basses_mask)

    #quarter, eighth, sixteenth rests
    short_rests_mask = erase_short_rests(img=staff_removed, note_height=note_height, threshold = 0.63)
    static_symbols_mask = cv.bitwise_or(static_symbols_mask, short_rests_mask)

    words_mask = erase_words_no_resize(img=staff_removed, threshold = 92, note_height=note_height) #pytesseract wants thresh as %
    static_symbols_mask = cv.bitwise_or(static_symbols_mask, words_mask)

    braces_mask = erase_braces(img=staff_removed, threshold=0.55, note_height=note_height)
    if(braces_mask.__contains__(255)):
        is_piano = True
    static_symbols_mask = cv.bitwise_or(static_symbols_mask, braces_mask)

    time_signatures_mask = erase_time_signatures(img=staff_removed, note_height=note_height, threshold = 0.73)
    static_symbols_mask = cv.bitwise_or(static_symbols_mask, time_signatures_mask)


    #get make sure the next template matches are not touching notes
    noteheads_mask = get_notehead_mask(img=staff_removed, note_height=note_height)
    
    double_bar_lines_mask = erase_double_bar_lines(img=staff_removed,note_height=note_height,threshold=0.8, is_piano=is_piano)
    static_symbols_mask = cv.bitwise_or(static_symbols_mask, double_bar_lines_mask)

    #TODO: erase only if passing template match is perfect height
    bar_lines_mask = erase_bar_lines(img = staff_removed, noteheads_mask = noteheads_mask,  threshold = 0.55 , note_height = note_height, is_piano = is_piano)
    static_symbols_mask = cv.bitwise_or(static_symbols_mask, bar_lines_mask)
    # cv.imwrite(intermediate_output_file_name, static_symbols_mask)
    # print("made it out of static_symbol")
    # erase_long_rests(img = staff_removed, noteheads_mask=noteheads_mask, threshold = 0.5, note_height=note_height)

    '''
    END TEMPLATE MATCHING
    '''

    '''
    MOVE DYNAMIC PARTS AND COMBINE WITH STATIC PARTS
    '''
    #pure staff - dark overlaps changed to avg (median) color
    #TODO: find these actual average and threshold
    # staff_avg_intensity = 200

    # cv.imwrite(intermediate_output_file_name, grayscale_staff)
    # pure_staff = grayscale_staff
    # overlapped_indices = np.where(pure_staff < staff_upper_threshold)
    # pure_staff[overlapped_indices] = staff_avg_intensity
    # pure_staff[staff_mask == 0] = 255
    #TODO: use blurring
    pure_staff = grayscale_staff.copy()
    overlapped_indices = np.where(pure_staff < staff_upper_threshold + 20)
    pure_staff[overlapped_indices] = staff_upper_threshold + 80

    end_time = datetime.datetime.now()
    print("\nPre-processing completed in ", end_time - start_time)
    #TODO: pass in staff first thresholded
    start_time = datetime.datetime.now()
    if(desired_clef == "bass"):
        print("\nStarting to convert treble areas to bass")
        output = treble_to_bass(notes=staff_removed, staff=pure_staff, note_height=note_height, music_lines_mask=music_lines)
    elif(desired_clef == "treble"):
        print("\nStarting to convert bass areas to treble")
        output = bass_to_treble(notes=staff_removed, staff=pure_staff, note_height=note_height, music_lines_mask = music_lines)
    end_time = datetime.datetime.now()
    print("\nConversion completed in ", end_time - start_time)
    # print("started at ", start_time)
    # print("finished at ", end_time)
    # return music_lines

    staff_removed[static_symbols_mask == 0] = 255

    output = get_combined_ink(output, staff_removed)
    static_indices = np.where(static_symbols_mask.copy() == 255)
 
    output[static_indices] = staff_removed_copy[static_indices]

    output = get_combined_ink(output, new_clefs)
    '''
    FINISH COMBINE
    '''

    return output
    
def main():
    start_time = datetime.datetime.now()
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    desired_clef = sys.argv[3]

    #case 1: input is a pdf file
    #TODO: pdf functionality not finished
    if(len(input_file_name) > 4 and input_file_name.find(".pdf") == len(input_file_name) - 4):
        assert False, "pdfs functionality is yet to come. Please try jpg, png, or other single image format"
        pages = convert_from_path(input_file_name,fmt = 'jpeg')
        for page in pages:
            page = convert_clef(img = np.array(page), desired_clef = desired_clef)
            print("page shape :", page.shape)
            page = cv.cvtColor(page, cv.COLOR_GRAY2BGR)
            # print("page shape length after:", len(page.shape))

        images = [
            Image.open(page)
            for page in pages
        ]

        images[0].save(output_file_name, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:])
    #case 2: a single image file fomrat
    else:
        img = cv.imread(input_file_name, cv.IMREAD_GRAYSCALE)
        # start_time = datetime.datetime.now()
        transposed_image = convert_clef(img = img, desired_clef = desired_clef)
        # print('time elapsed: ', datetime.datetime.now() - start_time)
        write_status = cv.imwrite(output_file_name, transposed_image)

        
    
    if(write_status):
        end_time = datetime.datetime.now()
        print("\nEntire pipeline took ", end_time - start_time, "\n")
        return 0
    
    return -1


if __name__ == "__main__":
#  main(sys.argv[1:])
    main()