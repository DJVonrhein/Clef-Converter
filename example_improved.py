import cv2 as cv
import numpy as np
from math import ceil, floor
# from matplotlib import pyplot as plt


def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def segment_staffs(bw): # takes binary image of entire sheet.  Returns list of heights chosen as boundaries btwn staffs 
    
    rows, cols = bw.shape
    print('rows: ', rows)
    print('cols: ', cols)
    full_width_structure = cv.getStructuringElement(cv.MORPH_RECT, (cols,1))
    staff_gaps = cv.erode(bw, full_width_structure)

    # show_wait_destroy('staff_gaps', staff_gaps)

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
    if(len(gap_heights) == 0):
        print("Error finding any gaps between the staff. Proceed if image is one line") 
    # print("gap_heights:")
    # for i in gap_heights:
    #     print(i)

    return gap_heights


def isolateStaff(grayscale, line_width):
    ret,arbitrary_threshold = cv.threshold(grayscale,200,255,cv.THRESH_BINARY) # we normally use otsu thresholding, but use this so fewer grays will be turned to white
    ret2,otsu = cv.threshold(grayscale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) # otsu is generally used as the main binary form of image

    height, width = grayscale.shape
    #Step 1 : get just the staff
    select_staff_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,1))  #kernel for eroding thin white lines - first, will be used to get just the staff
    horizStruct = cv.getStructuringElement(cv.MORPH_RECT, (width // 30,1))  #kernel for eroding thin white lines - first, will be used to get just the staff

    # select_staff_kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]] ,np.uint8)  #kernel for eroding thin white lines - first, will be used to get just the staff
    staff_shortened = cv.erode(cv.bitwise_not(arbitrary_threshold),select_staff_kernel,iterations = width // 25) # Note: image will be blank if iterations is too high. Must be less than staff length divided by (I believe) 3       Perfect before our Hough transform
    staff_lengthened = cv.dilate(staff_shortened, select_staff_kernel, iterations = width // 10 ) # Note: can be done infinitely without affecting future steps; we expect the staff to be too long anyway after this step


    staff_heights = []

    rows, cols = grayscale.shape
    for i in range(rows):
        if (staff_lengthened.item(i,floor(cols/2))== 0):
            staff_heights.append(i)
            # print(i)
    if(len(staff_heights) == 0):
        raise Exception("Sorry, no staff was found, this sheet failed to be transposed") 

    
    staff_lengthened = cv.erode(staff_lengthened, horizStruct)
    complete_staff_img = cv.bitwise_and(staff_lengthened, cv.bitwise_not(arbitrary_threshold)) 
    complete_staff_img = cv.bitwise_not(complete_staff_img )


    '''CLEAN HORIZONTAL LINES NOT PART OF STAFF'''

    full_height_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, height))
    # full_height_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, line_width))

    right_border_shaved = cv.erode(complete_staff_img, full_height_structure)
    # cv.imshow('right_border_shaved',right_border_shaved)

    '''END CLEAN'''

    return complete_staff_img



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



def shift_notes_up(staff, notes, full_step_width, half_steps):
    M = np.float32([[1, 0, 0], [0, 1,  half_steps * full_step_width // 8]])
    shifted = cv.warpAffine(notes, M, (notes.shape[1], notes.shape[0]))
    
    title = 'Shfted by ' + str(half_steps)
    # cv.imshow(title, shifted)
    print( staff.shape, shifted.shape)
    anded =  cv.bitwise_and(staff, shifted)
    # cv.imshow("title", anded)
    cv.imwrite('img/shifted.png', anded)
    return cv.bitwise_and(staff, shifted)

def shift_notes_down(staff, notes, full_step_width, half_steps):
    M = np.float32([[1, 0, 0], [0, 1,  half_steps * full_step_width // 13 * -1]])
    shifted = cv.warpAffine(notes, M, (notes.shape[1], notes.shape[0]))
    
    title = 'Shfted by ' + str(half_steps)
    # cv.imshow(title, shifted)
    print( staff.shape, shifted.shape)
    anded =  cv.bitwise_and(staff, shifted)
    # cv.imshow("title", anded)
    cv.imwrite('img/shifted.png', anded)
    return cv.bitwise_and(staff, shifted)


def main():

    
    # input_file_name = 'img/takefive.png'
    # input_file_name = 'img/reminiscences.png'
    # input_file_name = 'img/remiFullPage.png'
    input_file_name = "img/maryHadALittleLamb.png"
    # img_color =  cv.imread(input_file_name) 
    grayscale = cv.imread(input_file_name, cv.IMREAD_GRAYSCALE)    # input is converted from color to grayscale
    assert grayscale is not None, "file could not be read, check with os.path.exists()"

    rows, cols = grayscale.shape
    # isolateStaff(gray)
    grayscale_inv = cv.bitwise_not(grayscale)
    img_bin_inv = cv.adaptiveThreshold(grayscale_inv,  255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    img_bin = cv.bitwise_not(img_bin_inv)

    
    ret,arbitrary_threshold = cv.threshold(grayscale,200,255,cv.THRESH_BINARY) # wider range of grays will be kept as black than in adaptive thresholding
    # cv.imshow('arbitrary_threshold', arbitrary_threshold)
    cv.imwrite('img/arbitrary_threshold.png',arbitrary_threshold)
    cv.imwrite('img/grayscale.png',grayscale)

    # ret2,otsu = cv.threshold(grayscale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) # otsu is generally used as the main binary form of image
    

    '''
    ESTIMATE THE STAFF LINES
    '''

    # complete_staff_img = isolateStaff(grayscale)
    # cv.imshow('complete_staff_img', complete_staff_img)

    gh = segment_staffs(img_bin)
    l_w = gh[1] - gh[0]
    complete_staff_img = isolateStaff(grayscale,l_w)
    # cv.imshow('complete_staff_img', complete_staff_img)

    '''
    END ESTIMATE STAFF
    '''



    '''
    SEGMENT IMAGE AT GAPS BETWEEN STAFFS
    '''
    # gap_heights = segment_staffs(arbitrary_threshold)
    gap_heights = segment_staffs(img_bin)
    gap_heights[-1] = rows - 1

    '''
    END SEGMENT IMAGE BETWEEN STAFFS
    '''



    '''
    BEGIN EXAMPLE CODE ( Staff removal )
    '''
    example = cv.bitwise_not(arbitrary_threshold).copy()
    # example = img_bin_inv.copy()
    # cv.imshow('example BEFORE', example)

    for i in range(len(gap_heights) - 1):       # CALL isolate_notes() ON EACH STAFF LINE, INDIVIDUALLy
        curr = np.copy(example[gap_heights[i]:gap_heights[i+1]])
        # isolate_notes(curr)
        example[gap_heights[i]:gap_heights[i+1]] = 255
        # print("type of return: ", type(isolate_notes(curr)), type(bw))
        cv.bitwise_and(example[gap_heights[i]:gap_heights[i+1]], isolate_notes(curr), example[gap_heights[i]:gap_heights[i+1]])
        
    # show_wait_destroy('example', example)
    cv.imshow('example',example)
    cv.imwrite('img/example.png', example)

    '''
    END EXAMPLE CODE 
    '''



    '''
    BEGIN RECOVERING LOST HORIZONTAL FEATURES
    '''

    anded = cv.bitwise_and(example, complete_staff_img) # Key: get what's missing, after some cleaning, then add it to example
    # show_wait_destroy('example AND complete_staff_img', anded)
    xored = cv.bitwise_xor(anded, arbitrary_threshold)
    # show_wait_destroy('xor\'d', xored) #GIVES LOST HORIZONTAL FEATURES
    kernel = np.ones((2,2), dtype=np.uint8);
    cleaned_xored = cv.erode(xored, kernel);
    example_recovered = cv.bitwise_and(example, cv.bitwise_not(cleaned_xored)) 
    # show_wait_destroy('example_recovered', example_recovered)

    all_black = example_recovered.copy()
    all_black[:] = 0
    print('SHOWING')
    grayscale_masked = cv.bitwise_or(grayscale, all_black,mask =cv.bitwise_not(example_recovered))
    # cv.imshow('grayscale_masked', grayscale_masked)
    grayscale_part_not_masked =cv.bitwise_and(grayscale, example_recovered)
    cv.imshow('grayscale_part_not_masked', grayscale_part_not_masked)

    cv.imshow('example_recovered', example_recovered)
    # cv.imwrite('img/example_recovered.png', example_recovered)
    # cv.waitKey(0)

    '''
    END RECOVERING LOST HORIZONTAL FEATURES
    '''


    '''
    LOCATE BAR LINES
    '''
    line_width = 0  #line width is the height of one measure ( 2 staffs or piano, 1 staff for other instruments)
    if(len(gap_heights) > 1):
        line_width = gap_heights[1] - gap_heights[0]
    print('line_width:',line_width)

    if(line_width > 0): 
        bar_line_struc = cv.getStructuringElement(cv.MORPH_RECT, (1,line_width   //3))
        # cv.imshow('img_bin_inv',img_bin_inv)
        bar_line_erosion = cv.erode(cv.bitwise_not(arbitrary_threshold), bar_line_struc) #shows all bar lines, shortened
        # cv.imshow('bar_line_erosion', bar_line_erosion)
    else:
        print('Sheet had one line, and caused there to be no bar lines detected')
    '''
    END LOCATE BAR LINES
    '''


    '''
    SHIFT NOTES VERTICALLY DEPENDING ON USER's TRANSPOSITION VALUE
    '''
    direction = ""
    while direction.lower() != "up" and direction.lower() != "down":
        direction = input("Shift notes up or down?\n")
    half_steps = ""
    while half_steps.isnumeric() == False:
        half_steps = input("How many half steps " + direction + "?\n")
    
    if direction == "up":
        notes_shifted = shift_notes_up(example_recovered, complete_staff_img, line_width//4, int(half_steps))
    else:
        notes_shifted = shift_notes_down(example_recovered, complete_staff_img, line_width//4, int(half_steps))

    cv.imshow('notes_shifted',notes_shifted)

    '''
    END SHIFT NOTES
    '''
    cv.waitKey(0)
    return 0














    '''Begin contour experimentation'''
    # staff_missing_cleaned = cv.morphologyEx(staff_missing, cv.MORPH_CLOSE, k) #clean up holes from staff removal
    # staff_missing_cleaned = cv.bitwise_and(cv.bitwise_not(staff_missing), cv.bitwise_not(otsu))


    # cv.imwrite('img/nostaff.jpg',cv.bitwise_not(staff_missing_cleaned))  # contains only the staff, correctly sized

    # staff_missing = cv.bitwise_not(staff_missing)
    # cv.imshow('staff_missing_cleaned', cv.bitwise_not(staff_missing_cleaned))
    # staff_l_masked = cv.bitwise_and(otsu, otsu, mask = staff_lengthened)
    # cv.imshow('staff_l_masked', staff_l_masked)
    # otsu_masked = cv.bitwise_and(staff_l_masked, )


    # contour_img = gray.copy()
    # contour_img[:] = 0
    # contours, hierarchy = cv.findContours(staff_missing, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv.findContours(cv.bitwise_not(staff_missing), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(contour_img, contours, -1,  255, 1 )

    # mask = np.ones(gray.shape[:2], dtype="uint8") * 255
    # # loop over the contours
    # for c in contours:
    # 	# if the contour is bad, draw it on the mask
    # 	if is_contour_bad(c):
    # 		cv.drawContours(mask, [c], -1, 0, -1)
    # # remove the contours from the image and show the resulting images
    # no_ellipse = cv.bitwise_and(contour_img, contour_img, mask=mask)
    # # cv.imshow('no_ellipse', no_ellipse)

    # cleaned_staff_missing = cv.bitwise_not(cv.bitwise_or(staff_missing, contour_img))
    # contour_img = cv.bitwise_not(contour_img)

    # # cv.imshow('contour_img',contour_img)


    # contour_img = cv.bitwise_not(contour_img)
    # notes_mask  = cv.bitwise_or(staff_missing, contour_img)

    # masked = cv.bitwise_or(cv.bitwise_not(complete_staff_img), cv.bitwise_not(otsu), contour_img)# TODO: CAN WE RECONSTRUCT LIKE THIS AFTER MOVING NOTES?
    # # cv.imshow('masked', cv.bitwise_not(masked))
    # # cv.imshow('notes_mask', notes_mask)
    # cv.imshow('staff_missing',cv.bitwise_not(staff_missing))

    # # cv.imshow('cleaned_staff_missing', cleaned_staff_missing) #GREAT MASK FOR JUST SYMBOLS (first, get bitwise_not)
    # perfect_staff_missing = cv.bitwise_and(cv.bitwise_not(cleaned_staff_missing),cv.bitwise_not(staff_missing), cv.bitwise_not(complete_staff_img))
    # # cv.imshow('perfect_staff_missing', perfect_staff_missing) #BEST VISUALIZER WE HAVE OF MEANINGFUL SYMBOLS
    '''end contour experimentation'''



    '''BEGIN WATERSHED experimentation (outline quarter notes in blue)'''
    # # noise removal
    # # opening = cv.morphologyEx(cv.bitwise_not(otsu),cv.MORPH_OPEN,k, iterations = 1)
    # opening = cv.morphologyEx(cv.bitwise_not(otsu),cv.MORPH_OPEN,k, iterations = 1)

    # # sure background area
    # sure_bg = cv.dilate(opening,k,iterations=3)
    
    # # Finding sure foreground area
    # dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    # ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv.subtract(sure_bg,sure_fg)

    # # Marker labelling
    # ret, markers = cv.connectedComponents(sure_fg)
    
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1
    
    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0

    # #apply watershed
    # img_color_watershed = img_color
    # # markers = cv.watershed(img_color_watershed,markers)
    # # img_color_watershed[:] = [255,255,255]
    # # img_color_watershed[markers == -1] = [255,0,0]
    # # cv.imshow('img_color_watershed', img_color_watershed)       #OUTLINES QUARTER NOTES IN BLUE
    ''' END WATERSHED '''        


    ''' HOUGH LINES experimentation '''
    # gray[:] = 255 # white out the image's binary

    # edges = cv.Canny(horizontal_erosion,50,150,apertureSize = 3)
    # lines = cv.HoughLines(edges,1,np.pi/180,400)
    # for line in lines:
    #  rho,theta = line[0]
    #  a = np.cos(theta)
    #  b = np.sin(theta)
    #  x0 = a*rho
    #  y0 = b*rho
    #  x1 = int(x0 + 1000*(-b))
    #  y1 = int(y0 + 1000*(a))
    #  x2 = int(x0 - 1000*(-b))
    #  y2 = int(y0 - 1000*(a))
    
    #  cv.line(gray,(x1,y1),(x2,y2), 0,1)
    # cv.imwrite('img/staff.jpg',gray)         # choppiness!!!!
    '''END HOUGH LINES '''


    #The size of one staff line is known - find distance between first 5 non-adjacent staff_heights
    x = 0
    idx = 1
    staff_size = 0
    prev = staff_heights[0]
    while(x < 4):
        if(idx >= len(staff_heights)):
            raise Exception("Sorry, no staff was found, this sheet cannot be transposed") 
        curr = staff_heights[idx]
        if(curr - prev > 1):
            staff_size += curr - prev
            prev = curr
            x += 1
        # prev = curr
        idx += 1

    print(staff_size)

    # cv.waitKey(0)





if __name__ == "__main__":
#  main(sys.argv[1:])
    main()