import cv2 as cv
import numpy as np

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
    print("gap_heights:")
    for i in gap_heights:
        print(i)

    return gap_heights

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

def main():
    # input_file_name = 'img/takefive.png'
    # input_file_name = 'img/example.png'
    # input_file_name = 'img/reminiscences.png'
    # input_file_name = 'img/remiOneLine.png'
    input_file_name = 'img/remiFullPage.png'

    # img_color =  cv.imread(input_file_name) 
    grayscale = cv.imread(input_file_name, cv.IMREAD_GRAYSCALE)    # input is converted from color to grayscale
    assert grayscale is not None, "file could not be read, check with os.path.exists()"

    grayscale_inv = cv.bitwise_not(grayscale)
    img_bin = cv.adaptiveThreshold(grayscale_inv,  255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    img_bin_inv = cv.bitwise_not(img_bin)
    
    # show_wait_destroy("thresholded", bw)
    '''
    SEGMENT IMAGE AT GAPS BETWEEN STAFFS
    '''
    gap_heights = segment_staffs(img_bin_inv)
    # for i in range(len(gap_heights) - 1):
    #     show_wait_destroy('Line %d' % i + 1,   bw[gap_heights[i]:gap_heights[i+1]])


    '''
    END SEGMENT IMAGE BETWEEN STAFFS
    '''

    '''
    BEGIN EXAMPLE CODE
    '''
    example = img_bin

    for i in range(len(gap_heights) - 1):
        curr = np.copy(example[gap_heights[i]:gap_heights[i+1]])
        # isolate_notes(curr)
        example[gap_heights[i]:gap_heights[i+1]] = 255
        # print("type of return: ", type(isolate_notes(curr)), type(bw))
        cv.bitwise_and(example[gap_heights[i]:gap_heights[i+1]], isolate_notes(curr), example[gap_heights[i]:gap_heights[i+1]])
        
    show_wait_destroy('result', example)
    # cv.waitKey(0)
    '''
    END EXAMPLE CODE 
    '''
    return 0
    
if __name__ == "__main__":
#  main(sys.argv[1:])
    main()