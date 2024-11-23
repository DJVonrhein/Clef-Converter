# Clef-Converter
Switching between reading bass clef and treble clef can be tricky. This program takes an image of sheet music and converts all bass clef music to treble, or vice versa. It is even designed to handle piano music which usually has both present.

Examples below (images get sized up for better template matching):


![Input Image](<img src="[https://github.com/favicon.ico](https://github.com/DJVonrhein/Clef-Converter/blob/main/examples/obsessed.png)">)![Output Image](<img src="[https://github.com/favicon.ico](https://github.com/DJVonrhein/Clef-Converter/blob/main/examples/obsessed-to-treble.png)">)

![Input Image](<img src="[https://github.com/favicon.ico](https://github.com/DJVonrhein/Clef-Converter/blob/main/examples/takefive.png)" >)![Output Image](<img src="[https://github.com/favicon.ico](https://github.com/DJVonrhein/Clef-Converter/blob/main/examples/takefive-to-bass.png)">)

This python project uses OpenCV to isolate the symbols present in sheet music in order to move the notes and accidentals and swap clefs for simple reading.

Instructions:

  From command line on your computer, do 
  
    git pull (this url)
    
  After the project files are downloaded, enter the project directory. You now need a python 3.7 environment with the correct dependencies installed. For simplicity, you can try it through pip:   
  
    pip install -r requirements.txt  

  Pytesseract is the one library in this project that has an extra step to add to your environment. It requires you to first install tesseract, either through github or homebrew.
    
  Once your environment is set correctly, run clefconverter.py from the command line with either command: 
  
    python3 clefconverter.py (input_image_name) (output_image_name) treble 
    
    or
    
    python3 clefconverter.py (input_image_name) (output_image_name) bass 

  
