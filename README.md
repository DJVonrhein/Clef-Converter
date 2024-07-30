# Clef-Converter
Takes images of sheet music, and moves the notes up or down the staff, automatically. Useful for changing from bass clef to treble clef, for example.

This work-in-progress python project uses OpenCV to isolate different details in a sheet of piano music. The overall goal is to isolate the notes and the staff (as well as other details) from each other cleanly, and to have compatibility with even the most complex sheets (i.e. img/remiFullPage.png , a detailed image from Reminiscences de Norma).

 After some improvements, this project can be applied to other things:
   
  - auto-transposing notes to different keys 
  - reading notes in a different clef.


Instructions:

  From command line on your computer, do 
  
    git pull (this url)
    
  After the project files are downloaded, enter the project directory. You now need a python 3.7 environment with the correct dependencies installed. For simplicity, you can try it through pip:   
  
    pip install -r requirements.txt  
    
  If you have the right environment, it can be run with   
  
    python3 example_improved.py 

  
