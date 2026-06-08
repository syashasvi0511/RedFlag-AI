# Download the correct dataset
!kaggle datasets download -d jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images
!unzip -q anerbc-anemia-diagnosis-using-rbc-images.zip -d AneRBC-II

# Check the folder structure
import os
for root, dirs, files in os.walk('AneRBC-II'):
    level = root.replace('AneRBC-II', '').count(os.sep)
    if level < 4:
        print(' ' * 2 * level + os.path.basename(root) + '/')
        for f in list(files)[:3]:
            print(' ' * 2 * (level+1) + f)
