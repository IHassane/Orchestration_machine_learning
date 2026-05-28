# 1. Install kaggle CLI


# 2. Set up your API key
#    Go to https://www.kaggle.com/settings -> "Create New Token"
#    This downloads a kaggle.json file
#    Place it at ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<you>\.kaggle\kaggle.json (Windows)

import os
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
# If on Colab, upload your kaggle.json first then:
# !cp kaggle.json ~/.kaggle/kaggle.json
# !chmod 600 ~/.kaggle/kaggle.json

# 3. Download HAM10000
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p /content/ham10000 --unzip