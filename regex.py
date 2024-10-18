
# usage: python regex.py <file_name>

import re
import sys

filePath = sys.argv[1]

def findMarathonTimes(text):
    out = re.findall("[2-9]:\\d{2}[:.]\\d{2}", text)   
    return out

with open(filePath, "r", encoding="utf-8") as f:
            text = f.read()
            times = findMarathonTimes(text)
            for t in times:
                    print(t)
            f.close()

# Author: Bartosz Piotr Ciereszyński
