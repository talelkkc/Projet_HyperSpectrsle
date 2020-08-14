import xml.etree.ElementTree as ET

def _Tree_ (file):
    tree = ET.parse(file)
    root = tree.getroot()
    samples = root[0][4].text #1024
    lines = root[0][8].text # 1020
    return (samples,lines)


