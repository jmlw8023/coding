# -*- encoding: utf-8 -*-
'''
@File    :   operation_xml.py
@Time    :   2022/07/28 20:41:44
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
'''

# import packets


import os
import lxml 
import lxml.etree  as etree
# 创建一个xml格式的文件
def create_xmls(img_dicts, suffix='.xml'):
    file = str(img_dicts[-1]['filename']) 
    xml_path = (img_dicts[-1]['xml_path'])[:-4]
    xml_file = os.path.join(xml_path + suffix)
    # print(xml_file)
    
    # domtree = minidom.parse(xml_file)
    # root = domtree.documentElement
     # create Annatation root
    root = etree.Element('annatation') 
    # root.text  = root.text + newline

    # create filename and path
    etree.SubElement(root, 'filename').text = str(img_dicts[-1]['filename'])  
    # etree.SubElement(root, 'path').text = str(img_dicts[-1]['path'])  
    # or
    path = etree.SubElement(root, 'path')   
    path.text =  img_dicts[-1]['path']   

    # create source
    source = etree.SubElement(root, 'source')               
    etree.SubElement(source, 'database').text = 'Unknown'    
    
    # create size
    sizes = etree.SubElement(root, 'size')  
    etree.SubElement(sizes, 'width').text = str(img_dicts[-1]['img_w'])    
    etree.SubElement(sizes, 'height').text = str(img_dicts[-1]['img_h'])   
    etree.SubElement(sizes, 'depth').text = str(img_dicts[-1]['img_d'])    

    # create segmented
    etree.SubElement(root, 'segmented').text = '0'   

    na = []
    na.clear()
    # for label_dict in range(len(img_dicts[:-1])):
    for num in range(len(img_dicts[:-1])):
        na.append(str(img_dicts[num]['name']))
        # create object node
        obj = etree.SubElement(root, 'object') 
        etree.SubElement(obj, 'name').text = str(img_dicts[num]['name'] )     
        etree.SubElement(obj, 'pose').text = 'Unspecified'       
        etree.SubElement(obj, 'truncated').text = '0'            
        etree.SubElement(obj, 'difficult').text = str(img_dicts[num]['difficult'])   
        # create bndbox node
        bndbox = etree.SubElement(obj, 'bndbox')   
        etree.SubElement(bndbox, 'xmin').text = str(int(img_dicts[num]['xmin']))     
        etree.SubElement(bndbox, 'ymin').text = str(int(img_dicts[num]['ymin']))     
        etree.SubElement(bndbox, 'xmax').text = str(int(img_dicts[num]['xmax']))    
        etree.SubElement(bndbox, 'ymax').text = str(int(img_dicts[num]['ymax']))   

    tree = etree.ElementTree(root)
        # print(root)
    # tree.write(xml_file, encoding='utf-8', pretty_print=True, xml_declaration=False)
    tree.write(xml_file, encoding='utf-8', pretty_print=True, xml_declaration=False)
    print(f'finish write {file} file successful, total {(len(na))} num , name is ---{na}------\n\n')


import os
import xml.etree.ElementTree as ET
# from xml.etree import ElementTree as ET
# 创建一个xml格式的文件, 这个没有换行格式
def create_xmls_old(img_dicts, save_path='./', name='result', suffix='.xml', indent='\t', newline='\n'):
    
    xml_path = (img_dicts[-1]['xml_path'])[:-4]
    xml_file = os.path.join(xml_path + suffix)
    print(xml_file)
    
    # domtree = minidom.parse(xml_file)
    # root = domtree.documentElement

    # print(img_dicts)
    print(len(img_dicts))
    print(' - '*20)
    # print(img_dicts[-1]['filename'])
     # create Annatation root
    root = ET.Element('annatation') 
    # root.text  = root.text + newline

    # create filename and path
    ET.SubElement(root, 'filename').text = str(img_dicts[-1]['filename']) + newline
    # ET.SubElement(root, 'path').text = str(img_dicts[-1]['path'])  
    # or
    path = ET.SubElement(root, 'path')   
    path.text =  img_dicts[-1]['path']  + newline

    # create source
    source = ET.SubElement(root, 'source')               
    ET.SubElement(source, 'database').text = 'Unknown'   + newline
    
    # create size
    sizes = ET.SubElement(root, 'size')  
    ET.SubElement(sizes, 'width').text = str(img_dicts[-1]['img_w'])   + newline
    ET.SubElement(sizes, 'height').text = str(img_dicts[-1]['img_h'])  + newline
    ET.SubElement(sizes, 'depth').text = str(img_dicts[-1]['img_d'])   + newline

    # create segmented
    ET.SubElement(root, 'segmented').text = '0'  + newline

    # for label_dict in range(len(img_dicts[:-1])):
    for num in range(len(img_dicts[:-1])):
        print('#' * 30)
        print(img_dicts[num])
        # create object node
        obj = ET.SubElement(root, 'object') 
        ET.SubElement(obj, 'name').text = img_dicts[num]['name']     + newline
        ET.SubElement(obj, 'pose').text = 'Unspecified'      + newline
        ET.SubElement(obj, 'truncated').text = '0'           + newline
        ET.SubElement(obj, 'difficult').text = str(img_dicts[num]['difficult'])  + newline
        # create bndbox node
        bndbox = ET.SubElement(obj, 'bndbox')   
        ET.SubElement(bndbox, 'xmin').text = str(int(img_dicts[num]['xmin']))    + newline
        ET.SubElement(bndbox, 'ymin').text = str(int(img_dicts[num]['ymin']))    + newline
        ET.SubElement(bndbox, 'xmax').text = str(int(img_dicts[num]['xmax']))    + newline
        ET.SubElement(bndbox, 'ymax').text = str(int(img_dicts[num]['ymax']))    + newline

        tree = ET.ElementTree(root)
        # print(root)
        tree.write(xml_file, encoding='utf-8', xml_declaration=False)

# 备案xml.dom.minidom，没有写完的
from xml.dom.minidom import Document
def make_xmls(img_dicts, suffix='.xml'):
    # 文件路径和名称
    xml_path = (img_dicts[-1]['xml_path'])[:-4]
    xml_file = os.path.join(xml_path + suffix)
    print(xml_file)

    # 创建xml格式
    root = Document()
    annotation = root.createElement('annotation')
    root.appendChild(annotation)

    filename = root.createElement('annotation')
    root.appendChild(annotation)
    annotation = root.createElement('annotation')
    root.appendChild(annotation)

    folder = root.createElement('folder')
    root.appendChild(folder)
    segmented = root.createTextNode('segmented')
    folder.appendChild(segmented)


