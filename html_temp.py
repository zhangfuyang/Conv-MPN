import sys
import os

HTML_NAME = './demo.html'
svg_dirs = "./conv_mpn_loop_3_pretrain_2/svg/pred_corner"
mask_dirs = "/local-scratch/fuyang/geometry-primitive-detector/result/valid"
f = open(HTML_NAME, 'w')
f.write("<html>\n")
size = 800
for i, name in enumerate(os.listdir(svg_dirs)):
    f.write('<img src="' + os.path.join('/local-scratch/fuyang/cities_dataset/rgb', name[:-4]+'.jpg') + '" width="' + str(size) + '" style="position: absolute; '
            'top: ' + str(i*size) + 'px; '
            'left: 0px ">')
    f.write('<img src="' + os.path.join(svg_dirs, name) + '" width="' + str(size) + '" style="position: absolute; '
            'top: ' + str(i*size) + 'px; '
            'left: 0px ">')
    f.write('<img src="' + os.path.join(mask_dirs, name[:-4]+'.jpg.jpg') + '" width="' + str(size) + '" style="position: absolute; '
                                                                                      'top: ' + str(i*size) + 'px; '
                                                                                      'left: ' + str(size+20) + 'px ">')
    if i > 200:
        break


f.write("</html>")
