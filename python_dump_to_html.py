import numpy as np
import sys
import argparse
import os

def dump_html(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', default='/local-scratch/fza49/cities_dataset/rgb')
    parser.add_argument('--SVG-dir', default='./svg')
    parser.add_argument('--num', type=int, default=400)
    parser.add_argument('--shape', type=int, default=1000)
    args = parser.parse_args(args)

    HTML_NAME = './demo.html'
    f = open(HTML_NAME, 'w')
    f.write("<html>\n")
    f.write('<head><style>\n' +
                '.same { position: absolute; top: 0; left: 0; }\n' +
                '.top { position: relative; top: 0; left: 0; }\n' +
            '</style> </head>\n')
    with open('/local-scratch/fza49/cities_dataset/valid_list.txt', 'r') as ff:
        datalist = ff.readlines()
    svg_dirs = args.SVG_dir.split(' ')
    _ycount = 0
    f.write('<p>')
    for svg_i in range(len(svg_dirs)):
        try:
            conf = svg_dirs[svg_i].split('models_mix_gt_')[1].split('/')[0]
        except:
            conf = ""
        f.write('<font size="1" style="position:absolute; top: 0px; left: ' + str(svg_i * args.shape) + 'px ">' + conf + '</font>')
    f.write('</p>')
    metriclists = []
    for svg_i in range(len(svg_dirs)):
        try:
            tf = open(os.path.join(svg_dirs[svg_i], '..', 'metric.txt'), 'r')
            metriclist = tf.readlines()
            metriclists.append(metriclist)
            tf.close()
        except:
            metriclists.append(["", "", "", "", ""])
    f.write('<p>')
    for svg_i in range(len(svg_dirs)):
        f.write('<font size="1" style="position:absolute; top: 10px; left: ' + str(svg_i * args.shape) + 'px ">' + metriclists[svg_i][0][:-1] + '</font>')
    f.write('</p>')
    f.write('<p>')
    for svg_i in range(len(svg_dirs)):
        f.write('<font size="1" style="position:absolute; top: 20px; left: ' + str(svg_i * args.shape) + 'px ">' + metriclists[svg_i][1][:-1] + '</font>')
    f.write('</p>')
    f.write('<p>')
    for svg_i in range(len(svg_dirs)):
        f.write('<font size="1" style="position:absolute; top: 30px; left: ' + str(svg_i * args.shape) + 'px ">' + metriclists[svg_i][2][:-1] + '</font>')
    f.write('</p>')
    for idx, svgname in enumerate(datalist):
        name = svgname[:-1]
        img_path = os.path.join(args.img_dir, name + '.jpg')

        files = []
        for i in range(len(svg_dirs)):
            SVG_DIR = svg_dirs[i]
            files.append(os.path.abspath(os.path.join(SVG_DIR, name + '.svg')))
        flag = True
        for file in files:
            if os.path.exists(file) is False:
                flag = False
                break
        if flag is False:
            continue
        f.write('<p>\n')
        _count = 0
        for t in range(len(files)):
            f.write('<img src="' + img_path + '" width="' + str(args.shape) + '" style="position: absolute; '
                                              'top: ' + str(_ycount * args.shape + 40) + 'px; '
                                            'left: ' + str(_count * args.shape) + 'px ">')
            _count += 1
        f.write('<font size="9" style="position:absolute; top: ' + str(_ycount * args.shape + 340) + 'px; '
                                            'left: ' + str(_count * args.shape) + 'px;"/>' + svgname + '</font>')
        f.write('</p>\n')
        _ycount += 1
        if idx > args.num:
            break
    _ycount = 0
    for idx, svgname in enumerate(datalist):
        name = svgname[:-1]
        files = []
        for i in range(len(svg_dirs)):
            SVG_DIR = svg_dirs[i]
            files.append(os.path.abspath(os.path.join(SVG_DIR, name + '.svg')))
        flag = True
        for file in files:
            if os.path.exists(file) is False:
                flag = False
                break
        if flag is False:
            continue
        _count = 0
        for t in range(len(files)):
            f.write('<img src="' + files[t] + '" width="' + str(args.shape) + '" style="position: absolute; '
                                              'top: ' + str(_ycount * args.shape + 40) + 'px; '
                                            'left: ' + str(_count * args.shape) + 'px;"/>')
            _count += 1
        _ycount += 1
        if idx > args.num:
            break

    f.write("</html>")


if __name__ == "__main__":
    dump_html(sys.argv[1:])
