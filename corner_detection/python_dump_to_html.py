import numpy as np
import sys
import argparse
import os

def dump_html(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', default='/local-scratch/fza49/cities_dataset/rgb')
    parser.add_argument('--SVG-dir', default='./svg')
    parser.add_argument('--num', type=int, default=100)

    HTML_NAME = 'test.html'
    f = open(HTML_NAME, 'w')
    args = parser.parse_args(args)
    f.write("<html>\n")
    f.write('<head><style>\n' + 
                '.same { position: relative; top: 0; left: 0; }\n' + 
                '.top { position: relative; top: 0; left: -100px; }\n' +
            '</style> </head>\n')
    for idx, svgname in enumerate(os.listdir(args.SVG_dir)):
        name = svgname[:-6]
        img_path = os.path.join(args.img_dir, name + '.jpg')
        svg_path = os.path.join(args.SVG_dir, svgname)
        #if idx % 4 == 0:
        #    f.write('<div style="position: relative; left: 0; top: 0;">\n')
        f.write('<img src="' + img_path + '" width="100" class="same" alt="cfg"/>')
        f.write('<img src="' + svg_path + '" width="100" class="top" alt="cfg"/>')
        #if idx %4 == 3:
        #    f.write('</div>\n')

    f.write("</html>")


if __name__ == "__main__":
    dump_html(sys.argv[1:])
