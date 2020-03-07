import svgwrite
import random
import colorsys


def random_colors(N, bright=True, same=False, colors=None):
    brightness = 1.0 if bright else 0.7
    if colors is None or same:
        if same:
            hsv = [(0, 1, brightness) for i in range(N)]
        else:
            hsv = [(i / N, 1, brightness) for i in range(N)]
    else:
        hsv = [(colors[i], 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def svg_generate(edges, name, nodes=None, samecolor=False, colors=None, image_link=None):
    dwg = svgwrite.Drawing(name+'.svg', size=(u'128', u'128'))
    shapes = dwg.add(dwg.g(id='shape', fill='black'))
    colors = random_colors(len(edges), same=samecolor, colors=colors)
    if image_link is not None:
        shapes.add(dwg.image(href=image_link, size=(128, 128)))
    for i, edge in enumerate(edges):
        x = edge[0] / 2
        y = edge[1] / 2
        if samecolor:
            shapes.add(dwg.line((int(x[0]), int(x[1])), (int(y[0]), int(y[1])),
                               stroke='red', stroke_width=1, opacity=0.8))
        else:
            shapes.add(dwg.line((int(x[0]), int(x[1])), (int(y[0]), int(y[1])),
                                stroke=svgwrite.rgb(colors[i][0] * 255, colors[i][1] * 255, colors[i][2] * 255, '%'),
                                stroke_width=2))
    if nodes is not None:
        for i, node in enumerate(nodes):
            shapes.add(dwg.circle((int(nodes[i][0] / 2), int(nodes[i][1]) / 2), r=2,
                                  stroke='green', fill='white', stroke_width=1, opacity=0.8))
    return dwg
