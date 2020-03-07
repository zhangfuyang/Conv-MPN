import matplotlib.pyplot as plt
import numpy as np
import colorsys
import matplotlib

# 1. precision  2. recall
# 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.8
data = {
            'gnn': np.load("gnn_loop_1/checkpoint_15_2.071/score.npy", allow_pickle=True).tolist(),
            'per edge classifier': np.load("per_edge_classification_new/BEST_checkpoint/score.npy", allow_pickle=True).tolist(),
            'zero message': np.load("conv_zero_padding/checkpoint_16_2.025/score.npy", allow_pickle=True).tolist(),
            'Conv-MPN(t=1)': np.load("conv_mpn_loop_1/checkpoint_16_2.025/score.npy", allow_pickle=True).tolist(),
            'Conv-MPN(t=2)': np.load("conv_mpn_loop_2_new/checkpoint_62_0.712/score.npy", allow_pickle=True).tolist(),
            'Conv-MPN(t=3)': np.load("conv_mpn_loop_3_pretrain_2/checkpoint_14_0.916/score.npy", allow_pickle=True).tolist()
}

def random_colors(N):
    hsv = [(i / N, 1, 1.0) for i in range(N)]
    hsv[1] = (1 / N, 1, 0)
    return list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))


colors = random_colors(len(data))
shape = ['--', '--', '--', '-', '-', '-']
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
print(len(data))

linewidth = 1.0

#plt.rcParams.update({'font.size': 6, 'font.family': "Times New Roman"})
font = {
        'family': 'Latin Modern Roman',
        'size': 6
}
matplotlib.rc('font', **font)
#plt.rcParams["figure.figsize"] = [10,10]

fig, ax = plt.subplots(3,2,sharex='col')#, figsize=(8,8))#,sharey='row')
#for (m,n), subplot in np.ndenumerate(ax):
#    subplot.set_xlim(0.05, 0.85)
#    subplot.set_ylim(0.05, 0.9)
for idx, name in enumerate(data.keys()):
    precision = data[name]['corner'][0]
    ax[0,0].plot(thresholds, precision, shape[idx], color=colors[idx], label=name, linewidth=linewidth)

ax[0,0].title.set_text('Corner-Precision')
#plt.setp(ax[0,0].get_xticklabels(), visible=False)
#plt.setp(ax[0,0].get_yticklabels(), visible=False)
ax[0,0].tick_params(axis='both', which='both', length=0)


for idx, name in enumerate(data.keys()):
    recall = data[name]['corner'][1]
    ax[0,1].plot(thresholds, recall, shape[idx], color=colors[idx], label=name, linewidth=linewidth)

ax[0,1].title.set_text('Corner-Recall')
#ax[0,1].set_xlabel('Threshold')
#plt.setp(ax[1,0].get_xticklabels(), visible=False)
#plt.setp(ax[1,0].get_yticklabels(), visible=False)
ax[0,1].tick_params(axis='both', which='both', length=0)


for idx, name in enumerate(data.keys()):
    precision = data[name]['edge'][0]
    ax[1,0].plot(thresholds, precision, shape[idx], color=colors[idx], label=name, linewidth=linewidth)

ax[1,0].title.set_text('Edge-Precision')
#plt.setp(ax[0,1].get_xticklabels(), visible=False)
#plt.setp(ax[0,1].get_yticklabels(), visible=False)
ax[1,0].tick_params(axis='both', which='both', length=0)


for idx, name in enumerate(data.keys()):
    recall = data[name]['edge'][1]
    ax[1,1].plot(thresholds, recall, shape[idx], color=colors[idx], label=name, linewidth=linewidth)

ax[1,1].title.set_text('Edge-Recall')
#ax[1,1].set_xlabel('Threshold')
#plt.setp(ax[1,1].get_xticklabels(), visible=False)
#plt.setp(ax[1,1].get_yticklabels(), visible=False)
ax[1,1].tick_params(axis='both', which='both', length=0)


for idx, name in enumerate(data.keys()):
    precision = data[name]['loop'][0]
    ax[2,0].plot(thresholds, precision, shape[idx], color=colors[idx], label=name, linewidth=linewidth)

ax[2,0].title.set_text('Region-Precision')
ax[2,0].set_xlabel('Threshold')
#plt.setp(ax[0,2].get_xticklabels(), visible=False)
#plt.setp(ax[2,0].get_yticklabels(), visible=False)
ax[2,0].tick_params(axis='both', which='both', length=0)


for idx, name in enumerate(data.keys()):
    recall = data[name]['loop'][1]
    ax[2,1].plot(thresholds, recall, shape[idx], color=colors[idx], label=name, linewidth=linewidth)

ax[2,1].title.set_text('Region-Recall')
ax[2,1].set_xlabel('Threshold')
#plt.setp(ax[1,2].get_xticklabels(), visible=False)
#plt.setp(ax[1,2].get_yticklabels(), visible=False)
ax[2,1].tick_params(axis='both', which='both', length=0)
handles, labels = ax[2,1].get_legend_handles_labels()
#leg = fig.legend(handles, labels, loc='lower center', frameon=False, ncol=6)
#leg = fig.legend(handles, labels, loc=(0,0), frameon=False, ncol=6, fontsize='x-large')
#bb = leg.get_bbox_to_anchor().inverse_transformed(ax[1,2].transAxes)
#xoffset = 0
#yoffset = 0
#bb.x0 += xoffset
#bb.x1 += yoffset
#leg.set_bbox_to_anchor(bb, transform=ax[1,2].transAxes)
plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.5)
fig.tight_layout()
#plt.show()

plt.savefig("all.pdf", bbox_inches='tight', dpi=300)
#plt.close()
