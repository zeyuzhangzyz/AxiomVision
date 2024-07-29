import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle




def draw_text_boxes(image_paths, text_list, fontsize=30, font_family='Arial', textcolor='#85586F', boxcolor='#FEECE2',
                    boxlw=1):

    for i,image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')

        bbox_props = dict(boxstyle="square,pad=0.3", fc=boxcolor, ec=boxcolor, lw=boxlw)
        height, width = img.shape[:2]
        positions = [(25, 100), (width-25, 100), (25, height - 100), (width - 25, height - 100)]
        alignments = [('left', 'bottom'), ('right', 'bottom'), ('left', 'top'), ('right', 'top')]


        ax.text(positions[i][0], positions[i][1], text_list[i], fontsize=fontsize, fontname=font_family,
                color=textcolor, bbox=bbox_props, verticalalignment=alignments[i][1],
                horizontalalignment=alignments[i][0])

        plt.tight_layout(pad=0)  
        plt.savefig(f'figure4_src/light{i+1}_added.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


def decimal_to_rgb(red_decimal, green_decimal, blue_decimal):

    red = red_decimal / 255
    green = green_decimal / 255
    blue = blue_decimal / 255
    return red, green, blue

def bar_plot_figure1(data, labels, category_labels, ylabel, output_file):
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(lw=0.5)
    plt.gca().set_axisbelow(True)
    bar_width = 0.3
    num_bars = len(data[0])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    x_indexes = np.arange(len(category_labels))
    for i in range(num_bars) :
        ax.bar(x_indexes + i * bar_width, data[:,i], bar_width, label=labels[i],
               edgecolor='#f4f1de', lw=1.5, alpha=0.8, color=colors[i % len(colors)], hatch='/')
    ax.set_ylabel(  ylabel, fontname = 'Arial',fontsize=30, fontweight='bold')
    ax.set_xlabel("DNN model", fontname = 'Arial', fontsize=30, fontweight='bold')

    ax.set_xticks(x_indexes + bar_width * (num_bars - 1) / 2)
    ax.set_xticklabels(category_labels, fontsize=20, fontweight='bold')
    ax.set_ylim(min(min(data_category) for data_category in data) -0.1, 1)
    ax.set_ylim(0,1)
    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(20)
        label.set_fontweight('bold')

    for label in  ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
        label.set_fontweight('bold')
    legend_font_props = {'family': 'Arial', 'size': 30, 'weight': 'bold'}
    ax.legend(loc='upper left', prop=legend_font_props)

    # left, bottom, width, height = pos.x0, pos.y0, pos.width, pos.height
    plt.subplots_adjust(top=0.95, bottom=0.15,left=0.175,right=0.95)
    fig.savefig(output_file, format='pdf')
    plt.close(fig)

def line_plot_figure2(score_plot, models_list, output_file):

    font = {'family': 'serif',
            'serif': 'Arial',
            'weight': 'bold',
            'size': 30}
    x_segment = [i for i in range(80)]
    score = score_plot


    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis='both', labelsize=30)

    plt.rc('font', **font)

    width = 3
    markersize = 12
    markers = ['v','*','P','p']
    labels = models_list
    colors = ['#e3716e', '#faa300', '#7ac7e2', '#ff7ed4']
    kernel = np.ones(5) / 5


    for i, label in enumerate(labels):
        new_scores = np.convolve(score[i, :], kernel, mode='valid')
        new_scores = new_scores[:46]
        ax.plot(x_segment[:len(new_scores)], new_scores, color=colors[i],label=labels[i], linewidth=width,marker=markers[i], ms=markersize, markevery=10)


    ax.set_xlabel('Time (s)', fontname='Arial', fontsize=30, fontweight='bold')
    ax.set_ylabel('Recall', fontname='Arial', fontsize=30, fontweight='bold')
    ax.set_ylim(0.2,1)
    ax.set_xticks([0, 15, 30, 45])
    ax.set_xticklabels([0,45,90,135])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
        label.set_fontweight('bold')
    legend_font_props = {'family': 'Arial', 'size': 25, 'weight': 'bold'}
    ax.legend(ncol=2, loc='upper center', prop=legend_font_props, handletextpad=1, columnspacing=5, handlelength=1)
    txt_font = {'family': 'sans-serif',
                'fontname': 'Arial',
                'weight': 'bold',
                'size': 30}


    # load jpg1
    image_path = 'E:/dataset/youtube/perspective4_4/perspective4_4_1.jpg'
    image = mpimg.imread(image_path)
    height, width = image.shape[:2]
    start_x, start_y = width // 2, height // 2
    cropped_image = image[start_y:, start_x:]
    zoom = 0.3
    oi = OffsetImage(cropped_image, zoom=zoom)

    xy = (15, 0.67)
    ab = AnnotationBbox(oi, xy, frameon=False, boxcoords="data", pad=0)
    ax.add_artist(ab)


    ax.text(xy[0]-4, xy[1]-0.15, 'T = 20s', txt_font)

    # load jpg2
    image_path = 'E:/dataset/youtube/perspective4_6/perspective4_6_1.jpg'
    image = mpimg.imread(image_path)
    height, width = image.shape[:2]
    start_x, start_y = width // 2, height // 2
    cropped_image = image[start_y:, start_x:]
    zoom = 0.3
    oi = OffsetImage(cropped_image, zoom=zoom)
    xy = (35, 0.67)
    ab = AnnotationBbox(oi, xy, frameon=False, boxcoords="data", pad=0)
    ax.add_artist(ab)
    ax.text(xy[0] - 4, xy[1] - 0.15, 'T = 100s', txt_font)
    plt.subplots_adjust(top=0.95, bottom=0.15,left=0.175,right=0.95)
    plt.savefig(output_file)
    plt.close(fig)


def line_plot_figure3(score_plot, models_list,output_file):
    score = score_plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis='both', labelsize=30)  # Set tick label font size

    font = {'family': 'serif',
            'serif': 'Arial',
            'weight': 'bold',
            'size': 30}
    plt.rc('font', **font)

    width = 3
    markersize = 12
    markers = ['D', 'X', 'o', 's', 'p']
    labels = models_list
    colors = ['#F2BED1', '#AD8B73', '#609966', '#CCA8E9', '#ff7ed4']
    kernel = np.ones(5) / 5

    for i, label in enumerate(labels):
        new_scores = np.convolve(score[i, :], kernel, mode='valid')
        new_scores = new_scores[:46]
        ax.plot([i for i in range(len(new_scores))], new_scores, color=colors[i],label=labels[i], linewidth=width,marker=markers[i], ms=markersize, markevery=10)


    ax.set_xlabel('Time (s)', fontname='Arial', fontsize=30, fontweight='bold')
    ax.set_ylabel('Recall', fontname='Arial', fontsize=30, fontweight='bold')
    ax.set_ylim(0.1, 0.9)
    ax.set_xticks([0, 15, 30, 45])
    ax.set_xticklabels([0,45,90,135])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
        label.set_fontweight('bold')

    txt_font = {'family': 'sans-serif',
                'fontname': 'Arial',
                'weight': 'bold',
                'size': 30}


    image_path = 'E:/dataset/youtube/perspective4_4/perspective4_4_1.jpg'
    image = mpimg.imread(image_path)

    height, width = image.shape[:2]
    start_x, start_y = width // 2, height // 2
    cropped_image = image[start_y:, start_x:]
    zoom = 0.3
    oi = OffsetImage(cropped_image, zoom=zoom)
    xy = (12, 0.63)

    ax.text(xy[0], xy[1], 'T = 30s gradually\nbegins to snow', txt_font, horizontalalignment='center', verticalalignment='center')

    legend_font_props = {'family': 'Arial', 'size': 25, 'weight': 'bold'}
    ax.legend(ncol=3, loc='upper center', prop=legend_font_props, handletextpad=0.2, columnspacing=0.5, handlelength=1)
    plt.subplots_adjust(top=0.95, bottom=0.15,left=0.175,right=0.95)
    plt.savefig(output_file)
    plt.close(fig)

def figure1():
    data_matrix = np.load('data/figure1_data.npy')
    model_selection_list = np.array([0, 5, 7, 1, 2, 6])
    labels = ['Nighttime', 'Daytime']
    models_list = ['YOLOv5-n', 'YOLOv5-s', 'YOLOv5-m', 'YOLOv5-l', 'YOLOv5-x', 'SSD', 'Faster\nRCNN', 'RTMDet\ntiny']
    performance_index = 1
    gt_index = 0
    selected_data = data_matrix[gt_index, model_selection_list, :, performance_index]
    category_labels = [models_list[i] for i in model_selection_list]
    output_filename = f'figure1.pdf'
    bar_plot_figure1(selected_data, labels, category_labels, 'Recall', output_filename)


def figure2():
    data_matrix = np.load('data/figure2_data.npy')
    models_list = ['YOLOv5-n', 'YOLOv5-s', 'YOLOv5-m', 'YOLOv5-l', 'YOLOv5-x', 'SSD', 'Faster\nRCNN', 'RTMDet\ntiny']
    gt_index = 1
    performance_index =1
    index = f"{gt_index}{performance_index}"

    data_matrix = data_matrix[gt_index, ... , performance_index]
    data_matrix = data_matrix.reshape(8, 60)
    model_selection_list = [0,1,5,7]
    data_samples = data_matrix[model_selection_list,:]
    category_labels = [models_list[i] for i in model_selection_list]
    line_plot_figure2(data_samples, category_labels,'figure2.pdf')


def figure3():
    data_matrix = np.load('data/figure3_data.npy')
    gt_index = 1
    performance_index =1
    data_matrix = data_matrix[gt_index, ... , performance_index]
    data_matrix = data_matrix.reshape(5, 60)
    category_labels = ['Pre-retrain1','Pre-retrain2','Pre-retrain3','Pre-retrain4','YOLOv5s']
    line_plot_figure3(data_matrix, category_labels,'figure3.pdf')

def figure4():
    """
    Insert four images into a plot at manually specified positions and export as a single PDF file.
    """

    output_file = 'figure4.pdf'
    text_list = ['F1 = 0.72', 'F1 = 0.67', 'F1 = 0.51', 'F1 = 0.45']
    image_paths = [f'figure4_src/light{i}.jpg' for i in range(1, 5)]
    draw_text_boxes(image_paths, text_list)
    image_paths = [f'figure4_src/light{i}_added.jpg' for i in range(1, 5)]
    fig = plt.figure(figsize=(10, 8))
    left, bottom, width, height = 0.125,0.15,0.775,0.8
    left = 0.075
    width = 0.85
    right,upper = left + width, height + bottom

    img_width = 0.4
    img_height = 0.38

    axes_positions = [
        [left, upper-img_height, img_width, img_height],
        [right-img_width, upper-img_height, img_width, img_height],
        [left, bottom, img_width, img_height],
        [right-img_width, bottom, img_width, img_height]
    ]

    for path, pos in zip(image_paths, axes_positions):
        ax = fig.add_axes(pos)
        img = mpimg.imread(path)
        ax.imshow(img, aspect='auto')
        ax.axis('off')

    plt.subplots_adjust(top=0.95, bottom=0.15,left=0.175,right=0.95)
    fig.savefig(output_file, format='pdf')
    plt.close(fig)

def figure5():
    text_list = ['F1 = 0.76', 'F1 = 0.85', 'F1 = 0.51', 'F1 = 0.61']
    image_paths = [f'figure5_src/perspective_{i}.jpg' for i in range(1, 5)]
    draw_text_boxes(image_paths, text_list)

    image_paths = [f'figure5_src/perspective_{i}_added.jpg' for i in range(1, 5)]
    output_file = 'figure5.pdf'
    fig = plt.figure(figsize=(10, 8))

    left, bottom, width, height = 0.125,0.15,0.775,0.8
    left = 0.075
    width = 0.85
    right,upper = left + width, height + bottom

    img_width = 0.4
    img_height = 0.38

    axes_positions = [
        [left, upper-img_height, img_width, img_height],
        [right-img_width, upper-img_height, img_width, img_height],
        [left, bottom, img_width, img_height],
        [right-img_width, bottom, img_width, img_height]
    ]


    for path, pos in zip(image_paths, axes_positions):
        ax = fig.add_axes(pos)
        img = mpimg.imread(path)
        ax.imshow(img, aspect='auto')
        ax.axis('off')


    plt.subplots_adjust(top=0.95, bottom=0.15,left=0.175,right=0.95)
    fig.savefig(output_file, format='pdf')
    plt.close(fig)


def bar_plot_figure6(data, labels,category_labels, xlabel, ylabel, output_file):

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(lw=0.5)
    plt.gca().set_axisbelow(True)

    bar_width = 0.25
    num_bars = len(data)


    colors_decimal = [
        (244, 111, 68),  # green
        (127, 203, 164),  # blue
        (75, 101, 176)  # yellow
    ]
    colors = [decimal_to_rgb(*color) for color in colors_decimal]
    x_indexes = np.arange(len(category_labels))
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    for i in range(num_bars):
        ax.bar(x_indexes + (i-1) * bar_width, data[i], bar_width, label=labels[i],
               edgecolor='#f4f1de', lw=1.5, alpha=0.8, color=colors[i], hatch=hatches[i])
    ax.set_ylabel(ylabel, fontname = 'Arial',fontsize=30, fontweight='bold')
    ax.set_xlabel(xlabel, fontname = 'Arial', fontsize=30, fontweight='bold')
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(category_labels, fontsize=30, fontweight='bold')
    ax.set_ylim(0.75,0.95)
    ax.set_yticks([0.75,0.8,0.85,0.9,0.95] )
    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(20)
        label.set_fontweight('bold')
    for label in  ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
        label.set_fontweight('bold')
    legend_font_props = {'family': 'Arial', 'size': 30, 'weight': 'bold'}
    ax.legend(loc='upper center',ncol=3, prop=legend_font_props, handletextpad=0.2, columnspacing=0.5, handlelength=1)

    plt.subplots_adjust(top=0.95, bottom=0.15,left=0.175,right=0.95)
    fig.savefig(output_file, format='pdf')
    plt.close(fig)

def figure6():
    file_path = 'data/figure6_data_segmentation.npy'
    # A_Dancer
    # Two_Badminton_Players
    # Two_Frisbee_Players
    data_matrix = np.load(file_path)
    release_index = 0
    viewpoint_index = [2, 5, 8, 11]
    category_index = [1, 7, 11]
    performance_index =0
    data_matrix = data_matrix[:,:,release_index,performance_index]
    values = data_matrix[category_index, :]
    values = values[:, viewpoint_index]

    labels = ['Dancing','Badminton','Frisbee']
    category_labels = [15,30,45,60]
    xlabel = 'Perspective (°)'
    ylabel = 'mAP'
    output_file = f'figure6.pdf'
    bar_plot_figure6(values, labels,category_labels, xlabel, ylabel, output_file)

if __name__ == '__main__':

    # figure1()
    # figure2()
    # figure3()
    # figure4()
    # figure5()
    # figure6()
    pass

