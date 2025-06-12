import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os
from Base.video import load_config
import seaborn as sns
def draw_text_boxes(image_paths, text_list, added_image_paths, fontsize=30, font_family='Arial', textcolor='#85586F', boxcolor='#FEECE2',
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
        plt.savefig(added_image_paths[i], bbox_inches='tight', pad_inches=0, dpi=300)

def plot_four_images(image_paths, output_file):
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

def figure1_1(motivation_config_path='motivation_config.json'):
    config = load_config(motivation_config_path)
    video_names = config["figure1_1_video_names"]
    data_des_path = config['data_des_path']
    img_dir = config['figure1_1_img_dir']
    figure1_1_data_name = config['figure1_1_data_name']
    result = np.load(os.path.join(data_des_path, figure1_1_data_name + '.npy'))
    F1s = result[:, 2]
    output_file = 'Motivation/figure1_1.pdf'
    text_list = [f'F1 = {f1:.2f}' for f1 in F1s]
    image_paths = [os.path.join(img_dir, video_name + '_first_frame.jpg') for video_name in video_names]
    added_image_paths = [os.path.join(img_dir, video_name + '_first_frame_added.jpg') for video_name in video_names]
    draw_text_boxes(image_paths, text_list, added_image_paths)
    image_paths = [os.path.join(img_dir, video_name + '_first_frame_added.jpg') for video_name in video_names]
    plot_four_images(image_paths, output_file)

def decimal_to_rgb(red_decimal, green_decimal, blue_decimal):

    red = red_decimal / 255
    green = green_decimal / 255
    blue = blue_decimal / 255
    return red, green, blue

def line_plot_figure1_2(score_plot, models_list, output_file):

    font = {'family': 'serif',
            'serif': 'Arial',
            'weight': 'bold',
            'size': 30}
    x_segment = [i for i in range(len(score_plot[0]))]


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
        padded_data = np.pad(score_plot[i,:], (len(kernel)//2, len(kernel)//2), mode='edge')
        smoothed_scores = np.convolve(padded_data, kernel, mode='valid')
        ax.plot(x_segment, smoothed_scores, color=colors[i],
                label=labels[i], linewidth=width, marker=markers[i], ms=markersize, markevery=10)


    ax.set_xlabel('Time (s)', fontname='Arial', fontsize=30, fontweight='bold')
    ax.set_ylabel('Recall', fontname='Arial', fontsize=30, fontweight='bold')
    ax.set_ylim(0,1)
    x_max = len(score_plot[0])
    num_ticks = 4
    tick_interval = (x_max-1) // (num_ticks - 1)
    x_ticks = list(range(0, x_max, tick_interval))
    x_labels = [str(tick) for tick in x_ticks]
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
        label.set_fontweight('bold')
    legend_font_props = {'family': 'Arial', 'size': 25, 'weight': 'bold'}
    ax.legend(ncol=2, loc='upper center', prop=legend_font_props, handletextpad=1, columnspacing=2, handlelength=1)
    txt_font = {'family': 'sans-serif',
                'fontname': 'Arial',
                'weight': 'bold',
                'size': 30}

    image_path = 'Dataset/time_vary/snowing/snowing_frame_21.jpg'
    image = mpimg.imread(image_path)
    height, width = image.shape[:2]
    start_x, start_y = width // 2, height // 2
    cropped_image = image[start_y:, start_x:]
    zoom = 0.2
    oi = OffsetImage(cropped_image, zoom=zoom)

    xy = (30, 0.67)
    ab = AnnotationBbox(oi, xy, frameon=False, boxcoords="data", pad=0)
    ax.add_artist(ab)
    ax.text(xy[0]-12, xy[1]-0.18, 'T = 20s', txt_font)
    image_path = 'Dataset/time_vary/snowing/snowing_frame_81.jpg'
    image = mpimg.imread(image_path)
    height, width = image.shape[:2]
    start_x, start_y = width // 2, height // 2
    cropped_image = image[start_y:, start_x:]
    zoom = 0.2
    oi = OffsetImage(cropped_image, zoom=zoom)
    xy = (90, 0.67)
    ab = AnnotationBbox(oi, xy, frameon=False, boxcoords="data", pad=0)
    ax.add_artist(ab)
    ax.text(xy[0] - 13, xy[1] - 0.18, 'T = 80s', txt_font)
    plt.subplots_adjust(top=0.95, bottom=0.15,left=0.175,right=0.95)
    plt.savefig(output_file)
    plt.close(fig)

def line_plot_figure1_3(score_plot, models_list,output_file):

    font = {'family': 'serif',
            'serif': 'Arial',
            'weight': 'bold',
            'size': 30}

    x_segment = [i for i in range(len(score_plot[0]))]
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc('font', **font)
    ax.tick_params(axis='both', labelsize=30)
    width = 3
    markersize = 12
    markers = ['v', '*', 'X', 'o', 's']
    labels = models_list
    colors = ['#e3716e', '#faa300', '#AD8B73', '#609966', '#CCA8E9']
    kernel = np.ones(5) / 5

    for i, label in enumerate(labels):
        padded_data = np.pad(score_plot[i,:], (len(kernel)//2, len(kernel)//2), mode='edge')
        smoothed_scores = np.convolve(padded_data, kernel, mode='valid')
        ax.plot(x_segment, smoothed_scores, color=colors[i],
                label=labels[i], linewidth=width, marker=markers[i], ms=markersize, markevery=10)


    ax.set_xlabel('Time (s)', fontname='Arial', fontsize=30, fontweight='bold')
    ax.set_ylabel('Recall', fontname='Arial', fontsize=30, fontweight='bold')
    ax.set_ylim(0,1)
    x_max = len(score_plot[0])
    num_ticks = 4
    tick_interval = (x_max-1) // (num_ticks - 1)
    x_ticks = list(range(0, x_max, tick_interval))
    x_labels = [str(tick) for tick in x_ticks]
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
        label.set_fontweight('bold')

    legend_font_props = {'family': 'Arial', 'size': 25, 'weight': 'bold'}
    ax.legend(ncol=2, loc='upper center', prop=legend_font_props, handletextpad=1, columnspacing=2, handlelength=1)
    
    txt_font = {'family': 'sans-serif',
                'fontname': 'Arial',
                'weight': 'bold',
                'size': 30}
    xy = (60, 0.75)
    ax.text(xy[0], xy[1], 'snow intensifies', txt_font, horizontalalignment='center', verticalalignment='center')

    legend_font_props = {'family': 'Arial', 'size': 25, 'weight': 'bold'}
    ax.legend(ncol=3, loc='upper center', prop=legend_font_props, handletextpad=0.2, columnspacing=0.5, handlelength=1)
    plt.subplots_adjust(top=0.95, bottom=0.15,left=0.175,right=0.95)
    plt.savefig(output_file)
    plt.close(fig)

def figure1_2():
    data_matrix = np.load(f'Motivation/data/figure1_2_data.npy')
    models_list = ['YOLOv5-n', 'YOLOv5-s', 'RTMDet-tiny', 'SSD']
    performance_index =1
    data_samples = data_matrix[:,:,performance_index]
    category_labels = models_list

    line_plot_figure1_2(data_samples, category_labels,'Motivation/figure1_2.pdf')

def figure1_3():
    data_matrix = np.load(f'Motivation/data/figure1_3_data.npy')
    models_list = ['YOLOv5-n', 'YOLOv5-s', 'Pre-retrain1', 'Pre-retrain2', 'Pre-retrain3']
    performance_index =1
    data_samples = data_matrix[:,:,performance_index]
    category_labels = models_list
    line_plot_figure1_3(data_samples, category_labels,f'Motivation/figure1_3.pdf')

def figure1_4(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    index = 1
    performance_index = 2
    file_path = os.path.join(data_des_path, f'figure1_4_data_label_{index}.npy')
    data_matrix = np.load(file_path)
    data_matrix = data_matrix[:,:,performance_index]
    data = np.zeros((5,5))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax,
                cbar_kws={"shrink": .82}, annot_kws={"size": 30, "weight": "bold", "family": "Arial"})
    ax.set_xlabel('Visual Model Index', fontsize=30, fontweight='bold', fontname='Arial')
    ax.set_ylabel('Environment Type Index', fontsize=30, fontweight='bold', fontname='Arial')
    ax.tick_params(axis='x', labelsize=30, labelcolor='black')
    ax.tick_params(axis='y', labelsize=30, labelcolor='black')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(30)
        label.set_fontweight('bold')
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=1)  # Adjust the layout
    output_file = f'Motivation/figure1_4.pdf'
    fig.savefig(output_file, format='pdf')

def figure2_1(config_path='motivation_config.json'):
    config = load_config(config_path)
    img_dir = config['figure2_1_img_dir']
    data_des_path = config['data_des_path']
    figure2_1_data_name = config['figure2_1_data_name']
    result = np.load(os.path.join(data_des_path, figure2_1_data_name + '.npy'))
    F1s = result[:, 2]
    text_list = [f'F1 = {f1:.2f}' for f1 in F1s]

    image_paths = [f'{img_dir}/perspective_{i}_first_frame.jpg' for i in range(1, 5)]
    added_image_paths = [f'{img_dir}/perspective_{i}_first_frame_added.jpg' for i in range(1, 5)]
    draw_text_boxes(image_paths, text_list, added_image_paths)

    output_file = 'Motivation/figure2_1.pdf'
    plot_four_images(added_image_paths, output_file)

def bar_plot_figure2_2(data, labels,category_labels, xlabel, ylabel, output_file):

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
    ax.set_ylim(0.6,1)
    ax.set_yticks([0.6,0.7,0.8,0.9,1] )
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

def figure2_2(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    data_name = config['figure2_2_data_name']
    data_matrix = np.load(os.path.join(data_des_path, data_name + '.npy'))
    viewpoint_index = [2, 5, 8, 11]
    # performance_index = 0
    values = data_matrix[:, viewpoint_index]
    labels = ['Dancing','Badminton','Frisbee']
    category_labels = [15,30,45,60]
    xlabel = 'Perspective (°)'
    ylabel = 'mAP'
    output_file = f'Motivation/figure2_2.pdf'
    bar_plot_figure2_2(values, labels,category_labels, xlabel, ylabel, output_file)

if __name__ == '__main__':
    # figure1_1()
    # figure1_2()
    figure1_3()
    # figure1_4()

    # figure2_1()
    # figure2_2()
    pass

