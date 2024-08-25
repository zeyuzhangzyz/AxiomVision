from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from Algorithm.video import *
from Algorithm.performance import *
import seaborn as sns
import matplotlib.pyplot as plt

def run_different_dnn_models(source_dir,video_names,versions):
    # faster_rcnn(source_dir, video_names)
    # mmdetection(source_dir, video_names)
    ssd(source_dir, video_names)
    for version in versions:
        yolov5(source_dir, video_names, version)
    pass

def run_different_dnn_models_images(source_dir,video_names,versions):
    faster_rcnn_images(source_dir, video_names)
    mmdetection_images(source_dir, video_names)
    ssd_images(source_dir, video_names)
    for version in versions:
        yolov5(source_dir, video_names, version)
    pass

def train_model(config_path='experiment_prepare_config.json'):
    config = load_config(config_path)
    gamma_values = config["gamma_values"]
    brightness_descriptions = config["brightness_descriptions"]
    yolov5_train_config_dir = config["yolov5_train_config_dir"]
    yolov5_train_source_dir = config["yolov5_train_source_dir"]
    dataset_name = config["dataset_name"]
    source_images_dir = os.path.join(yolov5_train_source_dir, dataset_name, "images", "train2017")
    source_labels_dir = os.path.join(yolov5_train_source_dir, dataset_name, "labels", "train2017")
    base_target_dir = os.path.join(yolov5_train_source_dir, f"{dataset_name}_different_lights")   
  
    epoch = 100
    versions = []
    for description, gamma in zip(brightness_descriptions, gamma_values):
        target_images_dir = os.path.join(base_target_dir, description, "images", "train2017")
        target_labels_dir = os.path.join(base_target_dir, description, "labels", "train2017")
        print(source_images_dir)
        print(target_images_dir)
        process_images_with_gamma_and_noise(source_images_dir, target_images_dir, gamma)
        copy_labels(source_labels_dir, target_labels_dir)
        new_name = f"{dataset_name}_{description}"
        update_and_save_yaml(yolov5_train_config_dir, dataset_name, new_name, os.path.join(base_target_dir, description))
        versions.append(new_name)

    for version in versions:
        yolo_train(epoch, version)


def prepare_video(config_path='experiment_prepare_config.json'):
    config = load_config(config_path)
    pans = config["pans"]
    tilts = config["tilts"]
    brightness_descriptions = config["brightness_descriptions"]
    gamma_values = config["gamma_values"]
    video_names = config["free_viewpoint_video_names"]
    video_src_path = config["free_viewpoint_video_src_path"]
    video_des_path = config["free_viewpoint_video_des_path"]
    img_des_path = config["free_viewpoint_img_des_path"]

    video_names_short = [f"{video_name}_short" for video_name in video_names]
    for video_name in video_names:
        set_video_frame(video_src_path, video_name , video_des_path, f"{video_name}_short", 3000)
        extract_frames_single(video_des_path, f"{video_name}_short.mp4",img_des_path, f"{video_name}_short")
    tasks = []
    for brightness, gamma in zip(brightness_descriptions, gamma_values):
        for video_name in video_names_short:
            for pan in pans:
                for tilt in tilts:
                    video_folder = os.path.join(img_des_path, video_name)
                    output_folder = os.path.join(img_des_path, f'{video_name}_{brightness}_{pan}_{tilt}')
                    tasks.append((video_folder, output_folder, gamma, pan, tilt))

    # Use ProcessPoolExecutor to process tasks in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        futures = {executor.submit(process_vr_brightness_images, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(f"Task {task} completed successfully.")
            except Exception as e:
                print(f"Task {task} generated an exception: {e}")



def run_data(config_path='experiment_prepare_config.json'):
    config = load_config(config_path)
    pans = config["pans"]
    tilts = config["tilts"]
    video_names = config['free_viewpoint_video_names']
    brightness_descriptions = config["brightness_descriptions"]
    video_names_short = [f"{video_name}_short_{brightness}_{pan}_{tilt}" for pan in pans for tilt in tilts  for video_name in video_names for brightness in brightness_descriptions ]
    source_dir = config['free_viewpoint_img_des_path']
    dataset_name = config["dataset_name"]
    versions = []
    epoch = 100
    for description in brightness_descriptions:
        new_name = f"{dataset_name}_{description}_{epoch}"
        versions.append(new_name)

    retrained_tasks = [(source_dir, video_names_short, version) for version in versions]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(yolov5_retrained, *task): task for task in retrained_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(f"Task {task} completed successfully.")
            except Exception as e:
                print(f"Task {task} generated an exception: {e}")
    video_names_short = [f"{video_name}_short_{brightness}_{pan}_{tilt}" for pan in pans for tilt in tilts for
                         video_name in video_names for brightness in brightness_descriptions]
    versions = config["yolov5_versions"]
    run_different_dnn_models_images(source_dir, video_names_short, versions)

def task_wrapper(args):
    return videos_element_accumulate(*args)



def different_models_plot():
    file_path = 'data_0.npy'
    data_matrix1 = np.load(file_path)
    file_path = 'data_2.npy'
    data_matrix2 = np.load(file_path)
    idx = 1
    data_matrix = data_matrix1 + data_matrix2
    summed_matrix = np.zeros((5,5,5))
    for i in range(0, 10, 2):
        for j in range(0, 10, 2):
            summed_matrix[i // 2, j // 2, :] = data_matrix[i, j, :] + data_matrix[i + 1, j, :] + data_matrix[i,
                                                                                                     j + 1,
                                                                                                     :] + data_matrix[
                                                                                                          i + 1, j + 1,
                                                                                                          :]
    result = element2result(summed_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(result[:,:,idx], annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax,
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
    output_file = f'heatmap.pdf'
    fig.savefig(output_file, format='pdf')



def save_data_for_heatmap(config_path='experiment_prepare_config.json'):
    # we set the tilts to be [0] because we only have one tilt in the experiment, if you want control the pans and tilts in the experiment, you can modify the code
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    video_names = config['free_viewpoint_video_names']
    brightness_descriptions = config["brightness_descriptions"]
    brightness_descriptions.pop(5) # to let the number of brightness descriptions be an even number
    pans = config["pans"]
    tilts = config["tilts"]
    video_names_env = [[f"{video_name}_short_{brightness}_{pan}_{tilt}" for pan in pans for tilt in tilts for
                         video_name in video_names] for brightness in brightness_descriptions]

    data_name = config['heatmap_data_name']
    confidence_threshold = float(config['confidence_threshold'])
    dataset_name = config["dataset_name"]
    versions = []
    epoch = 100

    for description in brightness_descriptions:
        new_name = f"{dataset_name}_{description}_{epoch}"
        versions.append(new_name)
    labels = [int(config['car_label']), int(config['people_label'])]
    for label in labels:
        data = np.zeros((len(versions),len(video_names_env),5))
        tasks = []
        for version_index,version in enumerate(versions):
            for video_names_bright_index, video_names_bright in enumerate(video_names_env):
                tasks.append(
                    (video_names_bright, 'yolov5', version, label, confidence_threshold, 0, 'yolov5', '_frame', '_frame'))
        with ThreadPoolExecutor(max_workers=14) as executor:
            results = list(executor.map(task_wrapper, tasks))
        for i, result in enumerate(results):
            version_index = i // (len(versions))
            video_names_bright_index = i % (len(versions))
            data[version_index, video_names_bright_index, :] = result
        np.save(os.path.join(data_des_path, data_name + f'_{label}.npy'),data)

    different_models_plot()

def process_single_file(args):
    tmp_data = np.zeros((100, 5))
    video_name, pan, tilt, env_light, dnn, version, label = args
    src_name = video_name
    output_name = video_name
    video_totalname = f'{video_name}_{env_light}_{pan}_0'
    stdpath = get_txt_path(video_totalname, "yolov5", 'x')
    testpath = get_txt_path(video_totalname, dnn, version)
    for count in range(1, 101):
        stdtxt = f'{src_name}_{pan}_{tilt}_frame_{count}'
        testtxt = f'{output_name}_{pan}_{tilt}_frame_{count}'
        tmp_data[count - 1, :] = performance_element(stdpath, testpath, stdtxt, testtxt, label=label, threshold=0.5)
    # print(tmp_data)
    return tmp_data

def save_data_for_main(config_path='experiment_prepare_config.json'):
    config = load_config(config_path)
    brightness_descriptions = config["brightness_descriptions"]
    pans = config["pans"]
    append_dnn_versions = config["append_dnn_versions"]
    append_versions = config["append_versions"]
    data_name = config["data_name"]
    dataset_name = config["dataset_name"]
    source_video_names = config["free_viewpoint_video_names"]
    versions = []
    dnns = []
    epoch = 100
    for description in brightness_descriptions:
        new_name = f"{dataset_name}_{description}_{epoch}"
        versions.append(new_name)
        dnns.append("yolov5")
    versions = versions + append_versions
    dnns = dnns + append_dnn_versions
    video_names = [f'{video_name}_short' for video_name in source_video_names]
    num_envs = len(brightness_descriptions)
    num_versions = len(versions)
    num_counts = 100
    labels = [int(config['car_label']), int(config['people_label'])]
    for label in labels:
        data = np.zeros((len(video_names), len(pans), num_envs, num_versions, num_counts, 5))
        tasks = []
        for video_index, video_name in enumerate(video_names):
            for pan_index, pan in enumerate(pans):
                for env_index, env_light in enumerate(brightness_descriptions):
                    for version_index, version in enumerate(versions):
                        tasks.append((video_name, pan, env_light, dnns[version_index], version, label))
        with ProcessPoolExecutor(max_workers=14) as executor:
            results = list(executor.map(process_single_file, tasks))
        for i, result in enumerate(results):
            video_index = i // (num_envs * num_versions * len(pans))
            pan_index = (i // (num_envs * num_versions)) % len(pans)
            env_index = (i // (num_versions)) % num_envs
            version_index = i % num_versions
            data[video_index, pan_index, env_index, version_index, :, :] = result
        np.save(f"{data_name}_{label}.npy", data)



def calculate_average_jpg_size(folder_path):
    total_size = 0
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            file_count += 1

    if file_count > 0:
        average_size = total_size / file_count
    else:
        average_size = 0

    return average_size


# def save_size_for_main(config_path='experiment_prepare_config.json'):
#     config = load_config(config_path)
#     pans = config["pans"]
#     tilts = config["tilts"]
#     brightness_descriptions = config["brightness_descriptions"]
#     gamma_values = config["gamma_values"]
#     video_names = config["free_viewpoint_video_names"]
#     video_src_path = config["free_viewpoint_video_src_path"]
#     video_des_path = config["free_viewpoint_video_des_path"]
#     img_des_path = config["free_viewpoint_img_des_path"]

#     video_names_short = [f"{video_name}_short" for video_name in video_names]

#     img_size = np.zeros((len(video_names_short),len(gamma_values), len(pans),len(tilts)))
#     for i, video_name in enumerate(video_names_short):
#         for env_light_index, env_light in enumerate(brightness_descriptions):
#             for pan_index,pan in enumerate(pans):
#                 for tilt_index, tilt in enumerate(tilts):
#                     target_images_dir = os.path.join(img_des_path, f'{video_name}_{env_light}_{pan}_{tilt}')
#                     img_size[i,env_light_index,pan_index,tilt_index] = calculate_average_jpg_size(target_images_dir)/ 1024
#     np.save('img_size.npy', img_size)


if __name__ == "__main__":



    # train_model()
    # prepare_video()
    # run_data()
    save_data_for_heatmap()
    save_data_for_main()
    # save_size_for_main()
    pass