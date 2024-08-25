from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from Base.video import *
from Base.performance import *

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



def run_data_figure1(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config["source_dir"]
    video_names = config["figure1_video_names"]
    versions = config["yolov5_versions"]
    run_different_dnn_models(source_dir,video_names,versions)


def save_data_figure1(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    versions = config['performance_versions']
    gts = config['gts']
    dnns = config['dnns']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    video_names = config['figure1_video_names']
    figure1_data_name = config['figure1_data_name']
    figure1_data = np.zeros((len(gts),len(dnns),len(video_names),5))
    for gt_index, gt in enumerate(gts):
        for dnn_index,dnn in enumerate(dnns):
            figure1_data[gt_index,dnn_index] = videos_performance_accumulate(video_names, dnn, versions[dnn_index],label , confidence_threshold,is_free_viewpoint = 0 , gt = gt)
    np.save(os.path.join(data_des_path, figure1_data_name + '.npy'),figure1_data)

def run_data_figure2(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config["source_dir"]
    video_names = config["figure2_video_names"]
    versions = config["yolov5_versions"]
    run_different_dnn_models(source_dir,video_names,versions)

def save_data_figure2(config_path='motivation_config.json'):
    config = load_config(config_path)
    gts = config['gts']
    dnns = config['dnns']
    video_names = config["figure2_video_names"]
    versions = config['performance_versions']
    data_des_path = config['data_des_path']
    figure2_data_name = config['figure2_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    figure2_data = np.zeros((len(gts),len(dnns),len(video_names), 20, 5))
    for gt_index, gt in enumerate(gts):
        for dnn_index,dnn in enumerate(dnns):
            figure2_data[gt_index, dnn_index] = segment_performance(video_names, dnn, versions[dnn_index], label, confidence_threshold, is_free_viewpoint = 0, gt=gt)

    np.save(os.path.join(data_des_path, figure2_data_name + '.npy'),figure2_data)


def run_data_figure3(config_path='motivation_config.json'):

    config = load_config(config_path)
    source_dir = config["source_dir"]
    video_names = config["figure3_video_names"]
    versions = config["figure3_versions"]
    for version in versions:
        yolov5_retrained(source_dir, video_names, version)
    run_different_dnn_models(source_dir,video_names,["s","x"])
def save_data_figure3(config_path='motivation_config.json'):
    config = load_config(config_path)
    video_names = config["figure3_video_names"]
    versions = config["figure3_versions"]
    gts = config['gts']
    data_des_path = config['data_des_path']
    figure3_data_name = config['figure3_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])

    figure3_data = np.zeros((len(gts),len(versions), len(video_names), 20, 5))
    for gt_index, gt in enumerate(gts):
        for version_index,version in enumerate(versions):
            figure3_data[gt_index, version_index] = segment_performance(video_names, 'yolov5', version, label, confidence_threshold, is_free_viewpoint = 0, gt = gt)
    np.save(os.path.join(data_des_path, figure3_data_name + '.npy'),figure3_data)



def run_data_figure4(config_path='motivation_config.json'):
    config = load_config(config_path)
    video_names = config["figure4_video_names"]
    versions = config["figure4_versions"]
    source_dir = config["source_dir"]
    run_different_dnn_models(source_dir, video_names, versions)

def save_data_figure4(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    figure4_data_name = config['figure4_data_name']
    video_names = config["figure4_video_names"]
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    figure4_data = videos_performance_accumulate(video_names, 'yolov5', 's' , label, confidence_threshold,is_free_viewpoint = 1, gt = 'yolov5')
    np.save(os.path.join(data_des_path, figure4_data_name + '.npy'),figure4_data)

def run_data_figure5(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config['source_dir']
    video_names = config["figure5_video_names"]
    versions = config["figure5_versions"]
    run_different_dnn_models(source_dir, video_names, versions)

def save_data_figure5(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    video_names = config["figure5_video_names"]
    gts = config['gts']
    versions = config["figure5_versions"]
    figure5_data_name = config['figure5_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    figure5_data = np.zeros((len(gts),len(versions),len(video_names),5))
    for gt_index, gt in enumerate(gts):
        for version_index,version in enumerate(versions):
            figure5_data[gt_index,version_index] = videos_performance_accumulate(video_names, 'yolov5', version, label, confidence_threshold ,is_free_viewpoint = 0 , gt = gt)
    np.save(os.path.join(data_des_path, figure5_data_name + '.npy'),figure5_data)


def run_segmentation(image_path, save_dir, config_path, model_path):
    command = f'conda activate Paddle & python D:/code/PycharmProjects/Relighting-Base-Environment/tools/predict.py --config {config_path} --model_path {model_path} --image_path {image_path} --save_dir {save_dir}'
    os.system(command)

def figure6_resize_videos(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        for i in range(12):
            name = str(i) + '.mp4'
            input_name = os.path.join(input_path, name)
            output_name = os.path.join(output_path, name)
            only_change_resolution(input_name, output_name, 720, 480)
    extract_frames(input_path, os.path.join(input_path, 'extract_frames'))
    extract_frames(output_path, os.path.join(output_path, 'extract_frames'))

def segmentation_imgs(path, config_path='motivation_config.json'):
    config = load_config(config_path)
    segmentation_dnns = config["figure6_segmentation_dnns"]
    segmentation_versions = config["figure6_segmentation_versions"]
    segmentation_params_dir = config["figure6_segmentation_params_dir"]
    segmentation_config_dir = config["figure6_segmentation_config_dir"]
    segmentation_model_dir = config["figure6_segmentation_model_dir"]
    for dnn_index, dnn in enumerate(segmentation_dnns):
        dir = os.path.join(path, dnn)
        run_segmentation(os.path.join(path, 'extract_frames'), dir, os.path.join(segmentation_config_dir,
              segmentation_versions[dnn_index]), os.path.join(segmentation_model_dir, segmentation_params_dir[dnn_index],"model.pdparams"))

def get_num(stdpath, prefix, suffix):
    count = 0
    for filename in os.listdir(stdpath):
        if filename.startswith(prefix) and filename.endswith(suffix):
            count += 1
    return count

def main_segmentation(config_path='motivation_config.json'):
    config = load_config(config_path)
    folder_path = config["figure6_source_dir"]
    subfolder_names = get_subfolder_names(folder_path)
    releases =  config["figure6_releases"]
    for file_name in subfolder_names:
        for release in releases:
            path = os.path.join(folder_path, file_name, release)
            segmentation_imgs(path, config_path='motivation_config.json')

def save_segmentation_data(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config["data_des_path"]
    video_names = config["figure6_video_names"]
    folder_path = config['figure6_source_dir']
    versions = config["figure6_versions"]
    releases = config["figure6_releases"]
    data_name =  config["figure6_data_name"]
    data_matrix = np.zeros((len(video_names),12, len(versions),3))

    for video_index, file_name in enumerate(video_names):
        for release_index, release in enumerate(releases):
            path = os.path.join(folder_path, file_name, release)
            for i in range(12):
                stdpath = os.path.join(path, "server1", "pseudo_color_prediction")
                testpath = os.path.join(path, "seq_lite", "pseudo_color_prediction")
                src_name = f"{i}_frame"
                output_name  = f"{i}_frame"
                data_matrix[video_index, i, release_index] = segment_F1(stdpath, testpath, src_name, output_name)
    np.save(os.path.join(data_des_path, data_name + '_segmentation.npy'),data_matrix)


def main_detection(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config["figure6_source_dir"]
    versions = config["figure6_versions"]
    video_names = config["figure6_video_names"]
    releases =  config["figure6_releases"]
    for file_name in video_names:
        for version_index, version in enumerate(versions):
            input_path = os.path.join(source_dir, file_name, releases[version_index])
            extract_frames(input_path, os.path.join(input_path, 'extract_frames'))
            run_yolo_imgs(os.path.join(input_path, 'extract_frames'), f"{file_name}_{version}", 's')
            run_yolo_imgs(os.path.join(input_path, 'extract_frames'), f"{file_name}_{version}", 'x')


def save_detection_data(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config["data_des_path"]
    video_names = config["figure6_video_names"]
    yolov5_free_viewpoint_project_dir = config['yolov5_free_viewpoint_project_dir']
    versions = config["figure6_versions"]
    label = int(config['people_label'])
    confidence_threshold = float(config['confidence_threshold'])
    data_name =  config["figure6_data_name"]
    data_matrix = np.zeros((len(video_names),12, len(versions),5))
    for video_index, file_name in enumerate(video_names):
        for i in range(12):
            for version_index, version in enumerate(versions):
                stdpath = os.path.join(yolov5_free_viewpoint_project_dir, f"{file_name}_{version}_x", "labels")
                testpath = os.path.join(yolov5_free_viewpoint_project_dir, f"{file_name}_{version}_s",  "labels")
                src_name = f"{i}_frame"
                output_name  = f"{i}_frame"
                print(file_name,i,version)
                data_matrix[video_index, i, version_index, :] = performance_accumulate(stdpath, testpath, src_name,
                    output_name, label=label, threshold=confidence_threshold)
    np.save(os.path.join(data_des_path, data_name + '_detect.npy'),data_matrix)

def run_data_figure6(config_path='motivation_config.json'):
    config = load_config(config_path)
    folder_path = config["figure6_source_dir"]
    subfolder_names = get_subfolder_names(folder_path)
    for file_name in subfolder_names:
        input_path = os.path.join(folder_path, file_name, 'RGB')
        output_path = os.path.join(folder_path, file_name, 'create1')
        figure6_resize_videos(input_path, output_path)
    main_detection()
    main_segmentation()

def save_data_figure6():
    save_detection_data()
    save_segmentation_data()



def train_model_figure7(config_path='motivation_config.json'):
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
        # process_images_with_gamma_and_noise(source_images_dir, target_images_dir, gamma)
        # copy_labels(source_labels_dir, target_labels_dir)
        new_name = f"{dataset_name}_{description}"
        # update_and_save_yaml(yolov5_train_config_dir, dataset_name, new_name, os.path.join(base_target_dir, description))
        versions.append(new_name)

    for version in versions:
        yolo_train(epoch, version)


def prepare_figure7_video(config_path='motivation_config.json'):
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

    # for video_name in video_names:
    #     set_video_frame(video_src_path, video_name , video_des_path, f"{video_name}_short", 3000)
    #     extract_frames_single(video_des_path, f"{video_name}_short.mp4",img_des_path, f"{video_name}_short")
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



def run_data_figure7(config_path='motivation_config.json'):
    config = load_config(config_path)
    pans = config["pans"]
    tilts = config["tilts"]
    video_names = config['free_viewpoint_video_names']
    brightness_descriptions = config["brightness_descriptions"]
    video_names_short = [f"{video_name}_short_{brightness}_{pan}_{tilt}" for pan in pans for tilt in tilts  for video_name in video_names for brightness in brightness_descriptions ]
    source_dir = config['free_viewpoint_img_des_path']
    # brightness_descriptions = config["brightness_descriptions"]
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

    # video_names_short = [f"1_1min_10min_short_Natural_Light_0_0"]
    # video_names_short = [f"{video_name}_short_{brightness}_{pan}_{tilt}" for pan in pans for tilt in tilts for
    #                      video_name in video_names for brightness in brightness_descriptions]
    # versions = config["figure8_yolov5_versions"]
    # dnns = config["figure8_dnns"]
    # run_different_dnn_models_images(source_dir, video_names_short, versions)

def task_wrapper(args):
    return videos_element_accumulate(*args)

def save_data_figure7(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    video_names = config['free_viewpoint_video_names']
    brightness_descriptions = config["brightness_descriptions"]
    brightness_descriptions.pop(5)
    pans = config["pans"]
    tilts = config["tilts"]
    video_names_env = [[f"{video_name}_short_{brightness}_{pan}_{tilt}" for pan in pans for tilt in tilts for
                         video_name in video_names] for brightness in brightness_descriptions]

    figure7_data_name = config['figure7_data_name']
    confidence_threshold = float(config['confidence_threshold'])
    dataset_name = config["dataset_name"]
    versions = []
    epoch = 100

    for description in brightness_descriptions:
        new_name = f"{dataset_name}_{description}_{epoch}"
        versions.append(new_name)
    labels = [int(config['car_label']), int(config['people_label'])]
    for label in labels:
        figure7_data = np.zeros((len(versions),len(video_names_env),5))
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
            figure7_data[version_index, video_names_bright_index, :] = result
        np.save(os.path.join(data_des_path, figure7_data_name + f'_{label}.npy'),figure7_data)



def process_single_file(args):
    tmp_data = np.zeros((100, 5))
    video_name, pan, env_light, dnn, version, label = args
    src_name = video_name
    output_name = video_name
    video_totalname = f'{video_name}_{env_light}_{pan}_0'
    stdpath = get_txt_path(video_totalname, "yolov5", 'x')
    testpath = get_txt_path(video_totalname, dnn, version)
    for count in range(1, 101):
        stdtxt = f'{src_name}_{pan}_0_frame_{count}'
        testtxt = f'{output_name}_{pan}_0_frame_{count}'
        tmp_data[count - 1, :] = performance_element(stdpath, testpath, stdtxt, testtxt, label=label, threshold=0.5)
    # print(tmp_data)
    return tmp_data

def save_data_figure8(config_path='motivation_config.json'):
    config = load_config(config_path)
    brightness_descriptions = config["brightness_descriptions"]
    pans = config["pans"]
    figure8_dnn_versions = config["figure8_dnn_versions"]
    figure8_versions = config["figure8_versions"]

    dataset_name = config["dataset_name"]
    versions = []
    dnns = []
    epoch = 100
    for description in brightness_descriptions:
        new_name = f"{dataset_name}_{description}_{epoch}"
        versions.append(new_name)
        dnns.append("yolov5")
    versions = versions + figure8_versions
    dnns = dnns + figure8_dnn_versions
    video_names = [f'{i}_1min_10min_short' for i in range(1, 5)]
    num_envs = len(brightness_descriptions)
    num_versions = len(versions)
    num_counts = 100
    labels = [int(config['car_label']), int(config['people_label'])]
    for label in labels:
        figure8_data = np.zeros((len(video_names), len(pans), num_envs, num_versions, num_counts, 5))
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
            figure8_data[video_index, pan_index, env_index, version_index, :, :] = result
        np.save(f"data_label_{label}.npy", figure8_data)

def test_data_figure8():
    label = 0
    figure8_data = np.load(f"data_label_{label}.npy")
    data_label_2 = np.sum(figure8_data, axis = (0,1,2,4))
    print(data_label_2)



if __name__ == "__main__":

    # run_data_figure1()
    # run_data_figure2()
    # run_data_figure3()
    # run_data_figure4()
    # run_data_figure5()
    # run_data_figure6()
    # prepare_figure7_video()
    # train_model_figure7()
    # run_data_figure7()


    # save_data_figure1()
    # save_data_figure2()
    # save_data_figure3()
    # save_data_figure4()
    # save_data_figure5()
    # save_data_figure6()
    # save_data_figure7()
    save_data_figure8()
    # test_data_figure8()
    # train_model_figure7()

    pass