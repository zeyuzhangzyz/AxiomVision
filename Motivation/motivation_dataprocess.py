from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from Base.video import *
from Base.performance import *
from pathlib import Path
from tqdm import tqdm
def run_different_dnn_models(source_dir,video_names, other_dnns, versions, env_version = []):
    if 'faster_rcnn' in other_dnns:
        faster_rcnn(source_dir, video_names)
    if 'mmdetection' in other_dnns:
        mmdetection(source_dir, video_names)
    if 'ssd' in other_dnns:
        ssd(source_dir, video_names)
    for version in versions:
        yolov5(source_dir, video_names, version)
    for version in env_version:
        # print(version)
        yolov5_retrained(source_dir, video_names, version)
    pass

def run_different_dnn_models_images(source_dir,video_names, other_dnns, versions, env_version = []):
    if 'faster_rcnn' in other_dnns:
        faster_rcnn_images(source_dir, video_names)
    if 'mmdetection' in other_dnns:
        mmdetection_images(source_dir, video_names)
    if 'ssd' in other_dnns:
        ssd_images(source_dir, video_names)
    for version in versions:
        yolov5(source_dir, video_names, version)
    for version in env_version:
        yolov5_retrained(source_dir, video_names, version)
    pass

def run_data_figure1_1(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config['figure1_1_source_dir']
    video_names = config["figure1_1_video_names"]
    versions = config["figure1_1_versions"]
    for version in versions:
        yolov5(source_dir, video_names, version)

def save_data_figure1_1(config_path='motivation_config.json'):
    config = load_config(config_path)       
    source_dir = config['figure1_1_source_dir']
    video_names = config["figure1_1_video_names"]
    img_dir = config['figure1_1_img_dir']
    ensure_dir(img_dir)
    for video_name in video_names:
        save_first_frame(os.path.join(source_dir, video_name) + '.mp4', os.path.join(img_dir, video_name + '_first_frame.jpg'))
    data_des_path = config['data_des_path']
    video_names = config["figure1_1_video_names"]
    gt = 'yolov5'
    version = 'n'
    figure1_1_data_name = config['figure1_1_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    figure1_1_data = videos_performance_accumulate(video_names, 'yolov5', version, label, confidence_threshold ,is_free_viewpoint = 0 , gt = gt)
    np.save(os.path.join(data_des_path, figure1_1_data_name + '.npy'), figure1_1_data)


def run_data_figure1_2(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config["figure1_2_source_dir"]
    video_name = config["figure1_2_video_name"]
    versions = config["figure1_2_versions"]  # 
    other_dnns = config["figure1_2_other_dnns"]
    extract_frames_single(source_dir, video_name + '.mp4', source_dir, f"{video_name}")
    run_different_dnn_models(source_dir, [video_name], other_dnns, versions)


def save_data_figure1_2(config_path='motivation_config.json'):
    config = load_config(config_path)
    other_dnns = config['figure1_2_other_dnns']
    gt = 'faster_rcnn'
    other_dnns.remove(gt)
    video_name = config["figure1_2_video_name"]
    yolov5_versions = config['figure1_2_versions']
    versions = ["" for _ in range(len(other_dnns))] + yolov5_versions
    dnns = other_dnns + ['yolov5' for _ in range(len(yolov5_versions))]
    data_des_path = config['data_des_path']
    figure1_2_data_name = config['figure1_2_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    time_length = config['time_length']
    figure1_2_data = np.zeros((len(dnns), time_length, 5))

    tasks = []
    for dnn, version in zip(dnns, versions):
        tasks.append((video_name, dnn, version, label, confidence_threshold, gt, 30, time_length))

    with tqdm(total=len(tasks), desc='Overall Progress', position=0) as pbar:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(segment_performance_wrapper, tasks), 
                              total=len(tasks), 
                              desc='Processing Tasks',
                              position=1,
                              leave=False))
            pbar.update(len(tasks))

    for i, result in enumerate(results):
        figure1_2_data[i] = result
    print(figure1_2_data)
    np.save(os.path.join(data_des_path, figure1_2_data_name + f'.npy'), figure1_2_data)


def run_data_figure1_3(config_path='motivation_config.json'):
    # some part of figure1_3 is the same as figure1_2
    # so the source_dir and video_name are the same as figure1_2
    # and if you have run figure1_2, you can skip versions (yolov5s, yolov5n)
    config = load_config(config_path)
    source_dir = config["figure1_2_source_dir"]
    video_name = config["figure1_2_video_name"]
    versions = []
    other_dnns = []
    # versions = config["figure1_2_versions"] # comment this two lines if you have run figure1_2
    # other_dnns = ["faster_rcnn"]

    env_versions = ["snowy_100","RTTS_100"]
    brightness_descriptions = ["Extremely_Bright"]
    dataset_name = config["dataset_name"]
    epoch = 100
    for description in brightness_descriptions:
        new_name = f"{dataset_name}_{description}_{epoch}"
        env_versions.append(new_name)

    run_different_dnn_models(source_dir, [video_name], other_dnns, versions, env_version =env_versions)

def segment_performance_wrapper(args):
    video_name, dnn, version, label, confidence_threshold, gt, gap, segments = args
    print(args)
    return segment_performance(video_name, dnn, version, label, confidence_threshold, gt=gt, gap=gap, segments=segments)

def save_data_figure1_3(config_path='motivation_config.json'):
    config = load_config(config_path)
    gt = 'faster_rcnn'
    video_name = config["figure1_2_video_name"]
    versions = config['figure1_2_versions'] + ["RTTS_100", "snowy_100"]
    dataset_name = config["dataset_name"]
    epoch = 100
    description = "Extremely_Bright"
    new_name = f"{dataset_name}_{description}_{epoch}"
    versions.append(new_name)
    dnns = ["yolov5" for _ in range(len(versions))]
    data_des_path = config['data_des_path']
    figure1_3_data_name = config['figure1_3_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    time_length = config['time_length']
    figure1_3_data = np.zeros((len(dnns), time_length, 5))

    tasks = []
    for dnn, version in zip(dnns, versions):
        tasks.append((video_name, dnn, version, label, confidence_threshold, gt, 30, time_length))

    figure1_3_data = np.zeros((len(dnns), time_length, 5))

    with tqdm(total=len(tasks), desc='Overall Progress', position=0) as pbar:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(segment_performance_wrapper, tasks), 
                              total=len(tasks), 
                              desc='Processing Tasks',
                              position=1,
                              leave=False))
            pbar.update(len(tasks))

    for i, result in enumerate(results):
        figure1_3_data[i] = result
    np.save(os.path.join(data_des_path, figure1_3_data_name + f'.npy'), figure1_3_data)


def run_data_figure2_1(config_path='motivation_config.json'):
    # run the yolov5 model and save the results
    config = load_config(config_path)
    source_dir = config['figure2_1_source_dir']
    video_names = config["figure2_1_video_names"]
    versions = config["figure2_1_versions"]
    for version in versions:
        yolov5(source_dir, video_names, version)

def save_data_figure2_1(config_path='motivation_config.json'):
    config = load_config(config_path)       
    source_dir = config['figure2_1_source_dir']
    video_names = config["figure2_1_video_names"]
    img_dir = config['figure2_1_img_dir']
    ensure_dir(img_dir)
    for video_name in video_names:
        save_first_frame(os.path.join(source_dir, video_name) + '.mp4', os.path.join(img_dir, video_name + '_first_frame.jpg'))
    data_des_path = config['data_des_path']
    video_names = config["figure2_1_video_names"]
    gt = 'yolov5'
    version = 's'
    figure2_data_name = config['figure2_1_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    figure2_data = videos_performance_accumulate(video_names, 'yolov5', version, label, confidence_threshold ,is_free_viewpoint = 0 , gt = gt)
    np.save(os.path.join(data_des_path, figure2_data_name + '.npy'), figure2_data)



def run_segmentation(image_path, save_dir, config_path, model_path):
    command = f'conda activate Paddle & python DNN/PaddleSeg/tools/predict.py --config {config_path} --model_path {model_path} --image_path {image_path} --save_dir {save_dir}'
    os.system(command)

def get_num(stdpath, prefix, suffix):
    count = 0
    for filename in os.listdir(stdpath):
        if filename.startswith(prefix) and filename.endswith(suffix):
            count += 1
    return count

def run_data_figure2_2(config_path='motivation_config.json'):
    # include two parts:
    # 1. extract frames
    # 2. segmentation
    config = load_config(config_path)
    folder_path = config["figure2_2_source_dir"]
    video_names = config["figure2_2_video_names"]
    for file_name in video_names:
        extract_frames(os.path.join(folder_path, file_name, "RGB"), os.path.join(folder_path, file_name, 'extract_frames'))
    current_dir = Path.cwd()

    for file_name in video_names:
        image_path = os.path.join(current_dir, folder_path, file_name, "extract_frames")
        segmentation_dnns = config["segmentation_dnns"]
        segmentation_versions = config["segmentation_versions"]
        segmentation_params_dir = config["segmentation_params_dir"]
        segmentation_config_dir = config["segmentation_config_dir"]
        segmentation_model_dir = config["segmentation_model_dir"]
        for dnn_index, dnn in enumerate(segmentation_dnns):
            save_dir = os.path.join(current_dir, folder_path, file_name, dnn)
            config_path = os.path.join(current_dir, segmentation_config_dir, segmentation_versions[dnn_index])
            model_path = os.path.join(current_dir, segmentation_model_dir, segmentation_params_dir[dnn_index],"model.pdparams")
            run_segmentation(image_path, save_dir, config_path, model_path)


def save_data_figure2_2(config_path='motivation_config.json'):
    #calculate the IOU of the segmentation results and save
    config = load_config(config_path)
    data_des_path = config["data_des_path"]
    video_names = config["figure2_2_video_names"]
    folder_path = config['figure2_2_source_dir']
    data_name =  config["figure2_2_data_name"]
    data_matrix = np.zeros((len(video_names), 12))
    
    for video_index, file_name in enumerate(video_names):
        path = os.path.join(folder_path, file_name)
        for i in range(12):
            stdpath = os.path.join(path, "server1", "pseudo_color_prediction")
            testpath = os.path.join(path, "seq_lite", "pseudo_color_prediction")
            src_name = f"{i}_frame"
            output_name  = f"{i}_frame"
            data_matrix[video_index, i] = segment_IOU(stdpath, testpath, src_name, output_name)
    np.save(os.path.join(data_des_path, data_name + '.npy'),data_matrix)

def prepare_env_data(config_path='motivation_config.json'):
    config = load_config(config_path)
    brightness_descriptions = config['brightness_descriptions']
    gamma_values = config['gamma_values']

    dataset_name = config['dataset_name']
    source_images_dir = f'Dataset/{dataset_name}/images/train2017'
    source_labels_dir = f'Dataset/{dataset_name}/labels/train2017'
    base_target_dir = f'Dataset/{dataset_name}_different_lights'
    original_yaml_path = config['dataset_yaml_path']
    project_root = Path(__file__).parent.parent.absolute()
    current_path = project_root.joinpath("Dataset")
    base_target_dir = current_path.joinpath(f"{dataset_name}_different_lights")
    processor = DatasetProcessor()

    for description, gamma in zip(brightness_descriptions, gamma_values):
        target_images_dir = os.path.join(base_target_dir, description, 'images/train2017')
        target_labels_dir = os.path.join(base_target_dir, description, 'labels/train2017')
        processor.process_images_with_gamma_and_noise(source_images_dir, target_images_dir, gamma)
        processor.copy_labels(source_labels_dir, target_labels_dir)

    processor.update_and_save_yaml(original_yaml_path, brightness_descriptions, base_target_dir)
    processor.update_and_save_yaml_single(original_yaml_path, "RTTS", current_path)
    processor.update_and_save_yaml_single(original_yaml_path, "snowy", current_path)



def train_env_model(config_path='motivation_config.json'):
    config = load_config(config_path)
    brightness_descriptions = config["brightness_descriptions"]
    dataset_name = config["dataset_name"]
    epoch = 100
    versions = [f"{dataset_name}_{description}" for description in brightness_descriptions]
    for version in versions:
        yolo_train(epoch, version)

def build_tasks(video_names, brightness_descriptions, gamma_values, pans, tilts, VR_img_src_path, VR_img_des_path):
    tasks = []
    for video_name in video_names:
        for brightness, gamma in zip(brightness_descriptions, gamma_values):
            for pan in pans:
                for tilt in tilts:
                    video_img_src_path = os.path.join(VR_img_src_path, video_name)
                    output_folder = os.path.join(VR_img_des_path, f'{video_name}_{brightness}_{pan}_{tilt}')
                    tasks.append((video_img_src_path, output_folder, brightness, gamma, pan, tilt))
    return tasks
def VR_data_process(config_path='motivation_config.json'):
    # 1. extract frames
    # 2. process brightness, pan and tilt to new frames
    # It will take a long time to run the code below, so you'd better run it one by one.
    config = load_config(config_path)
    pans = config["pans"]
    tilts = config["tilts"]
    video_names = config['VR_video_names']
    VR_video_src_path = config['VR_video_src_path']
    brightness_descriptions = config["brightness_descriptions"]
    VR_img_src_path = config['VR_img_src_path']
    VR_img_des_path = config['VR_img_des_path']
    gamma_values = config["gamma_values"]

    # part 1

    for video_name in video_names:
        extract_frames_single(VR_video_src_path, f"{video_name}.mp4",VR_img_src_path, f"{video_name}")


    # part 2  process brightness, pan and tilt to new frames

    tasks = build_tasks(video_names, brightness_descriptions, gamma_values, pans, tilts, VR_img_src_path, VR_img_des_path)

    with ProcessPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        futures = {executor.submit(process_vr_brightness_images, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(f"Task {task} completed successfully.")
            except Exception as e:
                print(f"Task {task} generated an exception: {e}")


def run_data_figure1_4(config_path='motivation_config.json'):
    config = load_config(config_path)
    pans = config["pans"]
    tilts = config["tilts"]
    video_names = config['VR_video_names']
    brightness_descriptions = config["brightness_descriptions"]
    VR_img_des_path = config['VR_img_des_path']
    dataset_name = config["dataset_name"]

    versions = config["yolov5_versions"]
    epoch = 100
    env_versions = []
    for description in brightness_descriptions:
        new_name = f"{dataset_name}_{description}_{epoch}"
        env_versions.append(new_name)

    video_names_output = [f"{video_name}_{brightness}_{pan}_{tilt}" for pan in pans for tilt in tilts  for video_name in video_names for brightness in brightness_descriptions ]
    other_dnns = config["overall_other_dnns"]
    run_different_dnn_models_images(VR_img_des_path, video_names_output, other_dnns, versions, env_versions)



def task_wrapper(args):
    return videos_element_accumulate(*args)


def process_single_file(args):
    video_name, pan, tilt, brightness, dnn, version, label, gt_dnn,gt_version = args
    txt_path = get_txt_path(f"{video_name}_{brightness}_{pan}_{tilt}", dnn, version)
    gt_path = get_txt_path(f"{video_name}_{brightness}_{pan}_{tilt}", gt_dnn, gt_version)
    tmp_data = np.zeros((100, 5))
    for count in range(1, 101):
        txt_name = f"{video_name}_{brightness}_{pan}_{tilt}_frame_{count}"
        stdpath = gt_path
        testpath = txt_path
        tmp_data[count - 1, :] = performance_element(stdpath, testpath, txt_name, txt_name, label=label, threshold=0.5)
    return tmp_data

    
def save_overall_data(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    video_names = config['VR_video_names']
    brightness_descriptions = config["brightness_descriptions"]
    pans = config["pans"]
    tilts = config["tilts"]
    dataset_name = config["dataset_name"]
    versions = []
    dnns = []

    epoch = 100
    for description in brightness_descriptions:
        new_name = f"{dataset_name}_{description}_{epoch}"
        versions.append(new_name)
        dnns.append("yolov5")
    
    versions = versions + ['m', 'l', '', '', 'x', '']
    dnns = dnns + ["yolov5", "yolov5", "mmdetection", "ssd", "yolov5", "faster_rcnn"]
    num_envs = len(brightness_descriptions)
    num_versions = len(versions)
    num_counts = 100
    labels = [int(config['car_label']), int(config['people_label'])]
    # labels = [int(config['people_label'])]
    num_perspectives = len(pans)*len(tilts)
    perspectives = [(pan, tilt) for pan in pans for tilt in tilts]
    tasks = []
    gt_dnn = "yolov5"
    gt_version = "x"
    # gt_dnn = "faster_rcnn"
    # gt_version = ""
    for label in labels:
        overall_data = np.zeros((len(video_names), num_perspectives, num_envs, num_versions, num_counts, 5))
        tasks = []
        total_processed = 0

        batch_size = 1000
        for video_index, video_name in enumerate(video_names):
            for perspective_index, perspective in enumerate(perspectives):
                for env_index, env_light in enumerate(brightness_descriptions):
                    for version_index, version in enumerate(versions):
                        tasks.append((video_name, perspective[0], perspective[1], env_light, 
                                    dnns[version_index], version, label, gt_dnn, gt_version))
                        
                        if len(tasks) >= batch_size:
                            with ProcessPoolExecutor(max_workers=14) as executor:
                                results = list(tqdm(executor.map(process_single_file, tasks), 
                                                  total=len(tasks),
                                                  desc=f'Processing label {label} batch'))
                            
                            for i, result in enumerate(results):
                                current_index = total_processed + i
                                video_index = current_index // (num_envs * num_versions * num_perspectives)
                                perspective_index = (current_index // (num_envs * num_versions)) % num_perspectives
                                env_index = (current_index // (num_versions)) % num_envs
                                version_index = current_index % num_versions
                                overall_data[video_index, perspective_index, env_index, version_index, :] = result
                            
                            total_processed += len(tasks)
                            tasks = []
        
        if tasks:
            with ProcessPoolExecutor(max_workers=14) as executor:
                results = list(tqdm(executor.map(process_single_file, tasks), 
                                  total=len(tasks),
                                  desc=f'Processing label {label} final batch'))
            
            for i, result in enumerate(results):
                current_index = total_processed + i
                video_index = current_index // (num_envs * num_versions * num_perspectives)
                perspective_index = (current_index // (num_envs * num_versions)) % num_perspectives
                env_index = (current_index // (num_versions)) % num_envs
                version_index = current_index % num_versions
                overall_data[video_index, perspective_index, env_index, version_index, :] = result
        np.save(os.path.join(data_des_path, f'overall_data_label_{label}.npy'), overall_data)

def save_data_figure1_4(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    labels = [int(config['car_label']), int(config['people_label'])]
    index1 = [labels[0], labels[0], labels[1]]
    index2 = [labels[0], labels[1], labels[1]]
    for index in range(3):
        data_label1 = np.load(os.path.join(data_des_path, f'overall_data_label_{index1[index]}.npy'))
        data_label2 = np.load(os.path.join(data_des_path, f'overall_data_label_{index2[index]}.npy'))
        data = data_label1 + data_label2
        data = np.sum(data, axis=(0,1,4))
        data = data[:,:11,:]
        data = np.concatenate((data[:,:5,:], data[:,6:11,:]), axis=1)
        data = np.concatenate((data[:5,:,:], data[6:11,:,:]), axis=0)
        len1 = data.shape[0]
        len2 = data.shape[1]
        row = 5
        column = 5
        data_matrix = np.zeros((row,column,5))
        for i in range(row):
            for j in range(column):
                target_row = (len1-1) * i / (row-1)
                target_col = (len2-1) * j / (column-1)
                row1 = int(np.floor(target_row))
                row2 = min(row1 + 1, len1-1)
                col1 = int(np.floor(target_col))
                col2 = min(col1 + 1, len2-1)
                w_row = target_row - row1
                w_col = target_col - col1
                data_matrix[i,j,:] = (1-w_row)*(1-w_col)*data[row1,col1,:] + \
                                    w_row*(1-w_col)*data[row2,col1,:] + \
                                    (1-w_row)*w_col*data[row1,col2,:] + \
                                    w_row*w_col*data[row2,col2,:]
        accumulate_data = element2result(data_matrix)
        np.save(os.path.join(data_des_path, f'figure1_4_data_label_{index}.npy'), accumulate_data)


def train_snowy_model(config_path='motivation_config.json'):
    dataset_names = ["RTTS", "snowy"]
    epoch = 100
    for dataset_name in dataset_names:
        yolo_train(epoch, dataset_name)



if __name__ == "__main__":

    # prepare_env_data()
    # train_env_model()
    # train_snowy_model()
    # VR_data_process()

    # run_data_figure1_1()
    # run_data_figure1_2()
    # run_data_figure1_3()
    # run_data_figure1_4()
    # run_data_figure2_1()
    # run_data_figure2_2()

    # save_data_figure1_1()
    # save_data_figure1_2()
    # save_data_figure1_3()
    # save_overall_data()
    # save_data_figure1_4()
    # save_data_figure2_1()
    # save_data_figure2_2()
    pass