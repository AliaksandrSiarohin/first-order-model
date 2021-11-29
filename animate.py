import os
from tqdm import tqdm
import copy

import face_alignment
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np
from PIL import Image

from sync_batchnorm import DataParallelWithCallback


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def animate(config, generator, kp_detector, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]

            kp_source = kp_detector(source_frame)
            kp_driving_initial = kp_detector(driving_video[:, :, 0])

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                out['kp_driving'] = kp_driving
                out['kp_source'] = kp_source
                out['kp_norm'] = kp_norm

                del out['sparse_deformed']

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame,
                                                                                    driving=driving_frame, out=out)
                visualization = visualization
                visualizations.append(visualization)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

            
def animate_and_compare(config, generator, kp_detector, checkpoints, log_dir, dataset, kp=False):
    log_dir = os.path.join(log_dir, 'compare_animation')
    png_dir = os.path.join(log_dir, 'compare_png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'] * 2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    generator_new = copy.deepcopy(generator)
    kp_detector_new = copy.deepcopy(kp_detector)
    if checkpoints is not None and len(checkpoints) == 2:
        Logger.load_cpk(checkpoints[0], generator=generator, kp_detector=kp_detector)
        Logger.load_cpk(checkpoints[1], generator=generator_new, kp_detector=kp_detector_new)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate' and consist of 2 checkpoints to compare.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        generator_new = DataParallelWithCallback(generator_new)
        kp_detector_new = DataParallelWithCallback(kp_detector_new)

    generator.eval()
    kp_detector.eval()
    generator_new.eval()
    kp_detector_new.eval()
    
    resizer = transforms.Resize((256, 256))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
                

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]

            kp_source = kp_detector(source_frame)
            kp_driving_initial = kp_detector(driving_video[:, :, 0])

            kp_source_new = kp_detector_new(source_frame)
            kp_driving_initial_new = kp_detector_new(driving_video[:, :, 0])

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                
                # Keypoints
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])

                kp_driving_new = kp_detector_new(driving_frame)
                kp_norm_new = normalize_kp(kp_source=kp_source_new, kp_driving=kp_driving_new,
                                       kp_driving_initial=kp_driving_initial_new, **animate_params['normalization_params'])

                # Predictions
                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)
                pred = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

                out_new = generator_new(source_frame, kp_source=kp_source_new, kp_driving=kp_norm_new)
                pred_new = np.transpose(out_new['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                
                # Getting landmarks and cropping
                lm = np.array(fa.get_landmarks(pred * 255))
#                 import pdb; pdb.set_trace()
                x_min, y_min = lm.min(axis=1)[0].astype(int)
                x_max, y_max = lm.max(axis=1)[0].astype(int)
                pred_crop = resizer(Image.fromarray(np.uint8(255 * pred[y_min: y_max, x_min: x_max])))
                
                lm_new = np.array(fa.get_landmarks(pred_new * 255))
                x_min, y_min = lm_new.min(axis=1)[0].astype(int)
                x_max, y_max = lm_new.max(axis=1)[0].astype(int)
                pred_crop_new = resizer(Image.fromarray(np.uint8(255 * pred_new[y_min: y_max, x_min: x_max])))
                
                predictions.append(np.concatenate([pred_crop, pred_crop_new], axis=0))
                
                # Collecting
                source_img = np.transpose(source_frame.cpu().numpy(), [0, 2, 3, 1])[0]
                driving_img = np.transpose(driving_frame.cpu().numpy(), [0, 2, 3, 1])[0]
                driving_img[:, -1] = np.array((1., 1., 1.))

                visualization = np.concatenate([255 * source_img, 255 * driving_img, 255 * pred, 255 * pred_new, pred_crop, pred_crop_new], axis=1)
                visualizations.append(visualization)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            imageio.imsave(os.path.join(png_dir, result_name + '.png'), (predictions))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
