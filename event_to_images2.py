import torch
from e2vid_utils.utils.loading_utils import load_E2VID, get_device
import numpy as np
import argparse
import pandas as pd
import os
from e2vid_utils.utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from e2vid_utils.utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
import time
from e2vid_utils.utils.image_reconstructor import ImageReconstructor

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=False, type=str,
                        help='path to model weights', default='e2vid_utils/pretrained/E2VID_lightweight.pth.tar')
    parser.add_argument('-i', '--input_file', required=False, type=str, default='data/Event_ReId/')
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=True)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=True)
    parser.add_argument('-o', '--output_folder', default='event_to_image/', type=str)  # if None, will not write the images to disk
    parser.add_argument('--dataset_name', default='reconstruction5', type=str)

    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=True)

    """ Display """
    parser.add_argument('--display', dest='display', action='store_true')
    parser.set_defaults(display=True)  # !!!!

    parser.add_argument('--show_events', dest='show_events', action='store_true')
    parser.set_defaults(show_events=True)  # !!!!

    parser.add_argument('--event_display_mode', default='red-blue', type=str,
                        help="Event display mode ('red-blue' or 'grayscale')")

    parser.add_argument('--num_bins_to_show', default=-1, type=int,
                        help="Number of bins of the voxel grid to show when displaying events (-1 means show all the bins).")

    parser.add_argument('--display_border_crop', default=0, type=int,
                        help="Remove the outer border of size display_border_crop before displaying image.")

    parser.add_argument('--display_wait_time', default=1, type=int,
                        help="Time to wait after each call to cv2.imshow, in milliseconds (default: 1)")

    """ Post-processing / filtering """

    # (optional) path to a text file containing the locations of hot pixels to ignore
    parser.add_argument('--hot_pixels_file', default=None, type=str)

    # (optional) unsharp mask
    parser.add_argument('--unsharp_mask_amount', default=0.3, type=float)
    parser.add_argument('--unsharp_mask_sigma', default=1.0, type=float)

    # (optional) bilateral filter
    parser.add_argument('--bilateral_filter_sigma', default=0.0, type=float)

    # (optional) flip the event tensors vertically
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.set_defaults(flip=False)

    """ Tone mapping (i.e. rescaling of the image intensities)"""
    parser.add_argument('--Imin', default=0.0, type=float,
                        help="Min intensity for intensity rescaling (linear tone mapping).")
    parser.add_argument('--Imax', default=1.0, type=float,
                        help="Max intensity value for intensity rescaling (linear tone mapping).")
    parser.add_argument('--auto_hdr', dest='auto_hdr', action='store_true',
                        help="If True, will compute Imin and Imax automatically.")
    parser.set_defaults(auto_hdr=True)  # !!!
    parser.add_argument('--auto_hdr_median_filter_size', default=10, type=int,
                        help="Size of the median filter window used to smooth temporally Imin and Imax")

    """ Perform color reconstruction? (only use this flag with the DAVIS346color) """
    parser.add_argument('--color', dest='color', action='store_true')
    parser.set_defaults(color=False)

    """ Advanced parameters """
    # disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results)
    parser.add_argument('--no-normalize', dest='no_normalize', action='store_true')
    parser.set_defaults(no_normalize=False)

    # disable recurrent connection (will severely degrade the results; for testing purposes only)
    parser.add_argument('--no-recurrent', dest='no_recurrent', action='store_true')
    parser.set_defaults(no_recurrent=True)

    args = parser.parse_args()

    path_to_events = args.input_file

    header = pd.read_csv(path_to_events + '001' + '/' + 'cam01' + '/' + 'events.txt', delim_whitespace=True, header=None, names=['width', 'height'],
                         dtype={'width': np.int, 'height': np.int},
                         nrows=1)  # 第一行存的是长宽

    width = 192#header.values[0, 0]
    height = 384#header.values[0, 1]
    print('Sensor size: {} x {}'.format(width, height))

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    E2VID = load_E2VID(args.path_to_model, device)
    E2VID = E2VID.to(device)
    E2VID.eval()

    reconstructor = ImageReconstructor(E2VID, height, width, E2VID.num_bins, args)

    """ Read chunks of events using Pandas """

    N = args.window_size
    if not args.fixed_duration:
        if N is None:
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')


    if args.fixed_duration:
        event_window_iterator = {}
        for l in range(33):
            for c in ['cam01', 'cam02', 'cam03', 'cam04']:
                event_window_iterator[str(l) + c] = FixedDurationEventReader(path_to_events + str("{:03d}".format(l + 1)) + '/' + c + '/'+ 'events.txt',
                                                                             duration_ms=args.window_duration,
                                                                             start_index=start_index)
    else:
        event_window_iterator = FixedSizeEventReader(path_to_events, num_events=N, start_index=start_index)

    for l in range(33):
        for c in ['cam01', 'cam02', 'cam03', 'cam04']:
            num_events_in_window = 0
            reconstructor.initialize(height=height, width=width, options=args)
            for event_window in event_window_iterator[str(l) + c]:
                last_timestamp = event_window[-1, 0]

                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=E2VID.num_bins,
                                                        width=width,
                                                        height=height)
                    #event_tensor = torch.from_numpy(event_tensor)
                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=E2VID.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)

                num_events_in_window += 1
                reconstructor.update_reconstruction(event_tensor, event_tensor_id=str("{:03d}".format(l + 1)) + '_' + c + '_' +  str("{:03d}".format(num_events_in_window)),
                                                    stamp=last_timestamp, options=args)