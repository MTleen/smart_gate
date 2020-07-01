import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
import requests
import json
from flask import Flask
import base64
from threading import Thread
import logging.config
import logging

# app = Flask(__name__)
# flask_logger = logging.getLogger('werkzeug')
# flask_logger.setLevel('ERROR')
server_url = 'http://192.168.31.10:1880/gate'
api_key = '0V8w14CdUFW2q8hzG11wjVKC'
secret_key = 'mGGtPZpIx5FlkBFL2Eus3oVUaiIfq74h'
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s' % (
    api_key, secret_key)
response = requests.get(host)
access_token = response.json()['access_token']

# states = {
#     'car': {
#         'status': False,
#         'timestamp': time.time() * 1000,
#         'number': None
#     },
#     'person': {
#         'status': False,
#         'timestamp': time.time() * 1000
#     },
#     'close': {
#         'status': False,
#         'timestamp': time.time() * 1000
#     }
# }


def setup_logging(default_path="logger_config.json", default_level=logging.DEBUG):
    path = default_path
    if os.path.exists(path):
        with open(path, "r") as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def send_command(url, command, number=None):
    try:
        params = {'operation': command,
                  'number': number}
        res = requests.get(url, params=params, timeout=5)
        return res.json()
    except Exception:
        return 'nodered 请求出错，请检查。'


# @app.route('/states')
# def return_states():
#     return json.dumps(states)
def is_open(cls, pic_str):
    """
    :param pic_str:
    :param cls:
    :return: is_open, timestamp, result
    """
    if cls == 0:
        request_url = "https://aip.baidubce.com/rest/2.0/face/v3/search"
        params = "{\"image\":\"%s\",\"image_type\":\"BASE64\",\"group_id_list\":\"home\"}" % pic_str
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/json'}
        response = requests.post(request_url, data=params, headers=headers)
        result = response.json()
        if result['error_code'] == 0 and len(result['result']['user_list']) > 0:
            for user in result['result']['user_list']:
                if user['score'] > 80:
                    return True, time.time() * 1000, result
    elif cls == 2:
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate"
        params = {"image": pic_str}
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        result = response.json()
        if 'error_code' not in result.keys() and (
                result['words_result']['number'] == '浙AB259Z' or result['words_result'][
            'number'] == '浙AEQ356'):
            return True, time.time() * 1000, result
    return False, None, result


def detect(save_img=False, view_img=False):
    logging.info('##################### start detecting #####################')
    try:
        out, source, weights, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(opt.device)
        if not os.path.exists(out):
            os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        if half:
            model.half()  # to FP16
        # Set Dataloader
        if webcam:
            # view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            # save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        init_x = None
        end_x = None
        for path, img, im0s, vid_cap in dataset:
            output = out + '/' + time.strftime('%Y-%m-%d', time.localtime())
            if not os.path.exists(output):
                os.mkdir(output)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = torch_utils.time_synchronized()
            print('pred:', pred)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    logging.info('-------------------------------------------------\n%s' % det)

                    now = time.strftime('%Y-%m-%d %H.%M.%S', time.localtime())
                    if save_img:
                        save_path = os.path.join(output, now + '.jpg')
                        cv2.imwrite(save_path, im0)
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # Write results
                    for *xyxy, conf, cls in det:
                        cls = int(cls.detach().numpy())
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        if cls == 2 or cls == 0:
                            if init_x is None:
                                init_x = end_x = xyxy[0]
                            elif xyxy[0] <= init_x:
                                end_x = xyxy[0]
                                retval, buffer = cv2.imencode('.jpeg', im0)
                                pic_str = base64.b64encode(buffer).decode()
                                status, timestamp, result = is_open(cls, pic_str)
                                # states[names[cls]]['status'] = status
                                # states[names[cls]]['timestamp'] = timestamp
                                if status:
                                    logging.info('************ 开门 ***********')
                                    logging.info('result: %s' % json.dumps(result, ensure_ascii=False))
                                    # if cls == 2:
                                    # states[names[cls]]['number'] = result['words_result']['number']
                                    res_data = send_command(
                                        server_url,
                                        'open',
                                        number=result['words_result']['number'] if cls == 2 else None)
                                    logging.info('status code: %s' % json.dumps(res_data, ensure_ascii=False))
                                    time.sleep(3)
                                    init_x = end_x = None
                            else:
                                end_x = xyxy[0]
                                if init_x < 1200 and end_x - init_x > 270:
                                    logging.info('************ 关门 ************')
                                    # states['close']['status'] = True
                                    # states['close']['timestamp'] = time.time() * 1000
                                    res_data = send_command(server_url, 'close')
                                    logging.info('status code: %s' % json.dumps(res_data, ensure_ascii=False))
                                    logging.info(
                                        'init_x: %.3f, end_x: %.3f' % (init_x.detach().numpy(), end_x.detach().numpy()))
                                    init_x = end_x = None
                                    time.sleep(5)
                            # logging.info('status: %s' % json.dumps(states, ensure_ascii=False))
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # with open(txt_path + '.txt', 'a') as f:
                        #             f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                else:
                    # states['car']['status'] = False
                    # states['person']['status'] = False
                    if end_x is not None and init_x is not None and end_x > init_x and init_x < 1200 and (
                            end_x - init_x > 270 or end_x > 1200):
                        logging.info('*********** 首次从有物体转换为无物体，识别关门 ***********')
                        # states['close']['status'] = True
                        # states['close']['timestamp'] = time.time() * 1000
                        res_data = send_command(server_url, 'close')
                        logging.info('status code: %s' % json.dumps(res_data, ensure_ascii=False))
                        logging.info('init_x: %.3f, end_x: %.3f' % (init_x.detach().numpy(), end_x.detach().numpy()))
                        # logging.info('status: %s' % json.dumps(states, ensure_ascii=False))
                        time.sleep(5)
                    else:
                        #     states['close']['status'] = False
                        time.sleep(0.5)
                    init_x = end_x = None

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow(p, im0)
                    if cv2.waitKey(300) & 0xFF == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                # if save_img:
                # if dataset.mode == 'images':
                #     cv2.imwrite(save_path, im0)
                # else:
                #     if vid_path != save_path:  # new video
                #         vid_path = save_path
                #         if isinstance(vid_writer, cv2.VideoWriter):
                #             vid_writer.release()  # release previous video writer
                #
                #         fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                #     vid_writer.write(im0)

        # if save_txt or save_img:
        #     print('Results saved to %s' % os.getcwd() + os.sep + out)
        #     if platform == 'darwin':  # MacOS
        #         os.system('open ' + save_path)
        print('Done. (%.3fs)' % (time.time() - t0))
    except Exception:
        # traceback.print_exc()
        logging.exception('detect 报错')
        detect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)
    setup_logging()
    # torch.no_grad()
    # process_1 = Thread(target=app.run, kwargs={'host': '0.0.0.0'})
    # p2 = Thread(target=detect, kwargs={'save_img': True})
    # p2.start()
    # process_1.start()
    #
    with torch.no_grad():
        detect(True)