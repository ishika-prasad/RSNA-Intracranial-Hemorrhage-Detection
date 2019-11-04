import numpy as np

def fix_error_img(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100:
        return dcm
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000
    return dcm


def window_correction(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        dcm = fix_error_img(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img

def image_preprocessing(img, target_size=(256, 256)):
    brain_img = window_correction(img, 40, 80)
    subdural_img = window_correction(img, 80, 200)
    soft_img = window_correction(img, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380

    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)
    bsb_img = cv2.resize(bsb_img, target_size[:2], interpolation=cv2.INTER_LINEAR)
    return bsb_img
