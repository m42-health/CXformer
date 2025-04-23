import pydicom
import numpy as np
import cv2, os, sys
from pydicom.pixel_data_handlers.util import apply_voi_lut
import PIL


def read_dicom(
    fpath: str,
    voi_lut: bool = True,
    fix_monochrome: bool = True,
    return_metadata: bool = False,
):
    """
    It reads a DICOM file, applies VOI LUT (if available), and returns a numpy array of shape (H, W)
    with pixel values in range [0, 255]

    Args:
      fpath (str): path to the DICOM file
      voi_lut (bool): if True, the VOI LUT (if available by DICOM device) is used to transform raw DICOM
    data to "human-friendly" view. Defaults to True
      fix_monochrome (bool): some DICOM devices may produce inverted images (black background, white
    bones) - this parameter fixes that. Defaults to True
    """

    assert os.path.exists(fpath) and os.path.isfile(
        fpath
    ), f"Expected a dicom image to be present at path: {fpath}"
    dcm = pydicom.read_file(fpath)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    try:
        if voi_lut:
            data = apply_voi_lut(dcm.pixel_array, dcm)
        else:
            data = dcm.pixel_array
    except Exception as e:
        print(e)
        print(fpath)
        return None

    if dcm.PhotometricInterpretation == "YBR_FULL_422":
        data = pydicom.pixel_data_handlers.util.convert_color_space(
            arr=data, current="YBR_FULL_422", desired="RGB"
        )
        data = np.max(data) - data

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dcm.PhotometricInterpretation == "MONOCHROME1":
        data = np.max(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    if return_metadata:
        metadata = {}
        metavals = [
            "SOPInstanceUID",
            "StudyDate",
            "StudyTime",
            "pixel_array",
            "ViewPosition",
            "BodyPartExamined",
            "Modality",
            "PhotometricInterpretation",
            "SamplesPerPixel",
            "ConversionType",
            "PatientID",
            "AccessionNumber",
            "PatientOrientation",
            "SeriesDescription",
            "ProtocolName",
            "Manufacturer",
            "InstitutionName",
        ]
        metakeys = [
            "sop_instance_uid",
            "study_date",
            "study_time",
            "study_size",
            "view_position",
            "body_part",
            "modality",
            "photometric_interpretation",
            "Samples per Pixel",
            "Conversion Type",
            "mrn_id",
            "transaction_id",
            "patient_orientation",
            "series_description",
            "protocol_name",
            "manufacturer",
            "institution_name",
        ]

        for tk, tv in zip(metakeys, metavals):
            if tv == "pixel_array":
                metadata[tk] = str(dcm.pixel_array.shape)

            elif tv in dcm:
                metadata[tk] = str(dcm[tv].value)
            else:
                metadata[tk] = "None"

        return data, metadata

    return data


def save_image(data: np.ndarray, fname: str, save_path: str):
    """
    > It takes in a numpy array, a filename, and a save path, and saves the numpy array as a jpg image
    in the save path

    Args:
      data (np.ndarray): the image data
      fname (str): The name of the image file
      save_path (str): The path where the images will be saved.
    """
    os.makedirs(save_path, exist_ok=True)
    img_savepath = os.path.join(save_path, f"{fname}.jpg")
    img = PIL.Image.fromarray(data)
    img.save(img_savepath)
