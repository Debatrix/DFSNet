import cv2
import numpy as np


def fmeasure(img, measure='GRAS', roi=None):
    if roi is not None:
        img = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    img = img.astype(np.double)

    if measure == 'GRAT':
        # >5.5
        Th = 0
        img_x = img.copy()
        img_y = img.copy()
        img_y[:-1, :] = np.diff(img, axis=0)
        img_x[:, :-1] = np.diff(img, axis=1)
        fm = np.maximum(np.abs(img_x), np.abs(img_y))
        fm[fm < Th] = 0
        fm = np.sum(fm / np.sum(np.sum(fm != 0)))

    elif measure == 'GLVN':
        # < 17
        fm = np.std(img)**2 / np.mean(img)

    elif measure == 'GRAE':
        img_x = img.copy()
        img_y = img.copy()
        img_y[:-1, :] = np.diff(img, axis=0)
        img_x[:, :-1] = np.diff(img, axis=1)
        fm = np.mean(img_x**2 + img_y**2)

    elif measure == 'GRAS':
        # >20.5
        img_x = np.diff(img, axis=1)
        img_x[img_x < 0] = 0
        fm = img_x**2
        fm = np.mean(fm)

    elif measure == 'LAPV':
        """
        Implements the Variance of Laplacian (LAP4) focus measure
        operator. Measures the amount of edges present in the image.
        """
        fm = np.std(cv2.Laplacian(img, cv2.CV_64F))**2

    elif measure == "LAPM":
        """
        Implements the Modified Laplacian (LAP2) focus measure
        operator. Measures the amount of edges present in the image.
        """
        kernel = np.array([-1, 2, -1])
        laplacianX = np.abs(cv2.filter2D(img, -1, kernel))
        laplacianY = np.abs(cv2.filter2D(img, -1, kernel.T))
        fm = np.mean(laplacianX + laplacianY)

    elif measure == "TENG":
        """
        Implements the Tenengrad (TENG) focus measure operator.
        Based on the gradient of the image.
        """
        gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        fm = np.mean(gaussianX**2 + gaussianY**2)

    elif measure == "TENV":
        gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        fm = np.std(gaussianX**2 + gaussianY**2)**2

    elif measure == 'SFRQ':
        img_x = img.copy()
        img_y = img.copy()
        img_y[:-1, :] = np.diff(img, axis=0)
        img_x[:, :-1] = np.diff(img, axis=1)
        fm = np.mean(np.sqrt(img_x**2 + img_y**2))

    elif measure == 'ISO':
        kernel = np.array([
            [0, 1, 1, 2, 2, 2, 1, 1, 0],
            [1, 2, 4, 5, 5, 5, 4, 2, 1],
            [1, 4, 5, 3, 0, 3, 5, 4, 1],
            [2, 5, 3, -12, -24, -12, 3, 5, 2],
            [2, 5, 0, -24, -40, -24, 0, 5, 2],
            [2, 5, 3, -12, -24, -12, 3, 5, 2],
            [1, 4, 5, 3, 0, 3, 5, 4, 1],
            [1, 2, 4, 5, 5, 5, 4, 2, 1],
            [0, 1, 1, 2, 2, 2, 1, 1, 0],
        ])
        fm = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        fm = np.mean(fm * fm)
        fm = int(np.round(fm))
        # fm = 100 * fm * fm / (fm * fm + 1800000)

    else:
        raise ValueError

    return fm


if __name__ == "__main__":
    pass
