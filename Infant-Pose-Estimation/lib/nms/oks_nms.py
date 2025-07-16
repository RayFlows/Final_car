import numpy as np

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    # OKS IoU computation between a ground truth and detection
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]

    xd = d[0::3]
    yd = d[1::3]

    dx = xd - xg
    dy = yd - yg
    e = (dx**2 + dy**2) / vars / ((a_g + a_d) / 2.0 + np.spacing(1)) / 2

    if in_vis_thre is not None:
        ind = vg > in_vis_thre
        e = e[ind]
    iou = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return iou

def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    Arguments:
        kpts_db: list of dicts with keys: 'keypoints' (51-dim), 'area', 'score'
        thresh: OKS threshold
        sigmas: per-keypoint std deviation
    """
    if len(kpts_db) == 0:
        return []

    if sigmas is None:
        # COCO默认17个关键点的标准偏差
        sigmas = np.array([
            .26, .25, .25, .35, .35,
            .79, .79, .72, .72, .62,
            .62, 1.07, 1.07, .87, .87,
            .89, .89
        ]) / 10.0

    scores = np.array([entry['score'] for entry in kpts_db])
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        oks_ovr = []

        for j in order[1:]:
            iou = oks_iou(
                np.array(kpts_db[i]['keypoints']),
                np.array(kpts_db[j]['keypoints']),
                kpts_db[i]['area'],
                kpts_db[j]['area'],
                sigmas,
                in_vis_thre
            )
            oks_ovr.append(iou)

        oks_ovr = np.array(oks_ovr)
        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]  # +1 because we skipped the 0-th (i)

    return keep
