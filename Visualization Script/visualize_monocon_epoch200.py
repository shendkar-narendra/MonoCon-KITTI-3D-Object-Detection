import os
import cv2
import mmcv
import numpy as np
import torch


# ============================================================
# MONOCON EPOCH-200 VISUALIZATION V9
#
# OUTPUTS:
#
# 01_Image
# 02_Ground_Truth
# 03_GT_and_Prediction
# 04_BEV_Combined
# 05_BEV_GroundTruth_Only
# 06_BEV_Prediction_Only
#
# BEV:
#
# BLACK BACKGROUND
#
# Combined:
#   GT = GREEN outline
#   Prediction = RED outline
#   X/Y axes + 5 m grid
#
# GT-only:
#   FILLED GREEN boxes
#   no axes
#   no grid
#   no text
#
# Prediction-only:
#   FILLED RED boxes
#   no axes
#   no grid
#   no text
# ============================================================


ROOT = r"D:\MonoCon\mmdetection3d-0.14.0"


# ============================================================
# INPUT PATHS
# ============================================================

RESULT_FILE = os.path.join(
    ROOT,
    "work_dirs",
    "monocon_dla34_windows_train",
    "monocon_epoch200_results.pkl"
)

INFO_FILE = os.path.join(
    ROOT,
    "data",
    "kitti",
    "kitti_infos_val.pkl"
)

IMAGE_DIR = os.path.join(
    ROOT,
    "data",
    "kitti",
    "training",
    "image_2"
)

LABEL_DIR = os.path.join(
    ROOT,
    "data",
    "kitti",
    "training",
    "label_2"
)

CALIB_DIR = os.path.join(
    ROOT,
    "data",
    "kitti",
    "training",
    "calib"
)


# ============================================================
# OUTPUT V9
# ============================================================

OUTPUT_ROOT = os.path.join(
    ROOT,
    "work_dirs",
    "monocon_dla34_windows_train",
    "monocon_epoch200_visualization_v9"
)


IMAGE_OUTPUT = os.path.join(
    OUTPUT_ROOT,
    "01_Image"
)

GT_OUTPUT = os.path.join(
    OUTPUT_ROOT,
    "02_Ground_Truth"
)

COMBINED_OUTPUT = os.path.join(
    OUTPUT_ROOT,
    "03_GT_and_Prediction"
)

BEV_COMBINED_OUTPUT = os.path.join(
    OUTPUT_ROOT,
    "04_BEV_Combined"
)

BEV_GT_OUTPUT = os.path.join(
    OUTPUT_ROOT,
    "05_BEV_GroundTruth_Only"
)

BEV_PRED_OUTPUT = os.path.join(
    OUTPUT_ROOT,
    "06_BEV_Prediction_Only"
)


for folder in [
    IMAGE_OUTPUT,
    GT_OUTPUT,
    COMBINED_OUTPUT,
    BEV_COMBINED_OUTPUT,
    BEV_GT_OUTPUT,
    BEV_PRED_OUTPUT
]:

    os.makedirs(
        folder,
        exist_ok=True
    )


# ============================================================
# SETTINGS
# ============================================================

# Test 100 first.
#
# Use:
# NUM_IMAGES = None
#
# for all 3769 validation images.

NUM_IMAGES = 100


# ============================================================
# KITTI IoU THRESHOLDS
# ============================================================

KITTI_IOU_THRESHOLDS = {
    "Car": 0.70,
    "Pedestrian": 0.50,
    "Cyclist": 0.50
}


CLASS_NAMES = [
    "Pedestrian",
    "Cyclist",
    "Car"
]


# ============================================================
# COLORS - OpenCV BGR
# ============================================================

GT_COLOR = (
    0,
    255,
    0
)

PRED_COLOR = (
    0,
    0,
    255
)

BLACK = (
    0,
    0,
    0
)

WHITE = (
    255,
    255,
    255
)

GRID_COLOR = (
    55,
    55,
    55
)

AXIS_COLOR = (
    220,
    220,
    220
)

TEXT_BACKGROUND = (
    20,
    20,
    20
)


# ============================================================
# CAMERA IMAGE
# ============================================================

BOX_LINE_THICKNESS = 1

BB_FONT_SCALE = 0.27

BB_FONT_THICKNESS = 1


# ============================================================
# BEV
# ============================================================

BEV_WIDTH = 900

BEV_HEIGHT = 1100


# KITTI:
#
# X = left/right
# Z = forward
#
# Display:
#
# X-axis = KITTI X
# Y-axis = KITTI Z

X_MIN = -25.0
X_MAX = 25.0

Y_MIN = 0.0
Y_MAX = 70.0


GRID_INTERVAL = 5.0


LEFT_MARGIN = 75

RIGHT_MARGIN = 35

TOP_MARGIN = 40

BOTTOM_MARGIN = 90


# ============================================================
# CALIBRATION
# ============================================================

def read_calibration(calib_file):

    P2 = None

    with open(
        calib_file,
        "r"
    ) as f:

        for line in f:

            if line.startswith("P2:"):

                values = np.array(
                    [
                        float(x)
                        for x in
                        line.strip().split()[1:]
                    ],
                    dtype=np.float32
                )

                P2 = values.reshape(
                    3,
                    4
                )

                break

    if P2 is None:

        raise RuntimeError(
            "P2 not found: " + calib_file
        )

    return P2


# ============================================================
# PROJECT 3D -> IMAGE
# ============================================================

def project_points(
    points_3d,
    P2
):

    points_3d = np.asarray(
        points_3d,
        dtype=np.float32
    )

    ones = np.ones(
        (
            len(points_3d),
            1
        ),
        dtype=np.float32
    )

    points_h = np.concatenate(
        [
            points_3d,
            ones
        ],
        axis=1
    )

    projected = np.dot(
        P2,
        points_h.T
    ).T

    depth = projected[:, 2].copy()

    valid = depth > 0.1


    projected[:, 0] /= np.maximum(
        projected[:, 2],
        1e-6
    )

    projected[:, 1] /= np.maximum(
        projected[:, 2],
        1e-6
    )


    return (
        projected[:, :2],
        valid
    )


# ============================================================
# KITTI GT BOX
# ============================================================

def create_kitti_box_corners(
    h,
    w,
    l,
    x,
    y,
    z,
    ry
):

    x_corners = np.array([
        l / 2,
        l / 2,
        -l / 2,
        -l / 2,

        l / 2,
        l / 2,
        -l / 2,
        -l / 2
    ])

    y_corners = np.array([
        0,
        0,
        0,
        0,

        -h,
        -h,
        -h,
        -h
    ])

    z_corners = np.array([
        w / 2,
        -w / 2,
        -w / 2,
        w / 2,

        w / 2,
        -w / 2,
        -w / 2,
        w / 2
    ])

    corners = np.vstack([
        x_corners,
        y_corners,
        z_corners
    ])


    rotation = np.array([

        [
            np.cos(ry),
            0,
            np.sin(ry)
        ],

        [
            0,
            1,
            0
        ],

        [
            -np.sin(ry),
            0,
            np.cos(ry)
        ]
    ])


    corners = np.dot(
        rotation,
        corners
    )

    corners[0, :] += x

    corners[1, :] += y

    corners[2, :] += z


    return corners.T


# ============================================================
# READ GROUND TRUTH
# ============================================================

def read_ground_truth(
    label_file
):

    objects = []


    if not os.path.exists(
        label_file
    ):

        return objects


    with open(
        label_file,
        "r"
    ) as f:

        for line in f:

            parts = (
                line
                .strip()
                .split()
            )


            if len(parts) < 15:
                continue


            class_name = parts[0]


            if class_name not in [
                "Car",
                "Pedestrian",
                "Cyclist"
            ]:

                continue


            h = float(
                parts[8]
            )

            w = float(
                parts[9]
            )

            l = float(
                parts[10]
            )

            x = float(
                parts[11]
            )

            y = float(
                parts[12]
            )

            z = float(
                parts[13]
            )

            ry = float(
                parts[14]
            )


            corners = create_kitti_box_corners(
                h,
                w,
                l,
                x,
                y,
                z,
                ry
            )


            objects.append({

                "class":
                    class_name,

                "corners":
                    corners,

                "matched":
                    False
            })


    return objects


# ============================================================
# CAMERA BOX EDGES
# ============================================================

BOX_EDGES = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),

    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),

    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7)
]


# ============================================================
# CLEAR TEXT
# ============================================================

def draw_clear_text(
    image,
    text,
    position,
    scale,
    color
):

    x, y = position


    font = cv2.FONT_HERSHEY_SIMPLEX

    thickness = 1


    size, baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )


    text_width = size[0]

    text_height = size[1]

    padding = 2


    x1 = max(
        0,
        x - padding
    )


    y1 = max(
        0,
        y - text_height - padding
    )


    x2 = min(
        image.shape[1] - 1,
        x + text_width + padding
    )


    y2 = min(
        image.shape[0] - 1,
        y + baseline + padding
    )


    cv2.rectangle(
        image,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        TEXT_BACKGROUND,
        -1
    )


    cv2.putText(
        image,
        text,
        (
            x,
            y
        ),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# CAMERA 3D BOX
# ============================================================

def draw_projected_box(
    image,
    corners,
    P2,
    color,
    text=None
):

    points_2d, valid = project_points(
        corners,
        P2
    )


    if np.sum(
        valid
    ) < 4:

        return image


    points_2d = points_2d.astype(
        np.int32
    )


    for start, end in BOX_EDGES:

        if not valid[start]:

            continue

        if not valid[end]:

            continue


        cv2.line(
            image,
            tuple(
                points_2d[
                    start
                ]
            ),
            tuple(
                points_2d[
                    end
                ]
            ),
            color,
            BOX_LINE_THICKNESS,
            cv2.LINE_AA
        )


    if text is not None:

        valid_points = points_2d[
            valid
        ]


        x = int(
            np.min(
                valid_points[:, 0]
            )
        )


        y = int(
            np.min(
                valid_points[:, 1]
            )
        )


        y = max(
            10,
            y - 3
        )


        draw_clear_text(
            image,
            text,
            (
                x,
                y
            ),
            BB_FONT_SCALE,
            color
        )


    return image


# ============================================================
# EXTRACT PREDICTIONS
# ============================================================

def extract_predictions(
    result
):

    if not isinstance(
        result,
        dict
    ):

        raise TypeError(
            "Unexpected result type"
        )


    if "img_bbox" in result:

        pred = result[
            "img_bbox"
        ]


    elif "pts_bbox" in result:

        pred = result[
            "pts_bbox"
        ]


    else:

        pred = result


    boxes = pred.get(
        "boxes_3d",
        None
    )

    scores = pred.get(
        "scores_3d",
        None
    )

    labels = pred.get(
        "labels_3d",
        None
    )


    if boxes is None:

        return []


    if len(boxes) == 0:

        return []


    if scores is None:

        return []


    if labels is None:

        return []


    if torch.is_tensor(
        scores
    ):

        scores = (
            scores
            .detach()
            .cpu()
            .numpy()
        )

    else:

        scores = np.asarray(
            scores
        )


    if torch.is_tensor(
        labels
    ):

        labels = (
            labels
            .detach()
            .cpu()
            .numpy()
        )

    else:

        labels = np.asarray(
            labels
        )


    corners = boxes.corners


    if torch.is_tensor(
        corners
    ):

        corners = (
            corners
            .detach()
            .cpu()
            .numpy()
        )

    else:

        corners = np.asarray(
            corners
        )


    predictions = []


    for i in range(
        len(scores)
    ):

        label = int(
            labels[i]
        )


        if label < 0:
            continue


        if label >= len(
            CLASS_NAMES
        ):
            continue


        predictions.append({

            "class":
                CLASS_NAMES[
                    label
                ],

            "score":
                float(
                    scores[i]
                ),

            "corners":
                corners[i],

            "iou":
                0.0,

            "status":
                "FP"
        })


    return predictions


# ============================================================
# IMAGE ID
# ============================================================

def get_image_id(
    info
):

    if "image" in info:

        image_info = info[
            "image"
        ]


        if "image_idx" in image_info:

            return str(
                image_info[
                    "image_idx"
                ]
            ).zfill(6)


        if "image_path" in image_info:

            filename = os.path.basename(
                image_info[
                    "image_path"
                ]
            )

            return os.path.splitext(
                filename
            )[0]


    if "image_idx" in info:

        return str(
            info[
                "image_idx"
            ]
        ).zfill(6)


    raise KeyError(
        "Could not determine image ID."
    )


# ============================================================
# BEV POLYGON
# ============================================================

def box_bev_polygon(
    corners
):

    points = np.asarray(
        corners[
            :,
            [
                0,
                2
            ]
        ],
        dtype=np.float32
    )


    return cv2.convexHull(
        points
    )


# ============================================================
# BEV IoU
# ============================================================

def bev_iou(
    corners_a,
    corners_b
):

    poly_a = box_bev_polygon(
        corners_a
    )


    poly_b = box_bev_polygon(
        corners_b
    )


    area_a = abs(
        cv2.contourArea(
            poly_a
        )
    )


    area_b = abs(
        cv2.contourArea(
            poly_b
        )
    )


    if area_a <= 0:
        return 0.0


    if area_b <= 0:
        return 0.0


    try:

        intersection, _ = (
            cv2.intersectConvexConvex(
                poly_a,
                poly_b
            )
        )


    except cv2.error:

        return 0.0


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union <= 0:

        return 0.0


    return float(
        intersection
        /
        union
    )


# ============================================================
# MATCH PREDICTIONS
# ============================================================

def match_predictions_to_gt(
    predictions,
    ground_truth
):

    for gt in ground_truth:

        gt[
            "matched"
        ] = False


    order = sorted(
        range(
            len(predictions)
        ),
        key=lambda i:
            predictions[i][
                "score"
            ],
        reverse=True
    )


    for pred_index in order:

        pred = predictions[
            pred_index
        ]


        best_iou = 0.0

        best_gt_index = None


        for gt_index, gt in enumerate(
            ground_truth
        ):


            if (
                gt[
                    "class"
                ]
                !=
                pred[
                    "class"
                ]
            ):

                continue


            if gt[
                "matched"
            ]:

                continue


            iou = bev_iou(
                pred[
                    "corners"
                ],
                gt[
                    "corners"
                ]
            )


            if iou > best_iou:

                best_iou = iou

                best_gt_index = (
                    gt_index
                )


        pred[
            "iou"
        ] = best_iou


        threshold = (
            KITTI_IOU_THRESHOLDS[
                pred[
                    "class"
                ]
            ]
        )


        if (
            best_gt_index is not None

            and

            best_iou >= threshold
        ):

            pred[
                "status"
            ] = "TP"


            ground_truth[
                best_gt_index
            ][
                "matched"
            ] = True


        else:

            pred[
                "status"
            ] = "FP"


    return (
        predictions,
        ground_truth
    )


# ============================================================
# BEV COORDINATES
# ============================================================

def bev_point(
    x,
    y
):

    usable_width = (
        BEV_WIDTH
        -
        LEFT_MARGIN
        -
        RIGHT_MARGIN
    )


    usable_height = (
        BEV_HEIGHT
        -
        TOP_MARGIN
        -
        BOTTOM_MARGIN
    )


    px = int(
        LEFT_MARGIN
        +
        (
            (
                x
                -
                X_MIN
            )
            /
            (
                X_MAX
                -
                X_MIN
            )
        )
        *
        usable_width
    )


    py = int(
        TOP_MARGIN
        +
        (
            1.0
            -
            (
                (
                    y
                    -
                    Y_MIN
                )
                /
                (
                    Y_MAX
                    -
                    Y_MIN
                )
            )
        )
        *
        usable_height
    )


    return (
        px,
        py
    )


# ============================================================
# DRAW AXES + GRID ON BLACK BACKGROUND
# ============================================================

def draw_axes_and_grid(
    bev
):

    # --------------------------------------------------------
    # Y grid
    # --------------------------------------------------------

    y_value = Y_MIN


    while y_value <= Y_MAX:

        left = bev_point(
            X_MIN,
            y_value
        )


        right = bev_point(
            X_MAX,
            y_value
        )


        cv2.line(
            bev,
            left,
            right,
            GRID_COLOR,
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            bev,
            "{:.0f}".format(
                y_value
            ),
            (
                LEFT_MARGIN - 35,
                left[1] + 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            AXIS_COLOR,
            1,
            cv2.LINE_AA
        )


        y_value += (
            GRID_INTERVAL
        )


    # --------------------------------------------------------
    # X grid
    # --------------------------------------------------------

    x_value = X_MIN


    while x_value <= X_MAX:

        bottom = bev_point(
            x_value,
            Y_MIN
        )


        top = bev_point(
            x_value,
            Y_MAX
        )


        cv2.line(
            bev,
            bottom,
            top,
            GRID_COLOR,
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            bev,
            "{:.0f}".format(
                x_value
            ),
            (
                bottom[0] - 7,
                BEV_HEIGHT - 55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            AXIS_COLOR,
            1,
            cv2.LINE_AA
        )


        x_value += (
            GRID_INTERVAL
        )


    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    x_left = bev_point(
        X_MIN,
        Y_MIN
    )


    x_right = bev_point(
        X_MAX,
        Y_MIN
    )


    cv2.line(
        bev,
        x_left,
        x_right,
        WHITE,
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Y axis on extreme left
    # --------------------------------------------------------

    y_bottom = bev_point(
        X_MIN,
        Y_MIN
    )


    y_top = bev_point(
        X_MIN,
        Y_MAX
    )


    cv2.line(
        bev,
        y_bottom,
        y_top,
        WHITE,
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    cv2.putText(
        bev,
        "X - lateral position (m)",
        (
            int(
                BEV_WIDTH / 2
            )
            -
            80,
            BEV_HEIGHT - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1,
        cv2.LINE_AA
    )


    # Y-axis title

    label_canvas = np.zeros(
        (
            35,
            230,
            3
        ),
        dtype=np.uint8
    )


    cv2.putText(
        label_canvas,
        "Y - forward distance (m)",
        (
            3,
            23
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1,
        cv2.LINE_AA
    )


    rotated = cv2.rotate(
        label_canvas,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )


    h = rotated.shape[0]

    w = rotated.shape[1]


    y0 = int(
        (
            BEV_HEIGHT
            -
            h
        )
        /
        2
    )


    x0 = 2


    bev[
        y0:y0+h,
        x0:x0+w
    ] = rotated


# ============================================================
# CONVERT BOX TO BEV HULL
# ============================================================

def get_bev_hull(
    corners
):

    points = corners[
        :,
        [
            0,
            2
        ]
    ]


    pixel_points = []


    for x, z in points:

        px, py = bev_point(
            float(
                x
            ),
            float(
                z
            )
        )


        pixel_points.append(
            [
                px,
                py
            ]
        )


    pixel_points = np.asarray(
        pixel_points,
        dtype=np.int32
    )


    hull = cv2.convexHull(
        pixel_points
    )


    return hull


# ============================================================
# DRAW OUTLINE BEV BOX
# ============================================================

def draw_bev_outline(
    bev,
    corners,
    color,
    thickness=2
):

    hull = get_bev_hull(
        corners
    )


    cv2.polylines(
        bev,
        [
            hull
        ],
        True,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW FILLED BEV BOX
# ============================================================

def draw_bev_filled(
    bev,
    corners,
    color
):

    hull = get_bev_hull(
        corners
    )


    cv2.fillPoly(
        bev,
        [
            hull
        ],
        color,
        lineType=cv2.LINE_AA
    )


    # Slight outline around same filled object
    cv2.polylines(
        bev,
        [
            hull
        ],
        True,
        color,
        1,
        cv2.LINE_AA
    )


# ============================================================
# COMBINED BEV
#
# BLACK BACKGROUND
# OUTLINES ONLY
# ============================================================

def make_combined_bev(
    ground_truth,
    predictions,
    image_id
):

    bev = np.zeros(
        (
            BEV_HEIGHT,
            BEV_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    draw_axes_and_grid(
        bev
    )


    # GT = GREEN OUTLINE

    for gt in ground_truth:

        draw_bev_outline(
            bev,
            gt[
                "corners"
            ],
            GT_COLOR,
            2
        )


    # Prediction = RED OUTLINE

    for pred in predictions:

        draw_bev_outline(
            bev,
            pred[
                "corners"
            ],
            PRED_COLOR,
            2
        )


    # --------------------------------------------------------
    # Ego
    # --------------------------------------------------------

    ego_x, ego_y = bev_point(
        0,
        0
    )


    cv2.circle(
        bev,
        (
            ego_x,
            ego_y
        ),
        5,
        WHITE,
        -1
    )


    cv2.putText(
        bev,
        "Ego",
        (
            ego_x + 7,
            ego_y - 6
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        WHITE,
        1,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    cv2.rectangle(
        bev,
        (
            BEV_WIDTH - 205,
            15
        ),
        (
            BEV_WIDTH - 190,
            30
        ),
        GT_COLOR,
        2
    )


    cv2.putText(
        bev,
        "Ground Truth",
        (
            BEV_WIDTH - 183,
            28
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        WHITE,
        1,
        cv2.LINE_AA
    )


    cv2.rectangle(
        bev,
        (
            BEV_WIDTH - 205,
            40
        ),
        (
            BEV_WIDTH - 190,
            55
        ),
        PRED_COLOR,
        2
    )


    cv2.putText(
        bev,
        "Prediction",
        (
            BEV_WIDTH - 183,
            53
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        WHITE,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        bev,
        "Frame: " + image_id,
        (
            BEV_WIDTH - 205,
            78
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        WHITE,
        1,
        cv2.LINE_AA
    )


    return bev


# ============================================================
# GT-ONLY BEV
#
# BLACK BACKGROUND
# FILLED GREEN BOXES
# NO AXIS
# NO GRID
# NO TEXT
# ============================================================

def make_gt_only_bev(
    ground_truth
):

    bev = np.zeros(
        (
            BEV_HEIGHT,
            BEV_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    for gt in ground_truth:

        draw_bev_filled(
            bev,
            gt[
                "corners"
            ],
            GT_COLOR
        )


    return bev


# ============================================================
# PREDICTION-ONLY BEV
#
# BLACK BACKGROUND
# FILLED RED BOXES
# NO AXIS
# NO GRID
# NO TEXT
# ============================================================

def make_prediction_only_bev(
    predictions
):

    bev = np.zeros(
        (
            BEV_HEIGHT,
            BEV_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    for pred in predictions:

        draw_bev_filled(
            bev,
            pred[
                "corners"
            ],
            PRED_COLOR
        )


    return bev


# ============================================================
# LOAD DATA
# ============================================================

print()

print(
    "============================================"
)

print(
    "Loading MonoCon epoch-200 predictions"
)

print(
    "============================================"
)


results = mmcv.load(
    RESULT_FILE
)


print(
    "Prediction entries:",
    len(
        results
    )
)


print()

print(
    "Loading KITTI validation information..."
)


infos = mmcv.load(
    INFO_FILE
)


print(
    "Validation images:",
    len(
        infos
    )
)


# ============================================================
# NUMBER OF FRAMES
# ============================================================

total = min(
    len(
        results
    ),
    len(
        infos
    )
)


if NUM_IMAGES is not None:

    total = min(
        total,
        NUM_IMAGES
    )


print()

print(
    "Visualizing {} frames...".format(
        total
    )
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

for index in range(
    total
):

    info = infos[
        index
    ]


    image_id = get_image_id(
        info
    )


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    image_file = os.path.join(
        IMAGE_DIR,
        image_id + ".png"
    )


    label_file = os.path.join(
        LABEL_DIR,
        image_id + ".txt"
    )


    calib_file = os.path.join(
        CALIB_DIR,
        image_id + ".txt"
    )


    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image = cv2.imread(
        image_file
    )


    if image is None:

        print(
            "Missing image:",
            image_file
        )

        continue


    original = image.copy()


    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    P2 = read_calibration(
        calib_file
    )


    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    ground_truth = read_ground_truth(
        label_file
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = extract_predictions(
        results[
            index
        ]
    )


    # --------------------------------------------------------
    # IoU matching
    # --------------------------------------------------------

    predictions, ground_truth = (
        match_predictions_to_gt(
            predictions,
            ground_truth
        )
    )


    # ========================================================
    # 01 ORIGINAL IMAGE
    # ========================================================

    cv2.imwrite(
        os.path.join(
            IMAGE_OUTPUT,
            image_id + ".png"
        ),
        original
    )


    # ========================================================
    # 02 GROUND TRUTH CAMERA IMAGE
    # ========================================================

    gt_image = image.copy()


    for gt in ground_truth:

        gt_image = draw_projected_box(
            gt_image,
            gt[
                "corners"
            ],
            P2,
            GT_COLOR,
            "GT "
            +
            gt[
                "class"
            ]
        )


    cv2.imwrite(
        os.path.join(
            GT_OUTPUT,
            image_id + ".png"
        ),
        gt_image
    )


    # ========================================================
    # 03 GT + PREDICTION CAMERA IMAGE
    # ========================================================

    combined = image.copy()


    # GT = GREEN

    for gt in ground_truth:

        combined = draw_projected_box(
            combined,
            gt[
                "corners"
            ],
            P2,
            GT_COLOR,
            None
        )


    # Prediction = RED

    for pred in predictions:

        text = (
            "{} {:.2f} IoU:{:.2f}".format(
                pred[
                    "class"
                ],
                pred[
                    "score"
                ],
                pred[
                    "iou"
                ]
            )
        )


        combined = draw_projected_box(
            combined,
            pred[
                "corners"
            ],
            P2,
            PRED_COLOR,
            text
        )


    cv2.imwrite(
        os.path.join(
            COMBINED_OUTPUT,
            image_id + ".png"
        ),
        combined
    )


    # ========================================================
    # 04 COMBINED BEV
    # ========================================================

    combined_bev = make_combined_bev(
        ground_truth,
        predictions,
        image_id
    )


    cv2.imwrite(
        os.path.join(
            BEV_COMBINED_OUTPUT,
            image_id + ".png"
        ),
        combined_bev
    )


    # ========================================================
    # 05 GT-ONLY FILLED BEV
    # ========================================================

    gt_bev = make_gt_only_bev(
        ground_truth
    )


    cv2.imwrite(
        os.path.join(
            BEV_GT_OUTPUT,
            image_id + ".png"
        ),
        gt_bev
    )


    # ========================================================
    # 06 PREDICTION-ONLY FILLED BEV
    # ========================================================

    pred_bev = make_prediction_only_bev(
        predictions
    )


    cv2.imwrite(
        os.path.join(
            BEV_PRED_OUTPUT,
            image_id + ".png"
        ),
        pred_bev
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    tp = sum(
        1
        for p in predictions
        if p[
            "status"
        ]
        ==
        "TP"
    )


    fp = sum(
        1
        for p in predictions
        if p[
            "status"
        ]
        ==
        "FP"
    )


    fn = sum(
        1
        for gt in ground_truth
        if not gt[
            "matched"
        ]
    )


    print(
        "[{}/{}] Frame {} | GT:{} | Pred:{} | TP:{} | FP:{} | FN:{}".format(
            index + 1,
            total,
            image_id,
            len(
                ground_truth
            ),
            len(
                predictions
            ),
            tp,
            fp,
            fn
        )
    )


# ============================================================
# FINISHED
# ============================================================

print()

print(
    "============================================"
)

print(
    "MONOCON V9 VISUALIZATION COMPLETE"
)

print(
    "============================================"
)

print()

print(
    "Output folder:"
)

print(
    OUTPUT_ROOT
)

print()

print(
    "04_BEV_Combined:"
)

print(
    "Black background"
)

print(
    "Green GT outline"
)

print(
    "Red prediction outline"
)

print(
    "Axes + 5m grid"
)

print()

print(
    "05_BEV_GroundTruth_Only:"
)

print(
    "Black background"
)

print(
    "FILLED GREEN boxes"
)

print(
    "No axes / grid / text"
)

print()

print(
    "06_BEV_Prediction_Only:"
)

print(
    "Black background"
)

print(
    "FILLED RED boxes"
)

print(
    "No axes / grid / text"
)

print()

print(
    "Coordinate range:"
)

print(
    "X = -25 m to +25 m"
)

print(
    "Forward distance = 0 m to 70 m"
)

print()
