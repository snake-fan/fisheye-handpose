import numpy as np

from fisheye_handpose.qa import EpipolarQaConfig, summarize_rectified_correspondences


def _projection_pair():
    focal = 500.0
    p1 = np.array([[focal, 0.0, 800.0, 0.0], [0.0, focal, 650.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    p2 = p1.copy()
    p2[0, 3] = -focal * 0.12
    return p1, p2


def test_correspondence_summary_checks_rows_disparity_and_positive_depth():
    p1, p2 = _projection_pair()
    x = np.linspace(250.0, 1250.0, 80)
    y = np.linspace(300.0, 900.0, 80)
    left = np.column_stack((x, y))
    right = np.column_stack((x - 60.0, y + 0.1))

    report = summarize_rectified_correspondences(
        p1,
        p2,
        left,
        right,
        config=EpipolarQaConfig(min_total_inliers=20),
    )

    assert report.status == "PASS"
    assert report.expected_disparity_sign == 1
    assert report.expected_disparity_fraction == 1.0
    assert report.positive_depth_fraction == 1.0


def test_correspondence_summary_rejects_bad_rows_and_wrong_disparity_sign():
    p1, p2 = _projection_pair()
    x = np.linspace(250.0, 1250.0, 80)
    y = np.linspace(300.0, 900.0, 80)
    left = np.column_stack((x, y))
    right = np.column_stack((x + 60.0, y + 5.0))

    report = summarize_rectified_correspondences(
        p1,
        p2,
        left,
        right,
        config=EpipolarQaConfig(min_total_inliers=20),
    )

    assert report.status == "FAIL"
    assert report.expected_disparity_fraction == 0.0
    assert report.positive_depth_fraction == 0.0
    assert len(report.failures) >= 3


def test_correspondence_summary_is_inconclusive_without_enough_evidence():
    p1, p2 = _projection_pair()
    points = np.zeros((7, 2), dtype=np.float64)

    report = summarize_rectified_correspondences(
        p1,
        p2,
        points,
        points,
        config=EpipolarQaConfig(min_total_inliers=8),
    )

    assert report.status == "INCONCLUSIVE"
    assert report.inlier_count == 7
    assert report.reason is not None
