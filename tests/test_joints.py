from fisheye_handpose.joints import FHP21


def test_fhp21_contract_is_stable():
    assert FHP21.version == "fhp21/v1"
    assert len(FHP21.names) == 21
    assert FHP21.names[0] == "wrist_center"
    assert tuple(FHP21.names[index] for index in FHP21.tip_indices) == (
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "little_tip",
    )
    assert len(FHP21.edges) == 20
