from backend.yolo.yolo_tag_eval import evaluate_yolo_tag_conditions


def test_orientation_kind_rule_item():
    detections = [
        {"label": "box", "xyxy": [10, 20, 110, 50]},
    ]
    conditions = [
        {
            "id": "t0",
            "enabled": True,
            "tag_name": "wide-orient",
            "rule_items": [{"kind": "orientation", "orientation": "horizontal"}],
        }
    ]
    assert evaluate_yolo_tag_conditions(detections, conditions) == ["wide-orient"]


def test_orientation_horizontal_tag():
    detections = [
        {"label": "box", "xyxy": [10, 20, 110, 50]},
    ]
    conditions = [
        {
            "id": "t1",
            "enabled": True,
            "tag_name": "wide-box",
            "rule_items": [
                {"kind": "class", "text": "box"},
                {"kind": "size", "orientation": "horizontal"},
            ],
        }
    ]
    assert evaluate_yolo_tag_conditions(detections, conditions) == ["wide-box"]


def test_target_label_exact_match():
    detections = [
        {"label": "small-box", "xyxy": [0, 0, 80, 60]},
    ]
    conditions = [
        {
            "id": "t1",
            "enabled": True,
            "tag_name": "box-target",
            "rule_items": [{"kind": "target", "labels": ["small-box"]}],
        }
    ]
    assert evaluate_yolo_tag_conditions(detections, conditions) == ["box-target"]


def test_rule_items_and_across_multiple_classes():
    detections = [
        {"label": "green-small-box", "xyxy": [0, 0, 120, 40]},
        {"label": "green-box", "xyxy": [0, 0, 120, 40]},
    ]
    conditions = [
        {
            "id": "t3",
            "enabled": True,
            "tag_name": "strict",
            "rule_items": [
                {"kind": "class", "text": "green"},
                {"kind": "class", "text": "small"},
                {"kind": "size", "orientation": "horizontal"},
            ],
        }
    ]
    assert evaluate_yolo_tag_conditions(detections, conditions) == ["strict"]


def test_region_overlap_percent():
    detections = [
        {"label": "item", "xyxy": [0, 0, 100, 100]},
    ]
    conditions = [
        {
            "id": "t2",
            "enabled": True,
            "tag_name": "in-corner",
            "rule_items": [
                {
                    "kind": "region",
                    "x1": 0,
                    "y1": 0,
                    "x2": 50,
                    "y2": 50,
                    "min_overlap_pct": 20,
                },
            ],
        }
    ]
    assert evaluate_yolo_tag_conditions(detections, conditions) == ["in-corner"]