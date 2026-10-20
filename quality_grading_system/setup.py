from setuptools import setup


PACKAGE_NAME = "quality_grading_system"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    py_modules=[
        "apple_quality_dataset",
        "conveyor_camera_adapter_node",
        "depth_geometry",
        "inspection_session",
        # 착색 기준 판정 경로.
        "opencv_color_predictor",
        "opencv_size_grader",
        "predictor",
        "quality_inspection_node",
        "quality_rules",
        "train_quality_model",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TBD",
    maintainer_email="tbd@example.com",
    description="GPU PC 2 quality-inspection ROS 2 node.",
    license="TBD",
    entry_points={
        "console_scripts": [
            "inspect_apple_dataset = apple_quality_dataset:main",
            "inspect_apple_size = opencv_size_grader:main",
            "conveyor_camera_adapter_node = conveyor_camera_adapter_node:main",
            "quality_inspection_node = quality_inspection_node:main",
            "train_quality_model = train_quality_model:main",
        ],
    },
)
