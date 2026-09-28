"""面向中英文混合科研语料的轻量级本地查询扩展。"""

from __future__ import annotations


# 仅包含领域通用术语，不绑定题号、论文ID或标准答案。
_BILINGUAL_GLOSSARY: tuple[tuple[str, str], ...] = (
    ("DeepFish", "DeepFish dataset fish instances species labels instance masks size measurements homography"),
    ("PreCNN", "PreCNN sonar Mask R-CNN environment independent feature mapping"),
    ("固定预算", "fixed annotation budget"),
    ("全监督", "fully supervised"),
    ("逐像素", "per-pixel labels"),
    ("像素级", "pixel-level masks"),
    ("完整序列", "entire video sequence same subset"),
    ("分层划分", "stratified splitting"),
    ("稀有类别", "rare object classes"),
    ("连续帧", "consecutive video frames"),
    ("误差原因", "reasons for measurement error"),
    ("介质变化", "media change image distortion"),
    ("水下标定", "underwater stereo calibration"),
    ("相机到水面的距离", "camera distance to water surface"),
    ("平均相对百分比误差", "mean relative percentage error MRPE"),
    ("历史帧阈值", "history tracking frames counting detection threshold fast fish incorrect background detections"),
    ("多目标跟踪", "multi-object tracking MOT"),
    ("对比度拉伸", "contrast stretching"),
    ("双边滤波", "bilateral filtering"),
    ("优化解码器", "optimized decoder"),
    ("水下颜色", "underwater color correction"),
    ("跨养殖环境", "cross aquaculture environment"),
    ("跨环境", "cross environment"),
    ("养殖环境", "aquaculture environment"),
    ("标注预算", "annotation budget"),
    ("点标注", "point annotation"),
    ("像素标注", "pixel-wise annotation"),
    ("弱监督", "weakly supervised"),
    ("实例分割", "instance segmentation"),
    ("物种分类", "species classification"),
    ("尺寸估计", "size measurement"),
    ("尺寸测量", "size measurement"),
    ("体长测量", "body length measurement"),
    ("立体视觉", "stereo vision"),
    ("水折射", "water refraction"),
    ("折射", "refraction"),
    ("单应性", "homography"),
    ("声呐", "sonar"),
    ("鲑鱼", "salmon"),
    ("淡水鱼", "freshwater fish"),
    ("鲈鱼", "sea bass"),
    ("欧氏距离", "Euclidean distance"),
    ("头尾关键点", "head tail keypoints"),
    ("鱼体弯曲", "fish body bending"),
    ("人工计数", "manual counting"),
    ("跨数据集", "cross dataset"),
    ("合并", "combined dataset"),
)


def expand_cross_language_query(query: str) -> tuple[str, list[str]]:
    """为命中的中文科研术语追加英文别名；无命中时保持原查询。"""
    aliases: list[str] = []
    seen: set[str] = set()

    for phrase, translation in _BILINGUAL_GLOSSARY:
        normalized = translation.casefold()
        if phrase in query and normalized not in seen:
            aliases.append(translation)
            seen.add(normalized)

    if not aliases:
        return query, []

    expanded = f"{query}\nEnglish retrieval terms: {'; '.join(aliases)}"
    return expanded, aliases
